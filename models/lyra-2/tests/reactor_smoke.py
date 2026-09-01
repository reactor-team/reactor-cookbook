"""End-to-end smoke client for a locally running Reactor Lyra-2 image."""

import argparse
import asyncio
import json
from pathlib import Path

import httpx
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription


async def main(base: str, image_path: Path) -> None:
    messages: asyncio.Queue[dict] = asyncio.Queue()
    frames = 0
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    transceiver = pc.addTransceiver("video", direction="recvonly")
    channel = pc.createDataChannel("data")
    opened = asyncio.Event()
    heartbeat_task = None

    @channel.on("open")
    def on_open() -> None:
        opened.set()

    @channel.on("message")
    def on_message(raw) -> None:
        obj = json.loads(raw)
        if obj.get("scope") == "application":
            messages.put_nowait(obj["data"])

    @pc.on("track")
    def on_track(track) -> None:
        async def consume() -> None:
            nonlocal frames
            while True:
                try:
                    await track.recv()
                    frames += 1
                except Exception:
                    return
        asyncio.create_task(consume())

    async with httpx.AsyncClient(base_url=base, timeout=60) as client:
        health = (await client.get("/health")).json()
        if health["state"] not in {"available", "waiting", "running", "serving"}:
            raise RuntimeError(health)
        start = await client.post("/start_session", json={})
        if start.status_code == 409:
            sid = "00000000-0000-0000-0000-000000000000"
        else:
            start.raise_for_status(); sid = start.json()["session_id"]
        registration = (await client.post(f"/sessions/{sid}/transport/webrtc/connections")).json()
        cid = registration["connection_id"]
        await pc.setLocalDescription(await pc.createOffer())
        mapping = [{"mid": transceiver.mid, "name": "main_video", "kind": "video", "direction": "recvonly"}]
        offered = await client.post(
            f"/sessions/{sid}/transport/webrtc/connections/{cid}/sdp_params",
            json={"sdp_offer": pc.localDescription.sdp, "track_mapping": mapping, "ice_servers": []},
        )
        offered.raise_for_status()
        for _ in range(120):
            answer = await client.get(f"/sessions/{sid}/transport/webrtc/connections/{cid}/sdp_params")
            if answer.status_code == 200:
                await pc.setRemoteDescription(RTCSessionDescription(answer.json()["sdp_answer"], "answer"))
                break
            await asyncio.sleep(.25)
        else:
            raise TimeoutError("SDP answer")
        await asyncio.wait_for(opened.wait(), 20)

        async def heartbeat() -> None:
            while True:
                channel.send(json.dumps({"scope": "runtime", "data": {"type": "ping", "data": {}}}))
                await asyncio.sleep(5)
        heartbeat_task = asyncio.create_task(heartbeat())

        payload = image_path.read_bytes()
        meta = {"name": image_path.name, "size": len(payload), "mime_type": "image/png"}
        slot = (await client.post(f"/sessions/{sid}/uploads", json=meta)).json()
        put = await client.put(slot["presigned_url"], content=payload, headers={"content-type": "application/octet-stream"})
        put.raise_for_status()

        def send(kind: str, data: dict, uploads=None) -> None:
            inner = {"type": kind, "data": data}
            if uploads: inner["uploads"] = uploads
            channel.send(json.dumps({"scope": "application", "data": inner}))

        ref = {"upload_id": slot["presigned_id"], **meta}
        send("set_image", {"prompt": "A steady forward exploration of this realistic scene.", "seed": 7}, {"image": ref})
        while True:
            msg = await asyncio.wait_for(messages.get(), 90)
            if msg["type"] == "image_selected": break

        actions = [
            {"forward": 1, "strafe": 0, "vertical": 0, "pitch": 0, "yaw": 0, "roll": 0},
            {"forward": 1, "strafe": .2, "vertical": 0, "pitch": 0, "yaw": .1, "roll": 0},
            {"forward": .8, "strafe": .4, "vertical": 0, "pitch": 0, "yaw": .2, "roll": 0},
            {"forward": .7, "strafe": .3, "vertical": .1, "pitch": .1, "yaw": .2, "roll": 0},
            {"forward": .8, "strafe": 0, "vertical": .2, "pitch": .1, "yaw": 0, "roll": 0},
            {"forward": 1, "strafe": -.2, "vertical": .1, "pitch": 0, "yaw": -.1, "roll": 0},
            {"forward": .9, "strafe": -.4, "vertical": 0, "pitch": -.1, "yaw": -.2, "roll": 0},
            {"forward": .7, "strafe": -.2, "vertical": -.1, "pitch": 0, "yaw": -.1, "roll": .05},
            {"forward": .8, "strafe": 0, "vertical": 0, "pitch": .1, "yaw": .1, "roll": 0},
            {"forward": 1, "strafe": .1, "vertical": 0, "pitch": 0, "yaw": .15, "roll": 0},
            {"forward": 1, "strafe": 0, "vertical": 0, "pitch": 0, "yaw": 0, "roll": 0},
            {"forward": .9, "strafe": 0, "vertical": .1, "pitch": .05, "yaw": 0, "roll": 0},
        ]
        for action in actions:
            send("set_camera_motion", action)
            await asyncio.sleep(.08)

        completed = []
        seen = []
        for _ in range(300):
            msg = await asyncio.wait_for(messages.get(), 180)
            seen.append(msg["type"])
            if msg["type"] == "chunk_completed":
                completed.append(msg["data"])
                if len(completed) == 2: break
        if len(completed) != 2: raise TimeoutError("two ChunkCompleted messages")
        for _ in range(120):
            if frames: break
            await asyncio.sleep(.25)
        metrics = (await client.get("/metrics")).text
        accepted = sum(int(float(line.rsplit(" ", 1)[1])) for line in metrics.splitlines()
                       if 'runtime_commands_total{command="set_camera_motion",outcome="accepted"}' in line)
        emitted = sum(int(float(line.rsplit(" ", 1)[1])) for line in metrics.splitlines()
                      if 'runtime_media_frames_total{track="main_video"}' in line)
        print(json.dumps({"actions_sent": len(actions), "actions_accepted": accepted,
                          "chunks": completed, "received_video_frames": frames,
                          "runtime_emitted_video_frames": emitted,
                          "message_types": sorted(set(seen))}, indent=2))
        if accepted < 10 or any(item["video_frames"] != 80 for item in completed) or emitted < 160:
            raise RuntimeError("end-to-end assertions failed")
        await client.post("/stop_session", json={})
    if heartbeat_task: heartbeat_task.cancel()
    await pc.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:18086")
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(main(args.base, args.image))
