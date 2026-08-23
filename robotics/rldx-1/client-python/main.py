# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""
RLDX-1 client — keeping multi-camera input in sync, and correlating the model's
answers back to the client's own timeline.

RLDX-1 (RLWRLD/RLDX-1) is a vision-language-action model: three camera views in,
a 16-step action chunk out per inference. Over WebRTC each view is its own
track and the actions come back on the data channel, so what looks like one
observation is really four independent streams with no clock in common. Two
questions fall out of that, and neither has an answer unless the client puts one
on the wire:

  * *Which frames belong together?* Three tracks drift. A window that pairs a
    fresh left frame with a stale wrist frame shows the policy a scene that never
    existed. So: read the clock ONCE per tick and push every view with that same
    `capture_time_us`. The value, not the moment the push landed, is what the
    server aligns on.

  * *Where does this chunk belong on my timeline?* The model decides its own
    inference timing and pushes results when they are ready, so an arriving chunk
    carries no inherent position in the client's loop. So: stamp the tick
    (`capture_us`, `seq`), and the model echoes those values back on the chunk as
    `source_capture_us` / `source_seq`. `now - source_capture_us` is then the true
    age of the observation this chunk was computed from — measured entirely on the
    client's own clock, with no clock-offset guesswork.

The state rides along the same seam. Rather than sending proprioception as a
separate data-channel message that the server then has to pair up with frames by
guesswork, the client tags each frame with the state JSON it was read with
(`push_frame(..., user_data=...)`). The state is then attached to the frames it
belongs to, by construction.

Which carrier to use is not a flag — it comes from the `model_schema` handshake
(`state_source`). This client reads it and picks, so the same file runs against a
deployment that reads frame tags and against an older one that only accepts the
`set_state_json` command.

Usage:
    uv run python main.py --local
    uv run python main.py --api-key rk_...
    uv run python main.py --api-key rk_... --task "put the cup on the tray"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import time

import numpy as np

from reactor_sdk import Reactor, ReactorStatus, time_micros

logging.basicConfig(level=logging.WARNING)

# Used only to bootstrap publishing when the handshake has not landed yet — the
# authoritative list is `views` in `model_schema`.
DEFAULT_VIEWS = ("left_view", "right_view", "wrist_view")
DEFAULT_RESOLUTION = (256, 256)
DEFAULT_CONTROL_HZ = 20.0
# Fallback state layout, used only if a deployment announces no `state_dims`.
DEFAULT_STATE_DIMS = {
    "end_effector_position_relative": 3,
    "end_effector_rotation_relative": 4,
    "gripper_qpos": 2,
    "base_position": 3,
    "base_rotation": 4,
}


def synth_frame(h: int, w: int, view_index: int, t: float) -> np.ndarray:
    """One synthetic (H, W, 3) uint8 RGB camera frame — a circle orbiting the center.

    A stand-in for a real camera. Each view gets its own orbit speed and tint so
    the three tracks are visibly distinct in a recording.
    """
    frame = np.full((h, w, 3), (20, 20, 40), dtype=np.uint8)
    speed = 1.0 + 0.2 * view_index
    cx = int(w / 2 + (w / 3) * math.cos(t * speed))
    cy = int(h / 2 + (h / 3) * math.sin(t * speed * 0.7))
    y, x = np.ogrid[:h, :w]
    tint = [(255, 255, 255), (255, 200, 160), (160, 220, 255)][view_index % 3]
    frame[(x - cx) ** 2 + (y - cy) ** 2 < (h // 8) ** 2] = tint
    return frame


def synth_state(state_dims: dict[str, int], t: float) -> dict[str, list[float]]:
    """A slowly-varying proprioceptive snapshot, built FROM the announced dims.

    The shape of the state is not this client's to decide: `model_schema` says
    which vectors the loaded checkpoint wants and how long each one is, and a
    vector of the wrong length is rejected wholesale. Building the dict by
    walking `state_dims` means a checkpoint swap reconfigures the client instead
    of breaking it.

    The values are synthetic — as synthetic as the frames. Real numbers come off
    the robot at the instant the frames were grabbed.
    """
    return {
        key: [round(0.1 * math.sin(0.7 * t + i + j), 6) for j in range(dim)]
        for i, (key, dim) in enumerate(state_dims.items())
    }


def as_int(value: object) -> int | None:
    """A number off the wire as an `int`, or `None` if the field was absent.

    Worth its own function: the data channel carries JSON, so a field the model
    declares as an integer arrives as a JSON number — `source_capture_us` comes
    in as `2155019618705.0`, and `isinstance(that, int)` is False. Coerce, don't
    type-check. Nothing is lost doing so: a float64 holds integers exactly up to
    2**53, which is ~285 years of microseconds.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


class Client:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.reactor = Reactor(
            model_name=args.model,
            api_key=args.api_key,
            api_url=args.api_url,
            local=args.local,
        )
        self.t0 = time.monotonic()
        self.schema: dict | None = None
        self.schema_event = asyncio.Event()
        self.loop: asyncio.AbstractEventLoop | None = None

        # What we sent, so an echo can be checked against it rather than trusted.
        self.sent_capture_us: dict[int, int] = {}
        self.ticks = 0

        # What came back.
        self.chunks = 0
        self.arrivals: list[float] = []      # monotonic seconds, for inter-arrival
        self.ages_ms: list[float] = []        # client-clock age of the source snapshot
        self.skews_us: list[int] = []
        self.echo_ok = 0
        self.echo_mismatch = 0
        self.echo_absent = 0
        self.command_errors: list[dict] = []
        self.first_chunk_shapes: dict[str, tuple] = {}
        self.last_report = 0.0

    # ---------------------------------------------------------------- handlers

    def on_status(self, status: ReactorStatus) -> None:
        print(f"[status] {status.value} (+{time.monotonic() - self.t0:.1f}s)")

    def on_message(self, message: object, scope: object = None) -> None:
        # Model messages arrive wrapped: {"type": <message name>, "data": {...}}.
        if not isinstance(message, dict):
            return
        mtype = message.get("type")
        body = message.get("data")
        if not isinstance(body, dict):
            return
        if mtype == "model_schema":
            self.schema = body
            if self.loop is not None:
                self.loop.call_soon_threadsafe(self.schema_event.set)
            else:
                self.schema_event.set()
        elif mtype == "action_prediction":
            self.on_action(body)
        elif mtype == "command_error":
            self.command_errors.append(body)
            print(f"[command_error] {body.get('command')}: {body.get('reason')}")

    def on_action(self, body: dict) -> None:
        """Place one action chunk on the client's own timeline."""
        # Read the client clock as the chunk lands. `source_capture_us` is a value
        # this client itself minted, so the subtraction stays inside one clock —
        # no epoch alignment, no drift correction.
        now_us = time_micros()
        now = time.monotonic()
        self.chunks += 1
        self.arrivals.append(now)

        source_capture_us = as_int(body.get("source_capture_us"))
        source_seq = as_int(body.get("source_seq"))
        skew_us = as_int(body.get("view_skew_us"))
        if skew_us is not None:
            self.skews_us.append(skew_us)

        age_ms = None
        if source_capture_us is not None:
            age_ms = (now_us - source_capture_us) / 1000.0
            self.ages_ms.append(age_ms)

        # Correlate the echo back to the tick we actually sent. An echo that does
        # not match what we put on the wire is worse than no echo: it would put
        # the chunk at the wrong point in the timeline.
        if source_seq is not None:
            expected = self.sent_capture_us.get(source_seq)
            if expected is not None and expected == source_capture_us:
                self.echo_ok += 1
            else:
                self.echo_mismatch += 1
        else:
            self.echo_absent += 1

        if self.chunks == 1:
            for key in ("end_effector_position", "end_effector_rotation",
                        "gripper_close", "base_motion", "control_mode"):
                v = body.get(key)
                if isinstance(v, list) and v and isinstance(v[0], list):
                    self.first_chunk_shapes[key] = (len(v), len(v[0]))
            print("\n[first chunk]")
            print(f"  step={as_int(body.get('step'))}  shapes={self.first_chunk_shapes}")
            print(f"  source_seq={source_seq}  source_capture_us={source_capture_us}")
            if source_seq is None:
                print("  echo: none — nothing to tie this chunk to a tick")
            else:
                sent = self.sent_capture_us.get(source_seq)
                print(f"  echo matches the tick we sent: {sent == source_capture_us} "
                      f"(we pushed seq={source_seq} at capture_us={sent})")
            print(f"  age of source snapshot on our clock: "
                  f"{f'{age_ms:.0f} ms' if age_ms is not None else 'unavailable (no echo)'}")
            print(f"  view_skew_us={skew_us}  "
                  f"({'cross-view capture spread of the frames this chunk used' if skew_us is not None else 'transport carried no per-frame stamps'})")
            print(f"  first action step: {body.get('end_effector_position', [[]])[0]}\n")
            self.last_report = now
        elif now - self.last_report >= 2.0:
            # Compact running line — at ~7 Hz, one line per chunk is noise.
            self.last_report = now
            age = f"{pct(self.ages_ms, 50):.0f}ms" if self.ages_ms else "n/a"
            skew = (f"{pct([float(s) for s in self.skews_us], 50):.0f}us"
                    if self.skews_us else "n/a")
            print(f"[chunks {self.chunks}] seq={source_seq} age p50={age} "
                  f"skew p50={skew} echo ok={self.echo_ok}/{self.chunks}")

    # ------------------------------------------------------------------- setup

    async def await_schema(self, timeout: float) -> dict | None:
        try:
            await asyncio.wait_for(self.schema_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        return self.schema

    # -------------------------------------------------------------------- main

    async def run(self) -> None:
        args = self.args
        self.loop = asyncio.get_running_loop()
        self.reactor.on_status(self.on_status)
        self.reactor.on_message(self.on_message)

        print(f"Connecting to {'localhost' if args.local else args.api_url} "
              f"(model={args.model}) ...")
        await self.reactor.connect()
        while self.reactor.get_status() != ReactorStatus.READY:
            await asyncio.sleep(0.1)
            if time.monotonic() - self.t0 > args.connect_timeout:
                print(f"[timeout] not READY after {args.connect_timeout}s")
                await self.reactor.disconnect()
                return
        print(f"[ready] +{time.monotonic() - self.t0:.1f}s")

        # The model announces its contract on connect, but that send can race the
        # data channel opening. Ask for it explicitly — the model re-serves it on
        # demand — and, if it still has not landed, publish frames: media flowing
        # proves the channel is up, and the model re-announces then.
        await self.reactor.send_command("get_schema", {})
        schema = await self.await_schema(timeout=5.0)

        views = list(schema.get("views") or DEFAULT_VIEWS) if schema else list(DEFAULT_VIEWS)
        tracks = {v: await self.reactor.publish_track(v) for v in views}
        print(f"[publish] {views}")

        if schema is None:
            schema = await self.await_schema(timeout=10.0)
        if schema is None:
            print("[schema] none received — falling back to documented defaults")
            schema = {}

        # ---- configure from the handshake, not from constants ----
        views_now = list(schema.get("views") or views)
        if views_now != views:
            print(f"[schema] announces views {views_now}, published {views} — "
                  f"republish to match a checkpoint that wants different views")
        resolution = schema.get("resolution") or list(DEFAULT_RESOLUTION)
        height, width = int(resolution[0]), int(resolution[1])
        control_hz = float(schema.get("control_hz") or DEFAULT_CONTROL_HZ)
        state_dims = {str(k): int(v) for k, v in
                      (schema.get("state_dims") or DEFAULT_STATE_DIMS).items()}
        tag_keys = set(schema.get("state_tag_keys") or ())
        state_source = schema.get("state_source")

        print(f"[schema] views={views_now} resolution={height}x{width} "
              f"control_hz={control_hz:g} "
              f"action_horizon={as_int(schema.get('action_horizon'))} "
              f"exec_horizon={as_int(schema.get('exec_horizon'))} "
              f"state_fallback={schema.get('state_fallback')}")
        print(f"[schema] state_dims={state_dims}")

        # ---- carrier selection: the handshake decides, not a command-line flag ----
        if state_source == "frame_metadata":
            tag_frames = True
            print("[carrier] frame_metadata — the handshake asks for state on the "
                  "frames, so every view's frame is tagged with the state JSON it "
                  "was read with")
        else:
            tag_frames = False
            why = ("state_source not announced — this deployment predates frame "
                   "metadata" if state_source is None
                   else f"state_source={state_source!r}")
            print(f"[carrier] set_state_json ({why}), so the state goes over the "
                  f"data channel as a separate message")
        if tag_frames and tag_keys:
            print(f"[carrier] embedding {sorted(tag_keys)} in the state JSON — the "
                  f"model orders tags by our stamp and echoes it back on each chunk")
        elif tag_frames:
            print("[carrier] no state_tag_keys announced — nothing to correlate "
                  "chunks back with; results will land without an echo")

        if args.task:
            await self.reactor.send_command("set_task_description",
                                            {"task_description": args.task})
            print(f"[task] {args.task!r}")

        # ---- the tick loop ----
        period = 1.0 / control_hz
        print(f"\nStreaming {len(views)} views at {control_hz:g} Hz for "
              f"{args.duration:.0f}s ...")
        deadline = time.monotonic() + args.duration
        try:
            while time.monotonic() < deadline:
                tick_start = time.monotonic()
                seq = self.ticks
                self.ticks += 1

                # ONE clock reading for the whole tick. This is the entire trick:
                # every view of this observation goes on the wire carrying this
                # same number, so the server can recognise them as one moment
                # however far apart the three tracks actually deliver them. Read
                # the engine's clock — a `time.time()` value is not a substitute.
                capture_us = time_micros()

                state = synth_state(state_dims, tick_start - self.t0)
                # The two reserved keys ride inside the state object, alongside
                # the vectors — but only when the handshake says they are read.
                # Inventing keys a deployment does not know about is how a state
                # payload silently stops parsing.
                if "capture_us" in tag_keys:
                    state["capture_us"] = capture_us
                if "seq" in tag_keys:
                    state["seq"] = seq
                state_bytes = json.dumps(state, separators=(",", ":")).encode("utf-8")
                self.sent_capture_us[seq] = capture_us

                if not tag_frames:
                    # Fallback carrier: one message per tick, arriving on its own
                    # stream. The stamped frames still buy cross-view alignment;
                    # what is lost is the state being tied to specific frames.
                    await self.reactor.send_command(
                        "set_state_json", {"state_json": state_bytes.decode("utf-8")})

                for i, view in enumerate(views):
                    frame = synth_frame(height, width, i, tick_start - self.t0)
                    tracks[view].push_frame(
                        frame,
                        user_data=state_bytes if tag_frames else None,
                        capture_time_us=capture_us,
                    )

                # Keep only enough history to correlate chunks still in flight.
                if len(self.sent_capture_us) > 512:
                    for old in sorted(self.sent_capture_us)[:256]:
                        del self.sent_capture_us[old]

                await asyncio.sleep(max(0.0, period - (time.monotonic() - tick_start)))
        finally:
            await self.reactor.disconnect()

        self.summary(tag_frames, bool(tag_keys))

    # ----------------------------------------------------------------- summary

    def summary(self, tag_frames: bool, embedded_stamp: bool) -> None:
        print("\n===== RLDX-1 sync summary =====")
        print(f"ticks published: {self.ticks} ; action chunks received: {self.chunks}")
        print(f"state carrier: {'frame metadata' if tag_frames else 'set_state_json command'}")

        if len(self.arrivals) >= 4:
            deltas = np.diff(self.arrivals)[2:] * 1000.0  # drop warmup intervals
            print(f"inter-arrival ms: p50={pct(list(deltas), 50):.0f} "
                  f"p90={pct(list(deltas), 90):.0f} "
                  f"({1000.0 / max(pct(list(deltas), 50), 1e-9):.1f} chunks/s)")

        # The number that matters for control: how stale the observation behind a
        # chunk already is by the time the chunk is in the client's hands.
        if self.ages_ms:
            print(f"chunk age on our clock ms: p50={pct(self.ages_ms, 50):.0f} "
                  f"p90={pct(self.ages_ms, 90):.0f} "
                  f"(from {len(self.ages_ms)} echoed chunks)")
        else:
            print("chunk age on our clock: unavailable — no chunk carried "
                  "source_capture_us")

        total_echo = self.echo_ok + self.echo_mismatch
        if total_echo:
            print(f"echo correlation: {self.echo_ok}/{total_echo} chunks echoed a "
                  f"stamp matching the tick we sent"
                  + (f" ; {self.echo_mismatch} mismatched" if self.echo_mismatch else ""))
        elif not embedded_stamp:
            print("echo correlation: not attempted — the handshake announced no "
                  "state_tag_keys, so no stamp was embedded to echo")
        else:
            print(f"echo correlation: none of the {self.echo_absent} chunks carried "
                  f"source_capture_us / source_seq — this deployment predates the "
                  f"echo fields (rldx-1 < 0.5.0)")

        if self.skews_us:
            print(f"view_skew_us: p50={pct([float(s) for s in self.skews_us], 50):.0f} "
                  f"max={max(self.skews_us)} (across {len(self.skews_us)} chunks)")
        else:
            print("view_skew_us: not reported — the transport carried no per-frame "
                  "capture stamps, or this deployment predates the field")

        if self.command_errors:
            print(f"command_errors: {len(self.command_errors)}")
            for err in self.command_errors[:3]:
                print(f"  {err.get('command')}: {err.get('reason')}")
        else:
            print("command_errors: none — the state carrier worked")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RLDX-1 client: frame-metadata sync + timeline correlation")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--local", action="store_true", help="Connect to localhost:8080")
    mode.add_argument("--api-key", help="Reactor API key (rk_...)")
    p.add_argument("--api-url", default="https://api.reactor.inc",
                   help="Reactor API base URL")
    p.add_argument("--model", default="rldx-1", help="Model name (default: rldx-1)")
    p.add_argument("--task", default="", help="Task description to condition the policy")
    p.add_argument("--duration", type=float, default=60.0, help="Seconds to stream")
    p.add_argument("--connect-timeout", type=float, default=300.0,
                   help="Max seconds to wait for READY (cold start pulls weights)")
    return p.parse_args()


if __name__ == "__main__":
    try:
        asyncio.run(Client(parse_args()).run())
    except KeyboardInterrupt:
        pass
