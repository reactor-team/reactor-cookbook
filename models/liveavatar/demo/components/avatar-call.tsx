"use client";

import { Room, RoomEvent, Track, type RemoteTrack } from "livekit-client";
import { useEffect, useRef, useState } from "react";

/** OpenAI Realtime voices the agent can speak with. */
const VOICES = ["marin", "cedar", "alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"];

type Phase = "idle" | "starting" | "waiting" | "live" | "ending";

interface Status {
  avatar: string;
  livekit: string | null;
  agent: string | null;
}

export function AvatarCall() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<Status | null>(null);
  const [portrait, setPortrait] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [voice, setVoice] = useState("marin");
  const [speaking, setSpeaking] = useState<"avatar" | "you" | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const roomRef = useRef<Room | null>(null);
  const callRef = useRef<string | null>(null);

  useEffect(() => {
    fetch("/api/status")
      .then((r) => r.json())
      .then(setStatus)
      .catch(() => setStatus(null));
  }, []);

  useEffect(() => {
    if (!portrait) return setPreview(null);
    const url = URL.createObjectURL(portrait);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [portrait]);

  useEffect(() => () => void hangUp(), []); // eslint-disable-line react-hooks/exhaustive-deps

  async function start() {
    setError(null);
    setPhase("starting");
    const form = new FormData();
    if (portrait) form.set("portrait", portrait);
    form.set("voice", voice);
    try {
      const resp = await fetch("/api/calls", { method: "POST", body: form });
      const body = await resp.json();
      if (!resp.ok) throw new Error(body.error ?? `HTTP ${resp.status}`);
      callRef.current = body.room;
      const room = new Room({ adaptiveStream: true });
      roomRef.current = room;
      room.on(RoomEvent.TrackSubscribed, (track: RemoteTrack) => {
        if (track.kind === Track.Kind.Video && videoRef.current) {
          track.attach(videoRef.current);
          setPhase("live");
        }
        if (track.kind === Track.Kind.Audio && audioRef.current) track.attach(audioRef.current);
      });
      room.on(RoomEvent.ActiveSpeakersChanged, (speakers) => {
        const local = speakers.some((p) => p.identity === room.localParticipant.identity);
        setSpeaking(local ? "you" : speakers.length ? "avatar" : null);
      });
      room.on(RoomEvent.Disconnected, () => void hangUp());
      await room.connect(body.url, body.token);
      await room.localParticipant.setMicrophoneEnabled(true);
      setPhase((p) => (p === "live" ? p : "waiting"));
    } catch (e) {
      setError((e as Error).message);
      await hangUp();
    }
  }

  async function hangUp() {
    const room = roomRef.current;
    const call = callRef.current;
    roomRef.current = null;
    callRef.current = null;
    if (!room && !call) return;
    setPhase("ending");
    await room?.disconnect();
    if (call) await fetch(`/api/calls/${call}`, { method: "DELETE" }).catch(() => undefined);
    setSpeaking(null);
    setPhase("idle");
  }

  const inCall = phase === "waiting" || phase === "live";

  return (
    <main className="shell">
      <header>
        <h1>LiveKit avatar agent</h1>
        <p className="muted">
          A voice agent that speaks through a talking avatar. The avatar is a Reactor model in the LiveKit room;
          the agent sends it speech with LiveKit&apos;s avatar protocol.
        </p>
      </header>

      <section className="stage">
        <video ref={videoRef} autoPlay playsInline muted className={phase === "live" ? "on" : ""} />
        <audio ref={audioRef} autoPlay />
        {phase !== "live" && (
          <div className="placeholder">
            {preview ? <img src={preview} alt="The portrait" /> : null}
            <span>
              {phase === "starting" && "Starting the agent and the avatar…"}
              {phase === "waiting" && "Waiting for the avatar's video…"}
              {phase === "ending" && "Ending the call…"}
              {phase === "idle" && (preview ? "Ready" : "The model's default portrait")}
            </span>
          </div>
        )}
        {inCall && speaking && <div className="speaking">{speaking === "you" ? "You" : "Avatar"} speaking</div>}
      </section>

      {!inCall ? (
        <section className="controls">
          <label>
            Portrait <span className="muted">(optional: one face, frontal)</span>
            <input
              type="file"
              accept="image/png,image/jpeg,image/webp"
              disabled={phase !== "idle"}
              onChange={(e) => setPortrait(e.target.files?.[0] ?? null)}
            />
          </label>
          <label>
            Voice
            <select value={voice} disabled={phase !== "idle"} onChange={(e) => setVoice(e.target.value)}>
              {VOICES.map((v) => (
                <option key={v}>{v}</option>
              ))}
            </select>
          </label>
          <button onClick={start} disabled={phase !== "idle"}>
            {phase === "starting" ? "Starting…" : "Start call"}
          </button>
        </section>
      ) : (
        <section className="controls">
          <span className="muted">Talk to the avatar; interrupt it by talking over it.</span>
          <button className="end" onClick={hangUp}>
            End call
          </button>
        </section>
      )}

      {error && <p className="error">{error}</p>}
      <footer className="muted">
        Avatar: {status?.avatar ?? "…"} · LiveKit: {status?.livekit ?? "not set"} · Agent: {status?.agent ?? "not set"}
      </footer>
    </main>
  );
}
