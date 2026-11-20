"use client";

import { type Participant, ParticipantKind, type RemoteTrack, Room, RoomEvent, Track } from "livekit-client";
import { useEffect, useMemo, useRef, useState } from "react";

import { PRESETS, type PresetId, VOICES } from "@/lib/presets";

import { PORTRAITS } from "./portraits";
import { RoomDiagram, type RoomView } from "./room-diagram";

type Phase = "idle" | "starting" | "joining" | "live" | "ending";

interface Status {
  avatar: string;
  livekit: string | null;
  agent: string | null;
}

interface Caption {
  id: string;
  who: "you" | "avatar";
  text: string;
  final: boolean;
}

/** The identity the page's server gives the avatar. */
const AVATAR_IDENTITY = "avatar";

const PHASE_TEXT: Record<Phase, string> = {
  idle: "Ready",
  starting: "Starting the agent and the avatar",
  joining: "Joining the room",
  live: "Live",
  ending: "Ending the call",
};

export function LiveAvatar() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<Status | null>(null);
  const [preset, setPreset] = useState<PresetId | null>("maya");
  const [upload, setUpload] = useState<File | null>(null);
  const [uploadUrl, setUploadUrl] = useState<string | null>(null);
  const [voice, setVoice] = useState<string>(PRESETS[0].voice);
  const [instructions, setInstructions] = useState("");
  const [muted, setMuted] = useState(false);
  const [captions, setCaptions] = useState<Caption[]>([]);
  const [room, setRoom] = useState<RoomView>({
    connected: false,
    you: false,
    agent: false,
    avatar: false,
    youSpeaking: false,
    avatarSpeaking: false,
  });
  const [hasVideo, setHasVideo] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const roomRef = useRef<Room | null>(null);
  const callRef = useRef<string | null>(null);
  const startedAt = useRef<number | null>(null);

  // Ask the page's server what is configured, and keep asking until it answers: a dev server compiles the route on
  // its first request, and a server that was down comes back.
  useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      try {
        const resp = await fetch("/api/status", { cache: "no-store" });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        if (!stopped) setStatus(await resp.json());
      } catch {
        if (!stopped) timer = setTimeout(poll, 2000);
      }
    };
    void poll();
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, []);

  useEffect(() => {
    if (!upload) return setUploadUrl(null);
    const url = URL.createObjectURL(upload);
    setUploadUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [upload]);

  useEffect(() => {
    if (phase !== "live") return;
    const timer = setInterval(() => setElapsed(Math.floor((Date.now() - (startedAt.current ?? Date.now())) / 1000)), 500);
    return () => clearInterval(timer);
  }, [phase]);


  useEffect(() => () => void hangUp(), []); // eslint-disable-line react-hooks/exhaustive-deps

  const portraitSrc = upload ? uploadUrl : preset ? PORTRAITS[preset].src : null;
  const inCall = phase !== "idle";

  function choosePreset(id: PresetId) {
    setPreset(id);
    setUpload(null);
    setVoice(PRESETS.find((p) => p.id === id)!.voice);
  }

  function refreshRoom(r: Room) {
    const remotes = [...r.remoteParticipants.values()];
    const avatar = remotes.find((p) => p.identity === AVATAR_IDENTITY);
    const agent = remotes.find((p) => p.kind === ParticipantKind.AGENT && p.identity !== AVATAR_IDENTITY);
    setRoom((v) => ({ ...v, connected: true, you: true, agent: !!agent, avatar: !!avatar }));
  }

  async function start() {
    setError(null);
    setCaptions([]);
    setHasVideo(false);
    setPhase("starting");
    const form = new FormData();
    if (upload) form.set("portrait", upload);
    else if (preset) form.set("preset", preset);
    form.set("voice", voice);
    if (instructions.trim()) form.set("instructions", instructions.trim());
    try {
      const resp = await fetch("/api/calls", { method: "POST", body: form });
      const body = await resp.json();
      if (!resp.ok) throw new Error(body.error ?? `HTTP ${resp.status}`);
      callRef.current = body.room;
      setPhase("joining");
      const r = new Room({ adaptiveStream: true, dynacast: true });
      roomRef.current = r;
      r.on(RoomEvent.TrackSubscribed, (track: RemoteTrack) => {
        if (track.kind === Track.Kind.Video && videoRef.current) {
          track.attach(videoRef.current);
          setHasVideo(true);
          startedAt.current = Date.now();
          setPhase("live");
        }
        if (track.kind === Track.Kind.Audio && audioRef.current) track.attach(audioRef.current);
      });
      r.on(RoomEvent.ParticipantConnected, () => refreshRoom(r));
      r.on(RoomEvent.ParticipantDisconnected, () => refreshRoom(r));
      r.on(RoomEvent.ActiveSpeakersChanged, (speakers: Participant[]) => {
        const you = speakers.some((p) => p.identity === r.localParticipant.identity);
        setRoom((v) => ({ ...v, youSpeaking: you, avatarSpeaking: speakers.some((p) => p.identity !== r.localParticipant.identity) }));
      });
      r.on(RoomEvent.Disconnected, () => void hangUp());
      r.registerTextStreamHandler("lk.transcription", async (reader) => {
        const attrs = reader.info.attributes ?? {};
        const mic = r.localParticipant.getTrackPublication(Track.Source.Microphone)?.trackSid;
        const who: Caption["who"] = mic && attrs["lk.transcribed_track_id"] === mic ? "you" : "avatar";
        const id = attrs["lk.segment_id"] ?? reader.info.id;
        let text = "";
        for await (const chunk of reader) {
          text += chunk;
          const final = attrs["lk.transcription_final"] === "true";
          setCaptions((list) => {
            const rest = list.filter((c) => c.id !== id);
            return [...rest, { id, who, text, final }].slice(-30);
          });
        }
      });
      await r.connect(body.url, body.token);
      refreshRoom(r);
      await r.localParticipant.setMicrophoneEnabled(true);
      setMuted(false);
    } catch (e) {
      setError((e as Error).message);
      await hangUp();
    }
  }

  async function hangUp() {
    const r = roomRef.current;
    const call = callRef.current;
    roomRef.current = null;
    callRef.current = null;
    if (!r && !call) return;
    setPhase("ending");
    await r?.disconnect();
    if (call) await fetch(`/api/calls/${call}`, { method: "DELETE" }).catch(() => undefined);
    setRoom({ connected: false, you: false, agent: false, avatar: false, youSpeaking: false, avatarSpeaking: false });
    setHasVideo(false);
    setElapsed(0);
    setPhase("idle");
  }

  async function toggleMute() {
    const r = roomRef.current;
    if (!r) return;
    await r.localParticipant.setMicrophoneEnabled(muted);
    setMuted(!muted);
  }

  const ready = useMemo(() => !!status && !!status.livekit && !!status.agent && !status.avatar.startsWith("set "), [status]);

  const lastCaptions = captions.slice(-3);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <img src="/brand/reactor-logo-white.svg" alt="Reactor" className="logo" />
          <span className="product">Live Avatar</span>
        </div>
        <div className="status" title={status ? `${status.avatar} · LiveKit: ${status.livekit ?? "not set"} · agent: ${status.agent ?? "not set"}` : ""}>
          <span className={`dot ${ready ? "is-ok" : ""}`} />
          {status === null ? "Checking setup" : ready ? "Ready" : "Setup incomplete"}
        </div>
      </header>

      <main className="layout">
        <section className={`stage ${phase === "live" ? "is-live" : ""}`}>
          <video ref={videoRef} autoPlay playsInline muted className={hasVideo ? "is-visible" : ""} />
          <audio ref={audioRef} autoPlay />
          {!hasVideo && (
            <div className="stage-idle">
              {portraitSrc ? <img src={portraitSrc} alt="" /> : null}
              <span className="stage-state">
                {inCall ? <span className="spinner" /> : null}
                {PHASE_TEXT[phase]}
              </span>
            </div>
          )}
          {phase === "live" && (
            <span className="badge">
              <span className="live-dot" />
              {String(Math.floor(elapsed / 60)).padStart(2, "0")}:{String(elapsed % 60).padStart(2, "0")}
              {room.youSpeaking ? " · you" : room.avatarSpeaking ? " · avatar" : ""}
            </span>
          )}
          {lastCaptions.length > 0 && (
            <div className="captions" aria-live="polite">
              {lastCaptions.map((c) => (
                <p key={c.id} className={c.who === "you" ? "is-you" : ""}>
                  {c.text}
                </p>
              ))}
            </div>
          )}
        </section>

        <aside className="panel">
          <div className="field">
            <span className="label">Face</span>
            <div className="faces">
              {PRESETS.map((p) => (
                <button
                  key={p.id}
                  className={`face ${!upload && preset === p.id ? "is-selected" : ""}`}
                  onClick={() => choosePreset(p.id)}
                  disabled={inCall}
                  title={p.name}
                >
                  <img src={PORTRAITS[p.id].src} alt={p.name} />
                </button>
              ))}
              <label className={`face upload ${upload ? "is-selected" : ""} ${inCall ? "is-disabled" : ""}`} title="Upload a portrait">
                {uploadUrl ? <img src={uploadUrl} alt="Your portrait" /> : <span>+</span>}
                <input
                  type="file"
                  accept="image/png,image/jpeg,image/webp"
                  disabled={inCall}
                  onChange={(e) => {
                    const file = e.target.files?.[0] ?? null;
                    setUpload(file);
                    if (file) setPreset(null);
                  }}
                />
              </label>
            </div>
          </div>

          <div className="field">
            <span className="label">Voice</span>
            <select value={voice} disabled={inCall} onChange={(e) => setVoice(e.target.value)}>
              {VOICES.map((v) => (
                <option key={v}>{v}</option>
              ))}
            </select>
          </div>

          <div className="field">
            <span className="label">Instructions</span>
            <textarea
              rows={2}
              value={instructions}
              disabled={inCall}
              placeholder="Optional: who the agent is and how it talks"
              onChange={(e) => setInstructions(e.target.value)}
            />
          </div>

          <div className="actions">
            {!inCall ? (
              <button className="primary" onClick={start} disabled={!ready}>
                Start call
              </button>
            ) : (
              <>
                <button className="ghost" onClick={toggleMute} disabled={phase !== "live" && phase !== "joining"}>
                  {muted ? "Unmute" : "Mute"}
                </button>
                <button className="danger" onClick={hangUp} disabled={phase === "ending"}>
                  End call
                </button>
              </>
            )}
          </div>
          {error && <p className="error">{error}</p>}

          <div className="room">
            <div className="room-head">
              <span className="label">The room</span>
              <span className="room-state">
                {room.connected ? `${[room.you, room.agent, room.avatar].filter(Boolean).length} of 3 joined` : "Not in a call"}
              </span>
            </div>
            <RoomDiagram view={room} />
          </div>

        </aside>
      </main>
    </div>
  );
}
