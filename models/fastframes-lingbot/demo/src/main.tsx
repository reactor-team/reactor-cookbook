import { useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { ReactorProvider, ReactorView, useReactor, useReactorMessage } from "@reactor-team/js-sdk";

import { actionLabel, CameraControls, type CameraAction } from "./camera";
import "./style.css";

const DEFAULT_ENDPOINT = "http://localhost:8080";
const DEFAULT_PROMPT = "A continuous view of the scene in the reference image. " +
  "Preserve the subjects, spatial layout, lighting, and visual style as the camera viewpoint changes. " +
  "Keep motion smooth and consistent.";
const TRACKS = [{ name: "main_video", kind: "video", direction: "recvonly" }] as const;
const MAX_BYTES = 8 * 1024 ** 2;

type WorldState = {
  world_id: number;
  phase: "idle" | "active" | "stopped" | "complete";
  queued_action: CameraAction | null;
  held_action: CameraAction | null;
  last_action: CameraAction | null;
  prompt_pending: boolean;
  completed_chunks: number;
  max_chunks?: number;
  playback_fps?: number;
  output_frames: number;
  last_chunk_seconds: number | null;
};

function Workspace({ endpoint, applyEndpoint }: {
  endpoint: string;
  applyEndpoint: (url: string) => void;
}) {
  const status = useReactor((state) => state.status);
  const connect = useReactor((state) => state.connect);
  const disconnect = useReactor((state) => state.disconnect);
  const sendCommand = useReactor((state) => state.sendCommand);
  const uploadFile = useReactor((state) => state.uploadFile);
  const lastError = useReactor((state) => state.lastError);
  const [draftEndpoint, setDraftEndpoint] = useState(endpoint);
  const [world, setWorld] = useState<WorldState | null>(null);
  const [controller, setController] = useState<number | null>(null);
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [seed, setSeed] = useState("0");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [previousError, setPreviousError] = useState(lastError);
  const stage = useRef<HTMLElement>(null);

  const ready = status === "ready";
  const offline = status === "disconnected";
  const worldId = world?.world_id;
  const active = world?.phase === "active";
  const ownsWorld = ready && active && controller === worldId;
  const canMove = ownsWorld && !!world.completed_chunks;
  const hasOutput = ready && !!world?.output_frames;
  const seedNumber = Number(seed);
  const validSeed = seed !== "" && Number.isInteger(seedNumber) && seedNumber >= 0 && seedNumber <= 4294967295;
  const connectionLabel = ready ? "Connected" : offline ? "Offline" : "Connecting";
  const issue = error || (lastError && lastError !== previousError ? lastError.message : "");

  useReactorMessage((message) => {
    if (message.type === "state_update") setWorld(message.data as WorldState);
  });
  useEffect(() => {
    if (offline) {
      setWorld(null);
      setController(null);
    }
  }, [offline]);
  useEffect(() => {
    if (!image) {
      setPreview("");
      return;
    }
    const url = URL.createObjectURL(image);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [image]);

  const move = useCallback((action: CameraAction, hold: boolean) => {
    if (canMove) void sendCommand("move_camera", { world_id: worldId, action, hold });
  }, [canMove, sendCommand, worldId]);
  const release = useCallback((discardPending: boolean) => {
    if (canMove) void sendCommand("release_camera", { world_id: worldId, discard_pending: discardPending });
  }, [canMove, sendCommand, worldId]);

  async function perform(action: () => Promise<void>) {
    setError("");
    setPreviousError(lastError);
    setBusy(true);
    try {
      await action();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(false);
    }
  }

  const nextAction = world?.queued_action ?? world?.held_action;
  const statusLabel = !ready ? "Waiting for connection"
    : world?.phase === "complete" ? "World complete"
    : world?.phase === "stopped" ? "Stopped"
    : !active ? "Ready to start"
    : !world.completed_chunks ? "Preparing your world"
    : nextAction ? actionLabel(nextAction) : ownsWorld ? "Streaming" : "Watching";

  return (
    <>
      <header className="app-header">
        <a className="brand" href="https://reactor.inc" target="_blank" rel="noreferrer">
          <img src="/brand/reactor-mark-white.svg" width="32" height="24" alt="" />
          <span><strong>FastFrames-LingBot</strong> <span className="muted">on Reactor</span></span>
        </a>
        <form className="endpoint" onSubmit={(event) => {
          event.preventDefault();
          const url = new URL(draftEndpoint);
          if (!["http:", "https:"].includes(url.protocol)) {
            setError("Enter an HTTP or HTTPS Runtime URL.");
            return;
          }
          applyEndpoint(url.toString().replace(/\/$/, ""));
        }}>
          <label htmlFor="runtime-url">Runtime URL</label>
          <input id="runtime-url" type="url" value={draftEndpoint} required disabled={!offline || busy}
            onChange={(event) => setDraftEndpoint(event.target.value)} />
          <button disabled={!offline || busy || draftEndpoint === endpoint}>Apply</button>
        </form>
        <div className="connection">
          <span role="status"><i className={ready ? "online" : ""} />{connectionLabel}</span>
          <button disabled={busy || (!ready && !offline)} onClick={() => void perform(async () => {
            if (ready) await disconnect(false);
            else await connect();
          })}>{ready ? "Disconnect" : "Connect"}</button>
        </div>
      </header>
      <main>
        <aside aria-label="World setup">
          <h1>Start a world</h1>
          <label className="image-field">
            Start image
            {preview && <img className="anchor-preview" src={preview} alt="Selected start image" />}
            <input type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => {
              const file = event.target.files?.[0] ?? null;
              if (file && (file.size === 0 || file.size > MAX_BYTES)) {
                setImage(null);
                setError("Choose an image smaller than 8 MiB.");
              } else {
                setImage(file);
                setError("");
              }
            }} />
          </label>
          <label className="prompt-field">
            Scene prompt
            <textarea value={prompt} maxLength={1600} onChange={(event) => setPrompt(event.target.value)}
              placeholder="Describe the scene you want to explore…" />
          </label>
          <label className="seed">World seed
            <input type="number" min={0} max={4294967295} step={1} value={seed}
              onChange={(event) => setSeed(event.target.value)} />
          </label>
          <div className="actions">
            <button className="primary" disabled={!ready || busy || (active && !ownsWorld) || !image || !prompt.trim() || !validSeed}
              onClick={() => void perform(async () => {
                if (!image) return;
                const reference = await uploadFile(image);
                const reply = await sendCommand("start_world", { image: reference, prompt, seed: seedNumber });
                if (!reply) return;
                setController((reply.data as { world_id: number }).world_id);
                stage.current?.focus({ preventScroll: true });
              })}>{ownsWorld ? "Restart world" : "Start world"}</button>
            <button disabled={!ownsWorld || busy}
              onClick={() => void perform(async () => {
                await sendCommand("stop_world", { world_id: worldId });
              })}>Stop</button>
          </div>
          <button disabled={!ownsWorld || busy || !prompt.trim()}
            onClick={() => void perform(async () => {
              await sendCommand("update_prompt", { world_id: worldId, prompt });
            })}>Update prompt</button>
          {ownsWorld && world?.prompt_pending && !!world.completed_chunks &&
            <p className="muted compact">Prompt queued for the next block.</p>}
          {issue && <p className="error" role="alert">{issue}</p>}
          <CameraControls key={worldId} enabled={canMove} move={move} release={release} />
        </aside>
        <section ref={stage} tabIndex={-1} className="stage" aria-label="Live world">
          <div className="section-title">
            <h2>Live world</h2>
            <span className="muted">832 × 464{ready && world?.playback_fps ? ` · ${world.playback_fps} FPS` : ""}</span>
          </div>
          <div className="player">
            <ReactorView key={worldId ?? 0} track="main_video" muted videoObjectFit="contain" className="video" />
            {!hasOutput && <div className="placeholder"><p>{active ? "Preparing your world…" : "Choose an image and start a world."}</p></div>}
          </div>
          <div className="stage-status" aria-label="World progress">
            <span>{statusLabel}</span>
            {hasOutput && world && <span>
              {world.completed_chunks} / {world.max_chunks ?? 60} blocks · {world.output_frames} frames
              {world.last_action ? ` · ${actionLabel(world.last_action)}` : ""}
              {world.last_chunk_seconds === null ? "" : ` · ${Math.round(world.last_chunk_seconds * 1000)} ms`}
            </span>}
          </div>
        </section>
      </main>
    </>
  );
}

function App() {
  const [endpoint, setEndpoint] = useState(() => new URLSearchParams(location.search).get("endpoint") || DEFAULT_ENDPOINT);
  return (
    <ReactorProvider key={endpoint} modelName="your-org/fastframes-lingbot" local apiUrl={endpoint}
      modelTracks={[...TRACKS]} connectOptions={{ autoConnect: false }}>
      <Workspace endpoint={endpoint} applyEndpoint={setEndpoint} />
    </ReactorProvider>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
