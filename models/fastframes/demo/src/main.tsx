import { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { ReactorProvider, ReactorView, useReactor, useReactorMessage } from "@reactor-team/js-sdk";

import "./style.css";

const DEFAULT_ENDPOINT = "http://localhost:8080";
const EXAMPLE = "A corgi surfing a wave at sunset";
const TRACKS = [{ name: "main_video", kind: "video", direction: "recvonly" }] as const;

type ClipState = {
  variant: "ff-s" | "ff-l";
  world_id: number;
  phase: "idle" | "generating" | "paused" | "stopped" | "complete";
  prompt: string;
  prompt_pending: boolean;
  completed_chunks: number;
  output_frames: number;
  last_chunk_seconds: number | null;
};

function Workspace({ endpoint, applyEndpoint }: {
  endpoint: string;
  applyEndpoint: (endpoint: string) => void;
}) {
  const { status, connect, disconnect, sendCommand, lastError } = useReactor((state) => ({
    status: state.status,
    connect: state.connect,
    disconnect: state.disconnect,
    sendCommand: state.sendCommand,
    lastError: state.lastError,
  }));
  const [draftEndpoint, setDraftEndpoint] = useState(endpoint);
  const [clip, setClip] = useState<ClipState | null>(null);
  const [prompt, setPrompt] = useState("");
  const [seed, setSeed] = useState("0");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [dismissedError, setDismissedError] = useState(lastError);

  const ready = status === "ready";
  const modelLabel = ready ? (clip?.variant === "ff-l" ? "FastFrames-L" : clip?.variant === "ff-s" ? "FastFrames-S" : "FastFrames") : "FastFrames";
  const offline = status === "disconnected";
  const active = clip?.phase === "generating" || clip?.phase === "paused";
  const paused = clip?.phase === "paused";
  const hasOutput = ready && !!clip?.output_frames;
  const seedValue = Number(seed);
  const seedValid = seed !== "" && Number.isInteger(seedValue) && seedValue >= 0 && seedValue <= 4294967295;
  const connectionLabel = ready ? "Connected" : offline ? "Offline" : "Connecting";
  const sdkError = lastError && lastError !== dismissedError ? lastError.message : "";
  const stageLabel = !ready ? "Waiting for connection"
    : active ? paused ? "Paused" : "Generating"
    : clip?.phase === "complete" ? "Clip complete"
    : clip?.phase === "stopped" ? "Stopped" : "Ready to generate";

  useReactorMessage((message) => {
    if (message.type === "state_update") setClip(message.data as ClipState);
  });
  useEffect(() => {
    if (status === "disconnected") setClip(null);
  }, [status]);

  async function perform(action: () => Promise<void>) {
    setError("");
    setDismissedError(lastError);
    setBusy(true);
    try {
      await action();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(false);
    }
  }

  async function command(name: string, parameters: Record<string, unknown> = {}) {
    // A refusal resolves undefined and the SDK emits an error. Only broadcasts
    // change the displayed clip, including for viewers who did not send a command.
    const reply = await sendCommand(name, parameters);
    if (reply === undefined) return;
  }

  return (
    <>
      <header className="app-header">
        <a className="brand" href="https://reactor.inc" target="_blank" rel="noreferrer">
          <img src="/brand/reactor-mark-white.svg" width="32" height="24" alt="" />
          <span><strong>{modelLabel}</strong> <span className="muted">on Reactor</span></span>
        </a>
        <form className="endpoint" onSubmit={(event) => {
          event.preventDefault();
          setError("");
          const url = new URL(draftEndpoint);
          if (!["http:", "https:"].includes(url.protocol)) {
            setError("Enter an HTTP or HTTPS Runtime URL.");
            return;
          }
          applyEndpoint(url.toString().replace(/\/$/, ""));
        }}>
          <label htmlFor="runtime-url">Runtime URL</label>
          <input id="runtime-url" type="url" required value={draftEndpoint}
            disabled={!offline || busy} onChange={(event) => setDraftEndpoint(event.target.value)} />
          <button type="submit" disabled={!offline || busy || draftEndpoint === endpoint}>Apply</button>
        </form>
        <div className="connection">
          <span role="status"><i className={ready ? "online" : ""} />{connectionLabel}</span>
          <button disabled={busy || (!ready && !offline)} onClick={() => void perform(async () => {
            if (ready) {
              await disconnect(false);
              setClip(null);
            } else {
              await connect();
            }
          })}>{ready ? "Disconnect" : "Connect"}</button>
        </div>
      </header>

      <main>
        <aside aria-label="Generation controls">
          <div className="section-title">
            <h1>Scene prompt</h1>
            <button className="subtle" onClick={() => setPrompt(EXAMPLE)}>Load example</button>
          </div>
          <textarea id="prompt" aria-label="Prompt" value={prompt} maxLength={4096}
            placeholder="Describe the scene and how it moves…"
            onChange={(event) => setPrompt(event.target.value)} />
          <div className="actions">
            <button className="primary" disabled={!ready || busy || !prompt.trim() || !seedValid}
              onClick={() => void perform(() => command("start", { prompt, seed: seedValue }))}>
              {active ? "Start new clip" : "Start clip"}
            </button>
            <button disabled={!ready || busy || !active || !prompt.trim() || prompt === clip?.prompt}
              onClick={() => void perform(() => command("update_prompt", { prompt }))}>Update prompt</button>
          </div>
          {active && <p className="muted compact">
            {clip?.prompt_pending ? "Prompt update queued for the next generated chunk."
              : "Update prompt guides upcoming chunks within this clip."}
          </p>}
          <div className="actions secondary">
            <button disabled={!ready || busy || !active}
              onClick={() => void perform(() => command("set_paused", { paused: !paused }))}>
              {paused ? "Resume" : "Pause"}
            </button>
            <button disabled={!ready || busy || (!active && clip?.phase !== "complete")}
              onClick={() => void perform(() => command("stop"))}>Stop</button>
          </div>
          <label className="seed">New clip seed
            <input type="number" min={0} max={4294967295} step={1} value={seed}
              onChange={(event) => setSeed(event.target.value)} />
          </label>
          {(error || sdkError) && <p className="error" role="alert">{error || sdkError}</p>}
        </aside>

        <section className="stage" aria-label="Live video">
          <div className="section-title">
            <h2>Live video</h2><span className="muted">832 × 480 · Silent</span>
          </div>
          <div className="player">
            <ReactorView key={clip?.world_id ?? 0} track="main_video" muted videoObjectFit="contain" className="video" />
            {!hasOutput && <div className="placeholder">
              <span aria-hidden="true">▷</span>
              <p>{active ? "Waiting for the first frame" : ready ? "Start a clip to see it here" : "Connect to a Runtime to begin"}</p>
            </div>}
          </div>
          <div className="stage-status" aria-label="Clip progress">
            <span>{stageLabel}</span>
            {ready && clip && clip.phase !== "idle" && <span>{clip.output_frames} / 81 frames{clip.last_chunk_seconds === null
              ? "" : ` · ${Math.round(1000 * clip.last_chunk_seconds)} ms last chunk`}</span>}
          </div>
        </section>
      </main>
    </>
  );
}

function App() {
  const [endpoint, setEndpoint] = useState(
    () => new URLSearchParams(location.search).get("endpoint") || DEFAULT_ENDPOINT,
  );
  return (
    <ReactorProvider key={endpoint} modelName="your-org/fastframes" local apiUrl={endpoint}
      modelTracks={[...TRACKS]} connectOptions={{ autoConnect: false }}>
      <Workspace endpoint={endpoint} applyEndpoint={setEndpoint} />
    </ReactorProvider>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
