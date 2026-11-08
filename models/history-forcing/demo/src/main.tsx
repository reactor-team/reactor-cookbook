import { useEffect, useRef, useState, type ReactNode } from "react";
import { createRoot } from "react-dom/client";
import {
  ReactorProvider,
  ReactorView,
  useReactor,
  useReactorMessage,
} from "@reactor-team/js-sdk";
import "./style.css";

type World = {
  prompt: string;
  seed: number | null;
  active: boolean;
  paused: boolean;
  world_id: number;
  completed_chunks: number;
  output_frames: number;
  last_chunk_seconds: number | null;
};

function Workspace({ endpointSelector }: { endpointSelector: ReactNode }) {
  const { status, connect, disconnect, sendCommand, lastError } = useReactor(
    (s) => ({
      status: s.status,
      connect: s.connect,
      disconnect: s.disconnect,
      sendCommand: s.sendCommand,
      lastError: s.lastError,
    }),
  );
  const [world, setWorld] = useState<World | null>(null);
  const [prompt, setPrompt] = useState("");
  const [seed, setSeed] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [handledError, setHandledError] = useState(lastError);
  const dirty = useRef(false);
  const ready = status === "ready";
  useReactorMessage((message) => {
    if (message.type !== "state_update") return;
    const accepted = message.data as World;
    setWorld(accepted);
    if (!dirty.current) setPrompt(accepted.prompt);
  });
  useEffect(() => {
    if (lastError && lastError !== handledError) {
      setError(lastError.message);
      setHandledError(lastError);
    }
  }, [lastError, handledError]);
  useEffect(() => {
    if (status === "disconnected") setWorld(null);
  }, [status]);
  useEffect(() => {
    if (!ready) return;
    let listening = true;
    void sendCommand("get_state", {}).then((reply) => {
      if (!listening || reply === undefined) return;
      const accepted = reply.data as World;
      setWorld(accepted);
      if (!dirty.current) setPrompt(accepted.prompt);
    }).catch((e: unknown) => {
      if (listening) setError(e instanceof Error ? e.message : String(e));
    });
    return () => { listening = false; };
  }, [ready, sendCommand]);
  async function perform(action: () => Promise<unknown>) {
    setError("");
    setHandledError(lastError);
    setBusy(true);
    try {
      await action();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }
  async function command(name: string, args: Record<string, unknown> = {}) {
    const reply = await sendCommand(name, args);
    if (reply === undefined) return;
    const accepted = reply.data as World;
    setWorld(accepted);
    if (name === "start" || name === "set_prompt") {
      dirty.current = false;
      setPrompt(accepted.prompt);
    }
    return accepted;
  }
  return (
    <main>
      <header className="app-header">
        <a className="brand" href="https://reactor.inc" target="_blank" rel="noreferrer" aria-label="History-Forcing on Reactor">
          <img src="/brand/reactor-mark-white.svg" alt="" width="33" height="24" />
          <span className="brand-title"><strong>History-Forcing</strong><span>on Reactor</span></span>
        </a>
        {endpointSelector}
        <div className="connection-controls">
          <span className={`connection ${ready ? "connected" : ""}`} role="status">
            <i aria-hidden="true" />{ready ? "Connected" : status === "disconnected" ? "Offline" : status}
          </span>
          <button disabled={busy || status === "connecting" || status === "waiting"}
            onClick={() => void perform(async () => {
              if (ready) { await disconnect(); setWorld(null); }
              else await connect();
            })}>
            {ready ? "Disconnect" : "Connect"}
          </button>
        </div>
      </header>
      <div className="workspace">
        <aside>
          <div className="panel-intro">
            <p className="eyebrow">TEXT TO VIDEO</p>
            <h1>Your prompt</h1>
            <p>Describe the scene. Update it as the video unfolds.</p>
          </div>
          <label className="prompt-label">
            Scene description
            <textarea aria-label="Prompt" rows={9} maxLength={4096} value={prompt}
              placeholder="A musician strumming an acoustic guitar in a warmly lit room…"
              onChange={(e) => { dirty.current = true; setPrompt(e.target.value); }} />
          </label>
          <div className="prompt-actions">
            <button className="primary" disabled={!ready || busy || !prompt.trim()}
              onClick={() => void perform(() => command("start", { prompt, seed }))}>
              {world?.active ? "Start new video" : "Start video"}<span aria-hidden="true">↗</span>
            </button>
            <button disabled={!ready || busy || !world?.active || !prompt.trim()}
              onClick={() => void perform(() => command("set_prompt", { prompt }))}>Update prompt</button>
          </div>
          <p className="hint">Prompt updates guide the next frames and keep your video going.</p>
          <div className="actions">
            <button disabled={!ready || busy || !world?.active}
              onClick={() => void perform(() => command("set_paused", { paused: !world?.paused }))}>
              {world?.paused ? "Resume" : "Pause"}
            </button>
            <button disabled={!ready || busy || !world?.active} onClick={() => void perform(() => command("reset", {}))}>Reset</button>
            <button disabled={!ready || busy || !world?.active} onClick={() => void perform(() => command("stop"))}>Stop</button>
          </div>
          {error && <p className="error" role="alert">{error}</p>}
          <details className="settings">
            <summary>Generation settings</summary>
            <label>Seed
              <input aria-label="Seed" type="number" min={0} max={4294967295} value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
            </label>
            <p className="hint">Applied when you start a new video.</p>
          </details>
          <div className="model-note">
            <span>DMD 1.3B <b>·</b> Text to video</span>
            <p>This model uses text input. Image input belongs to the separate 5B world model.</p>
          </div>
        </aside>
        <section className="stage" aria-label="Generated video">
          <div className="stage-heading">
            <h2>Live video</h2>
            <span className="badge">∞ Continuous</span>
          </div>
          <div className="player">
            <ReactorView key={world?.world_id ?? 0} track="main_video" muted videoObjectFit="contain" className="video" />
            {(!ready || !world?.active || !world.completed_chunks) && (
              <div className="placeholder">
                <div className="play-icon" aria-hidden="true">
                  <svg viewBox="0 0 24 24" width="24" height="24"><path d="m9 5 10 7-10 7V5Z" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" /></svg>
                </div>
                <h3>{!ready ? "Ready for your first scene" : world?.active ? "Creating the first frames…" : "Ready when you are"}</h3>
                <p>{!ready ? "Connect, describe a scene, then start your video." : world?.active ? "Your video will appear here in a moment." : "Enter a prompt and press Start video."}</p>
              </div>
            )}
          </div>
          <div className="playback">
            <span>{world?.active ? world.paused ? "Paused · buffered frames may still play" : "● Generating" : ready ? "Connected · ready to start" : "Waiting to connect"}</span>
            <span>832 × 480 <b>·</b> Silent video</span>
          </div>
          <p className="progress" role="status">
            {world?.active ? `Video ${world.world_id} · ${world.completed_chunks} chunks · ${world.output_frames} frames${world.last_chunk_seconds === null ? "" : ` · ${world.last_chunk_seconds.toFixed(2)} s / chunk`}` : "No video generated yet."}
          </p>
          <footer>
            <span>Update prompt to continue. Start new video to begin again.</span>
            <a href="https://github.com/ziplab/History-Forcing" target="_blank" rel="noreferrer">Model & source ↗</a>
          </footer>
        </section>
      </div>
    </main>
  );
}

function App() {
  const [endpoint, setEndpoint] = useState(() =>
    new URLSearchParams(location.search).get("endpoint") ||
    import.meta.env.VITE_REACTOR_ENDPOINT || "http://localhost:8080",
  );
  const [active, setActive] = useState(endpoint);
  const endpointSelector = (
    <form className="endpoint" onSubmit={(e) => { e.preventDefault(); setActive(endpoint); }}>
      <label><span>Runtime URL</span>
        <input aria-label="Runtime URL" type="url" required value={endpoint} onChange={(e) => setEndpoint(e.target.value)} />
      </label>
      <button type="submit">Apply</button>
    </form>
  );
  return (
    <ReactorProvider key={active} modelName="history-forcing"
      modelTracks={[{ name: "main_video", kind: "video", direction: "recvonly" }]}
      local apiUrl={active} connectOptions={{ autoConnect: false }}>
      <Workspace endpointSelector={endpointSelector} />
    </ReactorProvider>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
