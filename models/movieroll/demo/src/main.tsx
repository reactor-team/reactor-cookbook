import { useEffect, useRef, useState, type FormEvent } from "react";
import { createRoot } from "react-dom/client";
import { ConflictError, ReactorProvider, ReactorView, useReactor, useReactorMessage } from "@reactor-team/js-sdk";
import "./style.css";

type Mode = "t2v" | "i2v";
type World = {
  prompt: string;
  active: boolean;
  paused: boolean;
  has_image: boolean;
  mode: Mode | null;
  seed: number | null;
  world_id: number;
  completed_chunks: number;
  output_frames: number;
  last_chunk_seconds: number | null;
  prompt_updates_remaining: number;
  end_reason: "stopped" | "history_full" | null;
};

function Workspace({ endpoint, onEndpoint }: { endpoint: string; onEndpoint: (url: string) => void }) {
  const { status, connect, disconnect, sendCommand, uploadFile, lastError } = useReactor((s) => ({
    status: s.status, connect: s.connect, disconnect: s.disconnect,
    sendCommand: s.sendCommand, uploadFile: s.uploadFile, lastError: s.lastError,
  }));
  const [world, setWorld] = useState<World | null>(null);
  const [prompt, setPrompt] = useState("");
  const [seed, setSeed] = useState(0);
  const [mode, setMode] = useState<Mode>("t2v");
  const [url, setUrl] = useState(endpoint);
  const [preview, setPreview] = useState("");
  const [busy, setBusy] = useState(false);
  const [connectionBusy, setConnectionBusy] = useState(false);
  const [error, setError] = useState("");
  const handledError = useRef(lastError);
  const dirty = useRef(false);
  const epoch = useRef(0);
  const ready = status === "ready";
  const accept = (accepted: World) => {
    setWorld(accepted);
    if (!dirty.current) setPrompt(accepted.prompt);
  };
  useReactorMessage((message) => {
    if (message.type === "state_update") accept(message.data as World);
  });
  useEffect(() => {
    if (lastError && lastError !== handledError.current) {
      handledError.current = lastError;
      if (isClosing(lastError) && (connectionBusy || ready)) return;
      setError(lastError.message);
    }
  }, [lastError, connectionBusy, ready]);
  useEffect(() => {
    if (!ready) return;
    const started = epoch.current;
    let current = true;
    // SDK 3.4.0 can miss the joining push while a shared video is paused.
    // One addressed read establishes the snapshot without resuming or polling.
    void sendCommand("get_state").then((reply) => {
      if (current && reply !== undefined && started === epoch.current) accept(reply.data as World);
    }).catch((e: unknown) => {
      if (current) setError(e instanceof Error ? e.message : String(e));
    });
    return () => { current = false; };
  }, [ready, sendCommand]);
  useEffect(() => {
    if (status === "disconnected") {
      epoch.current += 1;
      setWorld(null);
      setPreview("");
    }
  }, [status]);
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);

  function clearError() { setError(""); handledError.current = lastError; }
  async function perform(action: () => Promise<unknown>) {
    clearError(); setBusy(true);
    try { await action(); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }
  async function command(name: string, args: Record<string, unknown> = {}) {
    const started = epoch.current;
    const reply = await sendCommand(name, args);
    // SDK command refusals resolve undefined and report their reason via error.
    if (reply === undefined || started !== epoch.current) return;
    const accepted = reply.data as World;
    accept(accepted);
    if (name === "start" || name === "set_prompt") {
      dirty.current = false; setPrompt(accepted.prompt);
    }
    return accepted;
  }
  async function endSession() {
    epoch.current += 1;
    await disconnect();
    setWorld(null); setPreview("");
  }
  function isClosing(error: unknown): error is ConflictError {
    return error instanceof ConflictError && error.message.includes("while closing");
  }
  async function connectWhenReady() {
    const deadline = performance.now() + 20_000;
    for (;;) {
      try { await connect(); return; }
      catch (e) {
        // Local disconnect can finish before Runtime's old session teardown.
        // Retry only that explicit conflict, through the SDK, with a bound.
        if (!isClosing(e) || performance.now() >= deadline) throw e;
        await new Promise((resolve) => window.setTimeout(resolve, 500));
      }
    }
  }
  async function connectionAction() {
    clearError(); setConnectionBusy(true);
    try { if (ready) await endSession(); else await connectWhenReady(); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setConnectionBusy(false); }
  }
  async function applyEndpoint(event: FormEvent) {
    event.preventDefault();
    clearError(); setConnectionBusy(true);
    try {
      const parsed = new URL(url);
      if (!["http:", "https:"].includes(parsed.protocol)) throw new Error("Use an HTTP or HTTPS Runtime URL");
      await endSession();
      onEndpoint(parsed.href.replace(/\/$/, ""));
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setConnectionBusy(false); }
  }

  return <main>
    <header className="app-header">
      <a className="brand" href="https://reactor.inc" target="_blank" rel="noreferrer" aria-label="MovieRoll on Reactor">
        <img src="/brand/reactor-mark-white.svg" alt="" width="33" height="24" />
        <span className="brand-title"><strong>MovieRoll</strong><span>on Reactor</span></span>
      </a>
      <form className="endpoint" onSubmit={(e) => void applyEndpoint(e)}>
        <label><span>Runtime URL</span><input aria-label="Runtime URL" type="url" required value={url} onChange={(e) => setUrl(e.target.value)} /></label>
        <button type="submit" disabled={connectionBusy}>Apply</button>
      </form>
      <div className="connection-controls">
        <span className={`connection ${ready ? "connected" : ""}`} role="status"><i aria-hidden="true" />{ready ? "Connected" : status === "disconnected" ? "Offline" : status}</span>
        <button disabled={connectionBusy || status === "connecting" || status === "waiting"} onClick={() => void connectionAction()}>{ready ? "Disconnect" : "Connect"}</button>
      </div>
    </header>
    <div className="workspace">
      <aside>
        <div className="panel-intro"><p className="eyebrow">CONTINUOUS VIDEO</p><h1>Keep the story rolling.</h1><p>Start with a scene. Guide what happens next.</p></div>
        <label>Start from
          <select aria-label="Generation mode" value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
            <option value="t2v">Text to video</option><option value="i2v">Image to video</option>
          </select>
        </label>
        {mode === "i2v" && <div className="image-input">
          <label>Initial image<input aria-label="Initial image" type="file" accept="image/png,image/jpeg,image/webp" disabled={!ready || busy} onChange={(e) => {
            const file = e.target.files?.[0]; e.target.value = "";
            if (!file) return;
            void perform(async () => {
              if (file.size > 20 * 1024 * 1024) throw new Error("Choose an image smaller than 20 MiB");
              const started = epoch.current;
              const reference = await uploadFile(file);
              if (started !== epoch.current) return;
              const accepted = await command("set_image", { image: reference });
              if (accepted) setPreview(URL.createObjectURL(file));
            });
          }} /></label>
          {preview && <img src={preview} alt="Selected initial image" />}
          <p className="hint">{world?.has_image ? "Image ready for your next Start." : "Connect and upload an image, then press Start video."} Images are centered and cropped to fit.</p>
          {world?.has_image && <button disabled={busy} onClick={() => void perform(async () => { if (await command("clear_image")) setPreview(""); })}>Remove image</button>}
        </div>}
        <label className="prompt-label">Scene description
          <textarea aria-label="Prompt" rows={8} maxLength={8192} value={prompt} placeholder="A clownfish swims gently through the air, circling a young man seated in a softly lit living room. The camera stays still…" onChange={(e) => { dirty.current = true; setPrompt(e.target.value); }} />
        </label>
        <div className="prompt-actions">
          <button className="primary" disabled={!ready || busy || !prompt.trim() || (mode === "i2v" && !world?.has_image) || !Number.isInteger(seed) || seed < 0 || seed > 4294967295} onClick={() => void perform(() => command("start", { prompt, seed, mode }))}>{world?.active ? "Start new video" : "Start video"}<span aria-hidden="true">↗</span></button>
          <button disabled={!ready || busy || !world?.active || !prompt.trim()} onClick={() => void perform(() => command("set_prompt", { prompt }))}>Update prompt</button>
        </div>
        <p className="hint">Use a full scene description for each update. Changes blend through several chunks while the same video continues.</p>
        {world?.active && <p className="hint">{world.prompt_updates_remaining} prompt updates remaining in this video.</p>}
        <div className="actions">
          <button disabled={!ready || busy || !world?.active} onClick={() => void perform(() => command("set_paused", { paused: !world?.paused }))}>{world?.paused ? "Resume" : "Pause"}</button>
          <button disabled={!ready || busy || !world?.active} onClick={() => void perform(() => command("reset"))}>Reset</button>
          <button disabled={!ready || busy || !world?.active} onClick={() => void perform(() => command("stop"))}>Stop</button>
        </div>
        {error && <p className="error" role="alert">{error}</p>}
        {world?.end_reason === "history_full" && <p className="error" role="alert">The video reached its temporary storage limit. Start a new video to continue.</p>}
        <details className="settings"><summary>Generation settings</summary><label>Seed<input aria-label="Seed" type="number" min={0} max={4294967295} step={1} value={seed} onChange={(e) => setSeed(Number(e.target.value))} /></label><p className="hint">Applied when you start a new video.</p></details>
      </aside>
      <section className="stage" aria-label="Generated video">
        <div className="stage-heading"><h2>Live video</h2><span className="badge">∞ Continuous</span></div>
        <div className="player">
          <ReactorView key={world?.world_id ?? 0} track="main_video" muted videoObjectFit="contain" className="video" />
          {(!ready || !world?.active || !world.completed_chunks) && <div className="placeholder"><div className="play-icon" aria-hidden="true">▷</div><h3>{world?.active ? "Creating the first frames…" : "Your story starts here"}</h3><p>{!ready ? "Connect to your Runtime to begin." : world?.active ? "Preparing the rolling window and first frames." : "Choose text or an image, describe a scene, and press Start video."}</p></div>}
        </div>
        <div className="playback"><span>{world?.active ? world.paused ? "Paused · buffered frames may still play" : "● Generating" : ready ? "Connected · ready to start" : "Waiting to connect"}</span><span>864 × 480 <b>·</b> Silent video</span></div>
        <p className="progress" role="status">{world?.active ? `Video ${world.world_id} · ${world.completed_chunks} chunks · ${world.output_frames} frames${world.last_chunk_seconds === null ? "" : ` · ${world.last_chunk_seconds.toFixed(2)} s / chunk`}` : "No video generated yet."}</p>
        <footer><span>Update prompt to continue. Start new video to begin again.</span><a href="https://github.com/Tencent-Hunyuan/MovieRoll" target="_blank" rel="noreferrer">Model & source ↗</a></footer>
      </section>
    </div>
  </main>;
}

function App() {
  const [endpoint, setEndpoint] = useState(() => new URLSearchParams(location.search).get("endpoint") || import.meta.env.VITE_REACTOR_ENDPOINT || "http://localhost:8080");
  return <ReactorProvider key={endpoint} modelName="your-org/movieroll" modelTracks={[{ name: "main_video", kind: "video", direction: "recvonly" }]} local apiUrl={endpoint} connectOptions={{ autoConnect: false }}>
    <Workspace endpoint={endpoint} onEndpoint={setEndpoint} />
  </ReactorProvider>;
}
createRoot(document.getElementById("root")!).render(<App />);
