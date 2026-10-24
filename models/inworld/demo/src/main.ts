import { Reactor } from "@reactor-team/js-sdk";
import "./style.css";

document.querySelector<HTMLDivElement>("#app")!.innerHTML = `
<header>
  <a class="brand" href="https://reactor.inc" target="_blank" rel="noreferrer"><img src="/reactor-symbol.svg" alt="Reactor" width="32" height="23"><strong>InWorld</strong><span>on Reactor</span></a>
  <div class="connection-bar"><label class="endpoint-label" for="endpoint">Runtime endpoint</label><input id="endpoint" type="url" spellcheck="false" aria-label="Runtime endpoint" placeholder="http://localhost:8080"><span id="connection" class="connection-status" role="status">disconnected</span><button id="connect">Connect</button></div>
</header>
<main>
  <aside>
    <p class="eyebrow">01 / CREATE YOUR SCENE</p><h1>Let the story <br>continue.</h1><p class="intro">A continuous world of motion and sound.<br>Describe a scene, then shape what comes next.</p>
    <label class="field-title" for="prompt">Prompt <span>Scene · motion · sound</span></label><textarea id="prompt" rows="7" maxlength="8192" placeholder="Describe your scene, its motion, and what we hear…"></textarea>
    <button id="update" class="secondary" disabled>Update prompt</button><p class="hint">While generating, updates guide the next segment and keep the current story.</p>
    <div class="field-title input-heading">Visual input <span>Optional</span></div>
    <label class="upload-card" for="images"><span class="upload-icon" aria-hidden="true">＋</span><span><strong>Add reference images</strong><small>Up to 5 images · use &lt;Picture 1&gt; in your prompt</small></span><input id="images" class="file-input" type="file" accept="image/*" multiple></label>
    <label class="upload-card" for="parent"><span class="upload-icon" aria-hidden="true">↗</span><span><strong>Continue a video</strong><small>A 24 FPS video · continues its last 17 frames</small></span><input id="parent" class="file-input" type="file" accept="video/*"></label>
    <div id="selection" hidden><div id="previews"></div><button id="clear-inputs" class="text-button">Clear visual input</button></div>
    <p class="hint">No image required for text-to-video. Selecting files does not start generation.</p>
    <details class="settings"><summary>Generation settings</summary><div class="row"><label for="seed">Seed<input id="seed" type="number" min="0" max="2147483647" value="42"></label><label for="frames">Frames per segment<select id="frames"><option>24</option><option>48</option><option>72</option><option selected>120</option><option>240</option><option>360</option><option>464</option></select></label></div></details>
    <button id="start" class="primary" disabled>Start video <span aria-hidden="true">↗</span></button><p id="start-hint" class="hint">Connect to your Runtime to begin.</p>
    <p id="error" role="alert" hidden></p>
  </aside>
  <section class="stage">
    <div class="stage-heading"><div><p class="eyebrow">02 / LIVE OUTPUT</p><h2>Your story, unfolding.</h2></div><span class="tag">24 FPS <span>·</span> 48 kHz audio</span></div>
    <div class="video-shell" id="video-shell">
      <video id="video" playsinline aria-label="Generated InWorld video"></video>
      <div id="placeholder" class="placeholder"><div class="placeholder-mark" aria-hidden="true">▷</div><h3 id="placeholder-title">Your scene starts here</h3><p id="placeholder-detail">Connect, write a prompt, then press Start video.</p><span class="placeholder-tag">TEXT + IMAGE → AUDIO + VIDEO</span></div>
      <button id="enable-audio" class="enable-audio" hidden>Play with sound</button>
    </div>
    <div class="player-bar"><div class="playback-status"><span id="playback-label">Waiting to connect</span><span id="playback-time" hidden>00:00</span></div><div class="player-actions"><button id="pause" disabled title="Pause future generation; buffered media keeps playing">Pause</button><button id="sound" disabled aria-pressed="false">Sound on</button><button id="fullscreen" disabled>Fullscreen</button></div></div>
    <p id="progress" role="status">No video generated yet.</p>
    <div class="stage-note"><span>Change the prompt. Keep the story.</span><p>Video plays at a fixed 24 FPS. Generation can take longer than playback; gaps may occur. Audio is generated afresh for each segment.</p></div>
    <footer><span>Murea-Fast / InWorld</span><a href="https://reactor.inc" target="_blank" rel="noreferrer">Built on Reactor ↗</a></footer>
  </section>
</main>`;

const element = <T extends HTMLElement>(id: string) =>
  document.getElementById(id)! as T;
const button = (id: string) => element<HTMLButtonElement>(id);
const input = (id: string) => element<HTMLInputElement>(id);
const video = element<HTMLVideoElement>("video");
const prompt = element<HTMLTextAreaElement>("prompt");
input("endpoint").value = "http://localhost:8080";
try {
  input("endpoint").value =
    localStorage.getItem("inworld.endpoint") || input("endpoint").value;
} catch {
  /* Optional preference. */
}

let reactor: Reactor | undefined;
let ready = false;
let busy = false;
let paused = true;
let active = false;
let completed = 0;
let worldId = 0;
let media = new MediaStream();
let playbackStarted = false;
let mediaOrigin: number | undefined;
let frameCallback: number | undefined;
let previewUrls: string[] = [];

function showError(message: string) {
  element("error").textContent = message;
  element("error").hidden = !message;
}
function clearPlayer() {
  if (frameCallback !== undefined)
    video.cancelVideoFrameCallback(frameCallback);
  frameCallback = undefined;
  video.pause();
  video.srcObject = null;
  playbackStarted = false;
  mediaOrigin = undefined;
  element("playback-time").textContent = "00:00";
  element("playback-time").hidden = true;
  button("enable-audio").hidden = true;
}
function refresh() {
  button("start").disabled = !ready || busy || !prompt.value.trim();
  button("start").innerHTML =
    `${active ? "Start new video" : "Start video"} <span aria-hidden="true">↗</span>`;
  button("update").disabled = !ready || busy || !active || !prompt.value.trim();
  button("pause").disabled = !ready || busy || !active;
  button("pause").textContent = paused && active ? "Resume" : "Pause";
  button("connect").disabled = busy;
  input("endpoint").disabled = ready || busy;
  for (const id of ["images", "parent", "seed", "frames", "clear-inputs"]) {
    (element(id) as HTMLInputElement | HTMLButtonElement).disabled = busy;
  }
  element("connection").classList.toggle("live", ready);
  element("placeholder").hidden = playbackStarted;
  element("video-shell").classList.toggle("has-video", playbackStarted);
  button("sound").disabled = !playbackStarted;
  button("fullscreen").disabled = !playbackStarted;
  element("placeholder-title").textContent = active
    ? "Creating your first segment"
    : ready
      ? "Ready when you are"
      : "Your scene starts here";
  element("placeholder-detail").textContent = active
    ? "The first segment may take longer while the model warms up. Your video will appear here."
    : ready
      ? "Write a prompt, optionally add a visual input, then press Start video."
      : "Connect, write a prompt, then press Start video.";
  element("playback-label").textContent = !ready
    ? "Waiting to connect"
    : !active
      ? "Connected · waiting for input"
      : !playbackStarted
        ? "Generating first segment…"
        : paused
          ? "Generation paused · buffered video can still play"
          : "Live video";
  element("start-hint").textContent = !ready
    ? "Connect to your Runtime to begin."
    : !prompt.value.trim()
      ? "Add a prompt to start your video."
      : active
        ? "Start new video resets the story. Update prompt continues it."
        : "Your inputs are ready. Generation starts only when you press Start.";
}
async function playMedia() {
  const source = video.srcObject;
  try {
    await video.play();
    if (video.srcObject === source) button("enable-audio").hidden = true;
  } catch {
    if (source && video.srcObject === source)
      button("enable-audio").hidden = false;
  }
}
function attachGeneratedMedia() {
  // A received WebRTC track is not evidence of generated video. Keep the
  // player detached during connection, input staging and first-step warmup.
  if (
    !ready ||
    !active ||
    completed < 1 ||
    !media.getVideoTracks().length ||
    video.srcObject === media
  )
    return;
  video.srcObject = media;
  const onFrame = (_now: number, metadata: VideoFrameCallbackMetadata) => {
    if (!ready || !active || video.srcObject !== media) return;
    mediaOrigin ??= metadata.mediaTime;
    const seconds = Math.max(0, Math.floor(metadata.mediaTime - mediaOrigin));
    element("playback-time").textContent = `${Math.floor(seconds / 60)
      .toString()
      .padStart(2, "0")}:${(seconds % 60).toString().padStart(2, "0")}`;
    element("playback-time").hidden = false;
    if (!playbackStarted) {
      playbackStarted = true;
      refresh();
    }
    frameCallback = video.requestVideoFrameCallback(onFrame);
  };
  frameCallback = video.requestVideoFrameCallback(onFrame);
  void playMedia();
}
function resetConnection() {
  ready = false;
  active = false;
  paused = true;
  completed = 0;
  worldId = 0;
  clearPlayer();
  media = new MediaStream();
  element("connection").textContent = "disconnected";
  button("connect").textContent = "Connect";
  element("progress").textContent = "No video generated yet.";
  refresh();
}
async function act(fn: () => Promise<void>) {
  if (busy) return;
  busy = true;
  showError("");
  refresh();
  try {
    await fn();
  } catch (error) {
    showError(error instanceof Error ? error.message : String(error));
  } finally {
    busy = false;
    refresh();
  }
}
async function command(name: string, data: Record<string, unknown> = {}) {
  if (!reactor || !ready) throw new Error("Connect before sending inputs.");
  const answer = await reactor.sendCommand(name, data);
  if (!answer)
    throw new Error(element("error").textContent || "The command was refused.");
}
button("connect").onclick = () =>
  void act(async () => {
    if (reactor) {
      const client = reactor;
      reactor = undefined;
      resetConnection();
      await client.disconnect();
      return;
    }
    const endpoint = input("endpoint").value.trim();
    const url = new URL(endpoint);
    if (!["http:", "https:"].includes(url.protocol))
      throw new Error("Use an http:// or https:// Runtime endpoint.");
    try {
      localStorage.setItem("inworld.endpoint", endpoint);
    } catch {
      /* Optional preference. */
    }
    media = new MediaStream();
    const client = new Reactor({
      modelName: "inworld",
      apiUrl: endpoint,
      local: true,
      controlRequestTimeoutMs: 180000,
      modelTracks: [
        { name: "main_video", kind: "video", direction: "recvonly" },
        { name: "main_audio", kind: "audio", direction: "recvonly" },
      ],
    });
    reactor = client;
    client.on("statusChanged", (status) => {
      if (reactor !== client) return;
      ready = status === "ready";
      element("connection").textContent = status;
      button("connect").textContent =
        status === "disconnected" ? "Connect" : "Disconnect";
      if (status === "disconnected") {
        reactor = undefined;
        resetConnection();
      }
      refresh();
      attachGeneratedMedia();
    });
    client.on("error", (error) => {
      if (reactor === client) showError(error.message);
    });
    client.on("trackReceived", (_name, track) => {
      if (reactor !== client) return;
      if (!media.getTracks().some((t) => t.id === track.id))
        media.addTrack(track);
      attachGeneratedMedia();
    });
    client.on("message", (message) => {
      if (reactor !== client || message.type !== "state_update") return;
      const state = message.data as Record<string, unknown>;
      const nextWorld = Number(state.world_id);
      if (nextWorld !== worldId) clearPlayer();
      worldId = nextWorld;
      paused = Boolean(state.paused);
      active = worldId > 0;
      completed = Number(state.completed_segments);
      const seconds =
        state.last_segment_seconds == null
          ? ""
          : ` · ${Number(state.last_segment_seconds).toFixed(2)} s / segment`;
      element("progress").textContent = active
        ? `Video ${worldId} · ${completed} completed segments${seconds}`
        : "No video generated yet.";
      refresh();
      attachGeneratedMedia();
    });
    try {
      await client.connect();
    } catch (error) {
      if (reactor === client) reactor = undefined;
      try {
        await client.disconnect();
      } finally {
        resetConnection();
      }
      throw error;
    }
  });
function updateSelection() {
  previewUrls.forEach((url) => URL.revokeObjectURL(url));
  previewUrls = [];
  const previews = element("previews");
  previews.replaceChildren();
  const images = Array.from(input("images").files || []);
  const parent = input("parent").files?.[0];
  element("selection").hidden = !images.length && !parent;
  for (const [index, file] of images.entries()) {
    const item = document.createElement("div");
    item.className = "preview-item";
    const img = document.createElement("img");
    img.src = URL.createObjectURL(file);
    previewUrls.push(img.src);
    img.alt = `Reference ${index + 1}`;
    const name = document.createElement("span");
    name.textContent = `${index + 1}. ${file.name}`;
    item.append(img, name);
    previews.append(item);
  }
  if (parent) {
    const name = document.createElement("p");
    name.textContent = parent.name;
    previews.append(name);
  }
}
input("images").onchange = () => {
  if (input("images").files?.length) input("parent").value = "";
  updateSelection();
};
input("parent").onchange = () => {
  if (input("parent").files?.length) input("images").value = "";
  updateSelection();
};
button("clear-inputs").onclick = () => {
  input("images").value = "";
  input("parent").value = "";
  updateSelection();
};
prompt.oninput = refresh;
button("start").onclick = () =>
  void act(async () => {
    if (!prompt.value.trim())
      throw new Error("Enter a prompt before starting.");
    const images = Array.from(input("images").files || []);
    if (images.length > 5)
      throw new Error("Choose at most five reference images.");
    const seed = Number(input("seed").value);
    if (!Number.isInteger(seed) || seed < 0 || seed > 2147483647)
      throw new Error("Seed must be an integer from 0 to 2147483647.");
    await command("set_prompt", { prompt: prompt.value });
    await command("clear_media");
    for (let i = 0; i < images.length; i++) {
      const image = await reactor!.uploadFile(images[i]);
      await command("set_reference", { image, slot: i + 1 });
    }
    const parent = input("parent").files?.[0];
    if (parent)
      await command("set_video", { video: await reactor!.uploadFile(parent) });
    await command("start", {
      seed,
      new_frames: Number(element<HTMLSelectElement>("frames").value),
    });
  });
button("update").onclick = () =>
  void act(() => command("set_prompt", { prompt: prompt.value }));
button("pause").onclick = () =>
  void act(() => command("set_paused", { paused: !paused }));
button("sound").onclick = () => {
  video.muted = !video.muted;
  button("sound").textContent = video.muted ? "Sound off" : "Sound on";
  button("sound").setAttribute("aria-pressed", String(video.muted));
};
button("enable-audio").onclick = () => {
  void playMedia();
};
button("fullscreen").onclick = () => {
  const operation = document.fullscreenElement
    ? document.exitFullscreen()
    : element("video-shell").requestFullscreen();
  void operation.catch(() =>
    showError("Fullscreen is not available in this browser."),
  );
};
refresh();
