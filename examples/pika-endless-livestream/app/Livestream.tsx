"use client";

import { useEffect, useRef, useState } from "react";
import {
  PikaH3FlexMainVideoView,
  PikaH3FlexProvider,
  usePikaH3Flex,
  usePikaH3FlexMessage,
  usePikaH3FlexTrack,
} from "@reactor-models/pika-h3-flex";
import { InputsPanel } from "./InputsPanel";
import { KeysBar, type ServerKeys } from "./KeysBar";
import { EMPTY_KEYS, keyHeaders, loadKeys, saveKeys, type Keys, type Writer } from "./lib/keys";
import { EXAMPLES, exampleInputs, type Example } from "./lib/examples";
import { RECENT_PROMPTS, type ClipFormat, type DirectorRequest } from "./lib/director";
import { EMPTY_INPUTS, frameAlignmentLine, frameSet, hasReferences, referenceList, thumbnail, type Inputs, type Item } from "./lib/inputs";
import type { Story } from "./lib/story";
import type { Verdict } from "./lib/check";

// How many clips to keep queued (building or built) ahead of the one playing.
const CLIPS_AHEAD = 2;
// Clip length in seconds: whole seconds 5–15 are accepted.
const CLIP_SECONDS = 10 as const;
// Ask for the next episode when this many beats of the current one are left.
const PREFETCH_AT = 4;
const DEFAULT_PROMPT = EXAMPLES[0].concept;

// Which Reactor API runs the stream. Streaming: one realtime session whose
// playout queue plays each clip as soon as it is built, with latent continuation
// between clips. Generations: each clip is its own POST /generations, and the
// page plays the finished videos back to back.
type Api = "streaming" | "generations";
// How often to check on a generation that is still building.
const POLL_MS = 2000;

// One JWT per page load (or per Reactor key). A session can only be driven by the token that created it.
let token: Promise<string> | null = null;
function getToken(): Promise<string> {
  token ??= fetch("/api/token", { cache: "no-store", headers: keyHeaders() }).then(async (res) => {
    const data = (await res.json()) as { jwt?: string; error?: string };
    if (!res.ok || !data.jwt) {
      token = null;
      throw new Error(data.error ?? "Could not get a session token.");
    }
    return data.jwt;
  });
  return token;
}

type Model = ReturnType<typeof usePikaH3Flex>;
type FileRef = Awaited<ReturnType<Model["uploadFile"]>>;
type Clip = { id: string; prompt: string; seconds: number };
type QueueItem = { clip_id: string; prompt: string; seconds: number };
/** Which episode and beat a queued clip shows. */
type BeatRef = { episode: number; index: number; segment: string };
type Plan = { story: Story; episode: number; index: number };
type Frames = { first: Item | null; last: Item | null };
/** A clip whose prompt is written and waits only for its inputs to be sent. */
type Prepared = { from: string | null; format: ClipFormat; inputs: Inputs; plan: Plan | null; text: string; steer: string };
/** A clip on the generation API: queued, then built (url set) once its video is downloaded and checked. */
type GenClip = {
  prompt: string;
  seconds: number;
  prep: Prepared;
  url?: string;
  lastFrame?: string;
  score?: number;
  /** A second take, written after the first failed its check. */
  retake?: { verdict: Verdict; first: { url: string; lastFrame?: string; score: number } };
};

export function Livestream() {
  return (
    <PikaH3FlexProvider jwtToken={getToken}>
      <Stream />
    </PikaH3FlexProvider>
  );
}

function Stream() {
  const model = usePikaH3Flex();
  const audioTrack = usePikaH3FlexTrack("main_audio");
  const audio = useRef<HTMLAudioElement>(null);

  // What the viewer is typing, and the direction the stream follows (set by Steer).
  const [draft, setDraft] = useState(DEFAULT_PROMPT);
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  // Pika's raw inputs, and which kind the panel shows.
  const [inputs, setInputs] = useState<Inputs>(EMPTY_INPUTS);
  const [tab, setTab] = useState<"references" | "frames">("references");
  const [exampleId, setExampleId] = useState<string | null>(EXAMPLES[0].id);
  const [loadingExample, setLoadingExample] = useState(false);
  const [api, setApi] = useState<Api>("streaming");
  const [keys, setKeys] = useState<Keys>(EMPTY_KEYS);
  const [live, setLive] = useState(false);
  const [starting, setStarting] = useState(false);
  const [log, setLog] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  // Clips we queued that haven't started playing yet: the buffer ahead of the playhead.
  const [ahead, setAhead] = useState<string[]>([]);
  const [playing, setPlaying] = useState<Clip | null>(null);
  const [enqueuing, setEnqueuing] = useState(false);
  const [builtTick, setBuiltTick] = useState(0);
  const [prepTick, setPrepTick] = useState(0);
  // The model's own queues, from queue_update: built clips waiting, and clips building.
  const [ready, setReady] = useState<QueueItem[]>([]);
  const [building, setBuilding] = useState<QueueItem[]>([]);
  // When on, Claude plans episodes (Sonnet) and writes each clip (Haiku).
  const [enhance, setEnhance] = useState(true);
  const [soundBlocked, setSoundBlocked] = useState(false);
  // The episode on air, which beat the next clip shows, and what the writers are doing.
  const [story, setStory] = useState<Story | null>(null);
  const [episode, setEpisode] = useState(1);
  const [nextIndex, setNextIndex] = useState(0);
  const [writing, setWriting] = useState<null | "first" | "rewrite" | "next">(null);
  // The sidebar: the episode's story arc, or the raw prompts of the clips on air and queued.
  const [side, setSide] = useState<"story" | "prompts">("story");
  // Header stats.
  const [aired, setAired] = useState(0);
  const [buildTimes, setBuildTimes] = useState<number[]>([]);
  const [stalls, setStalls] = useState(0);
  const [liveSince, setLiveSince] = useState<number | null>(null);
  const [now, setNow] = useState(() => Date.now());

  // The latest prompts sent (without the alignment line), oldest first, for the director.
  const recent = useRef<string[]>([]);
  // The steer text the last queued clip was written from, so a direction is acted on once.
  const applied = useRef<string | null>(null);
  // The last clip we queued; the next one continues from it once it is built.
  const tip = useRef<string | null>(null);
  const built = useRef(new Set<string>());
  const queuedAt = useRef(new Map<string, number>());
  const readyCount = useRef(0);
  // Uploads, once per session per input, and frames waiting for the next clip.
  const uploads = useRef(new Map<string, FileRef>());
  const thumbs = useRef(new Map<string, string>());
  const armed = useRef<Frames | null>(null);
  // The next clip, written ahead while the clip it continues builds.
  const prepared = useRef<Prepared | null>(null);
  const preparing = useRef(false);
  // Story bookkeeping the loop reads between renders.
  const storyRef = useRef<Story | null>(null);
  const episodeRef = useRef(1);
  const beatIndex = useRef(0);
  const nextStory = useRef<Story | null>(null);
  const fetchingNext = useRef(false);
  const beatOf = useRef(new Map<string, BeatRef>());

  // The generation API's own queue: clips in the order they will play, each
  // one's video once built, uploads as file ids, and which run a poll belongs to.
  const genVideo = useRef<HTMLVideoElement>(null);
  const genOrder = useRef<string[]>([]);
  const genClips = useRef(new Map<string, GenClip>());
  const genUploads = useRef(new Map<string, string>());
  const genPlaying = useRef<string | null>(null);
  const genRun = useRef(0);
  // The last frame of the latest clip that passed, for the check's body comparison.
  const genPrevFrame = useRef<string | undefined>(undefined);

  // The hook returns new functions every render; read them through a ref.
  const latest = useRef({ model, prompt, enhance, live, inputs, api });
  latest.current = { model, prompt, enhance, live, inputs, api };

  // Which keys .env.local has, so the page can tell whether a writer will run.
  const [serverKeys, setServerKeys] = useState<ServerKeys | null>(null);
  useEffect(() => {
    setKeys(loadKeys());
    void fetch("/api/keys", { cache: "no-store" })
      .then((r) => r.json() as Promise<ServerKeys>)
      .then(setServerKeys)
      .catch(() => setServerKeys(null));
  }, []);

  const say = (line: string) =>
    setLog((l) => [`${new Date().toLocaleTimeString()}  ${line}`, ...l].slice(0, 200));
  const short = (id: string) => id.slice(0, 8);

  function showStory(s: Story | null, ep: number, index: number) {
    storyRef.current = s;
    episodeRef.current = ep;
    beatIndex.current = index;
    setStory(s);
    setEpisode(ep);
    setNextIndex(index);
  }

  usePikaH3FlexMessage((m) => {
    if (m.type === "queue_update") {
      setReady(m.playout);
      setBuilding(m.generation);
      readyCount.current = m.playout.length;
    } else if (m.type === "clip_generated") {
      built.current.add(m.clip.clip_id);
      setBuiltTick((n) => n + 1);
      const t = queuedAt.current.get(m.clip.clip_id);
      if (t) setBuildTimes((b) => [...b, (performance.now() - t) / 1000].slice(-20));
      const waited = t ? ((performance.now() - t) / 1000).toFixed(1) : "?";
      say(`built   ${short(m.clip.clip_id)}  (${m.clip.seconds.toFixed(1)}s clip, ${m.clip.adapter ?? "?"}, ${waited}s after enqueue)`);
    } else if (m.type === "clip_started") {
      setPlaying({ id: m.clip.clip_id, prompt: m.clip.prompt, seconds: m.clip.seconds });
      setAired((n) => n + 1);
      setAhead((a) => a.filter((id) => id !== m.clip.clip_id));
      say(`playing ${short(m.clip.clip_id)}`);
    } else if (m.type === "clip_finished") {
      // Nothing built to play next: the stream holds the last frame until a build lands.
      if (latest.current.live && readyCount.current === 0) setStalls((n) => n + 1);
      say(`done    ${short(m.clip.clip_id)}`);
    } else if (m.type === "clip_failed") {
      setAhead((a) => a.filter((id) => id !== m.clip.clip_id));
      // Continue from the last clip that did build.
      if (tip.current === m.clip.clip_id) tip.current = m.clip.continue_from_clip_id ?? null;
      setError(`A clip failed to build: ${m.reason}`);
      say(`FAILED  ${short(m.clip.clip_id)}: ${m.reason}`);
    }
  });

  // Session setup once connected: autoplay on, and hold the last frame between
  // clips instead of flashing black if the buffer ever runs dry.
  useEffect(() => {
    if (model.status === "ready") {
      say("connected");
      const { model: m } = latest.current;
      void m.setAutoplay({ enabled: true });
      void m.setFlushOnClipEnd({ enabled: false });
      // queue_update isn't broadcast on connect, so ask once.
      void m.getQueue().then((q) => {
        if (!q) return;
        setReady(q.playout);
        setBuilding(q.generation);
        readyCount.current = q.playout.length;
      });
    }
    if (model.status === "disconnected") {
      tip.current = null;
      recent.current = [];
      applied.current = null;
      readyCount.current = 0;
      nextStory.current = null;
      armed.current = null;
      prepared.current = null;
      built.current.clear();
      uploads.current.clear();
      beatOf.current.clear();
      setAhead([]);
      setPlaying(null);
      setReady([]);
      setBuilding([]);
      setLive(false);
      setLiveSince(null);
    }
  }, [model.status]);

  useEffect(() => {
    if (!model.lastError) return;
    setError(model.lastError.message);
    say(`error: ${model.lastError.message}`);
  }, [model.lastError]);

  // The live timer.
  useEffect(() => {
    if (liveSince === null) return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [liveSince]);

  // The loop, in two steps so the director never holds up the stream:
  //  1. prepare: as soon as there is room in the buffer, write the next clip's
  //     prompt, while the clip it continues is still building;
  //  2. send: once that clip is built, enqueue at once.
  // Clips continue the one before with a latent continuation, which carries
  // motion, camera, sound and the references across the cut, and needs that
  // clip built first. A clip with frames starts fresh and needs no wait.
  useEffect(() => {
    const engineReady = api === "generations" || model.status === "ready";
    if (!live || !engineReady || enqueuing || ahead.length >= CLIPS_AHEAD) return;
    const from = tip.current;
    const next = prepared.current;
    if (!next || next.from !== from) {
      if (!preparing.current) void prepare(from);
      return;
    }
    if (next.format.kind !== "fl2va" && from && !built.current.has(from)) return;
    prepared.current = null;
    void send(next);
  }, [live, api, model.status, enqueuing, ahead.length, builtTick, prepTick]);

  /** Write the next clip's prompt now, against the inputs as they are now. */
  async function prepare(from: string | null) {
    preparing.current = true;
    const { prompt: steer, enhance: useClaude, inputs: current } = latest.current;
    const frames = armed.current;
    armed.current = null;
    const snapshot: Inputs = frames ? { ...EMPTY_INPUTS, first: frames.first, last: frames.last } : current;
    const format = clipFormat(snapshot, Boolean(frames), frames ? null : from);
    const plan = useClaude ? takeBeat() : null;
    const text = useClaude ? await direct(steer, plan, format) : steer.trim();
    prepared.current = { from, format, inputs: snapshot, plan, text, steer: steer.trim() };
    preparing.current = false;
    setPrepTick((n) => n + 1);
  }

  async function send(next: Prepared) {
    if (latest.current.api === "generations") return sendGeneration(next);
    setEnqueuing(true);
    const m = latest.current.model;
    const { format, text, plan, from } = next;
    const started = performance.now();
    const body = {
      prompt: format.kind === "fl2va" ? `${frameAlignmentLine(format.frames, CLIP_SECONDS)}\n\n${text}` : text,
      seconds: CLIP_SECONDS,
    };
    let reply = await m.enqueue({ ...body, ...(await clipParams(m, next)) });
    if (!reply && uploads.current.size) {
      // An upload may not be reusable across clips: upload again once and retry.
      say("enqueue refused; uploading the inputs again and retrying");
      uploads.current.clear();
      reply = await m.enqueue({ ...body, ...(await clipParams(m, next)) });
    }
    if (!reply) {
      // Refused: the reason arrives on model.lastError. Stop instead of retrying in a loop.
      say("enqueue refused, stopping");
      setLive(false);
    } else {
      const id = reply.clip.clip_id;
      tip.current = id;
      recent.current = [...recent.current, text].slice(-RECENT_PROMPTS);
      applied.current = next.steer;
      queuedAt.current.set(id, started);
      if (plan) beatOf.current.set(id, { episode: plan.episode, index: plan.index, segment: plan.story.beats[plan.index].segment });
      setAhead((a) => [...a, id]);
      const what = format.kind === "fl2va" ? ` · frames: ${format.frames}` : format.kind === "ref2va" ? ` · ${format.pictures}p ${format.audios}a ${format.videos}v${format.tail ? " +tail" : ""}` : "";
      say(`queued  ${short(id)}${from && format.kind === "ref2va" && format.tail ? ` (continues ${short(from)})` : ""}${what}${plan ? ` · ep ${plan.episode} beat ${plan.index + 1}` : ""}`);
    }
    setEnqueuing(false);
  }

  /** Start one clip on the generation API and watch it until its video is ready. */
  async function sendGeneration(next: Prepared) {
    setEnqueuing(true);
    const run = genRun.current;
    const { format, text, plan } = next;
    const started = performance.now();
    try {
      const id = await createGeneration(next, text);
      if (run !== genRun.current) return;
      if (!id) {
        setLive(false);
        return;
      }
      recent.current = [...recent.current, text].slice(-RECENT_PROMPTS);
      applied.current = next.steer;
      queuedAt.current.set(id, started);
      if (plan) beatOf.current.set(id, { episode: plan.episode, index: plan.index, segment: plan.story.beats[plan.index].segment });
      genOrder.current.push(id);
      setAhead((a) => [...a, id]);
      showGenQueue();
      say(`queued  ${short(id)} (generation)${format.kind === "fl2va" ? ` · frames: ${format.frames}` : ""}${plan ? ` · ep ${plan.episode} beat ${plan.index + 1}` : ""}`);
      void watchGeneration(id, run);
    } catch (err) {
      if (run !== genRun.current) return;
      setError(err instanceof Error ? err.message : "The generation request failed.");
      say("generation request failed, stopping");
      setLive(false);
    } finally {
      setEnqueuing(false);
    }
  }

  /** POST one generation. Returns its id, or null when Reactor refused it (the reason goes to the error line). */
  async function createGeneration(next: Prepared, text: string): Promise<string | null> {
    const { format } = next;
    const prompt = format.kind === "fl2va" ? `${frameAlignmentLine(format.frames, CLIP_SECONDS)}\n\n${text}` : text;
    const input = { prompt, seconds: CLIP_SECONDS, aspect: "16:9", ...(await generationFiles(next)) };
    const res = await fetch("/api/generations", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...keyHeaders() },
      body: JSON.stringify({ input }),
    });
    const data = (await res.json().catch(() => null)) as { id?: string; error?: string; message?: string } | null;
    if (!res.ok || !data?.id) {
      const reason = data?.message ?? data?.error ?? `status ${res.status}`;
      setError(`The generation API refused the clip: ${reason}`);
      say(`generation refused (${reason})`);
      return null;
    }
    genClips.current.set(data.id, { prompt, seconds: CLIP_SECONDS, prep: { ...next, text } });
    return data.id;
  }

  /** Poll a generation; once it succeeds, download the video so playback never waits on the network. */
  async function watchGeneration(id: string, run: number) {
    while (run === genRun.current) {
      await new Promise((r) => setTimeout(r, POLL_MS));
      if (run !== genRun.current) return;
      const res = await fetch(`/api/generations/${id}`, { cache: "no-store", headers: keyHeaders() }).catch(() => null);
      if (!res) continue;
      const data = (await res.json().catch(() => null)) as {
        status?: string;
        files?: { name: string; url: string }[] | null;
        error?: { code: string; message: string } | string | null;
      } | null;
      if (!res.ok && res.status !== 429 && res.status < 500) {
        failGeneration(id, typeof data?.error === "string" ? data.error : `status ${res.status}`);
        return;
      }
      if (data?.status === "failed") {
        failGeneration(id, typeof data.error === "object" && data.error ? data.error.message : "failed");
        return;
      }
      if (data?.status !== "succeeded") continue;
      const file = data.files?.find((f) => /\.(mp4|webm|mov)$/i.test(f.name)) ?? data.files?.[0];
      const blob = file ? await fetch(file.url).then((r) => (r.ok ? r.blob() : null)).catch(() => null) : null;
      if (run !== genRun.current) return;
      if (!blob) {
        failGeneration(id, "the video could not be downloaded");
        return;
      }
      const clip = genClips.current.get(id);
      if (!clip) return;
      clip.url = URL.createObjectURL(blob);
      clip.lastFrame = data.files?.find((f) => /last_frame\.(png|jpe?g|webp)$/i.test(f.name))?.url;
      say(`built   ${short(id)}  (${((performance.now() - (queuedAt.current.get(id) ?? performance.now())) / 1000).toFixed(1)}s after request)`);
      await checkGeneration(id, run);
      return;
    }
  }

  /**
   * The shot check: with a story and a writer, judge the clip's last frame
   * against its beat and cast before it plays. A failed first take is written
   * again once, aimed at what failed; the better-scored take plays.
   */
  async function checkGeneration(id: string, run: number) {
    const clip = genClips.current.get(id);
    if (!clip) return;
    const plan = clip.prep.plan;
    if (!latest.current.enhance || !plan || !clip.lastFrame) return finishGeneration(id);
    const beat = plan.story.beats[plan.index];
    const onScreen = new Set(beat.on_screen.map((l) => l.toLowerCase()));
    const cast = [...plan.story.cast, ...plan.story.props.map((p) => ({ label: p.label, look: p.look }))]
      .filter((c) => onScreen.has(c.label.toLowerCase()))
      .map((c) => ({ label: c.label, look: c.look }));
    const started = performance.now();
    const res = await fetch("/api/check", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...keyHeaders() },
      body: JSON.stringify({ beat: beat.beat, prompt: clip.prompt, cast, frame: clip.lastFrame, earlier: genPrevFrame.current }),
    }).catch(() => null);
    const data = (await res?.json().catch(() => null)) as { verdict?: Verdict; error?: string } | null;
    if (run !== genRun.current) return;
    const verdict = data?.verdict;
    if (!verdict) {
      say(`check: ${data?.error ?? "no verdict"}; playing as built`);
      return finishGeneration(id);
    }
    clip.score = verdict.score;
    const secs = ((performance.now() - started) / 1000).toFixed(1);
    if (clip.retake) {
      // Second take: keep whichever take scored higher.
      const first = clip.retake.first;
      if (!verdict.passed && first.score > verdict.score) {
        URL.revokeObjectURL(clip.url!);
        clip.url = first.url;
        clip.lastFrame = first.lastFrame;
        say(`check   ${short(id)} retake ${verdict.score}/5 < first take ${first.score}/5: keeping the first (${secs}s)`);
      } else {
        URL.revokeObjectURL(first.url);
        say(`check   ${short(id)} retake ${verdict.passed ? "passed" : "failed"} ${verdict.score}/5: ${verdict.reason} (${secs}s)`);
      }
      return finishGeneration(id);
    }
    if (verdict.passed || !verdict.missed) {
      say(`check   ${short(id)} passed ${verdict.score}/5 (${secs}s)`);
      return finishGeneration(id);
    }
    say(`check   ${short(id)} FAILED (${verdict.missed}) ${verdict.score}/5: ${verdict.reason} · retaking`);
    const text = await direct(clip.prep.steer, plan, clip.prep.format, { prompt: clip.prompt, missed: verdict.missed, reason: verdict.reason });
    if (run !== genRun.current) return;
    const retakeId = await createGeneration(clip.prep, text).catch(() => null);
    if (run !== genRun.current) return;
    if (!retakeId) {
      say(`retake for ${short(id)} could not start; playing the first take`);
      return finishGeneration(id);
    }
    // The retake takes the first take's place in line.
    const second = genClips.current.get(retakeId)!;
    second.retake = { verdict, first: { url: clip.url!, lastFrame: clip.lastFrame, score: verdict.score } };
    genClips.current.delete(id);
    genOrder.current = genOrder.current.map((x) => (x === id ? retakeId : x));
    queuedAt.current.set(retakeId, queuedAt.current.get(id) ?? performance.now());
    const b = beatOf.current.get(id);
    if (b) beatOf.current.set(retakeId, b);
    setAhead((a) => a.map((x) => (x === id ? retakeId : x)));
    showGenQueue();
    say(`queued  ${short(retakeId)} (retake of ${short(id)})`);
    void watchGeneration(retakeId, run);
  }

  /** A clip is ready to play: count its build time and hand it to the player. */
  function finishGeneration(id: string) {
    const clip = genClips.current.get(id);
    if (!clip) return;
    if (clip.lastFrame) genPrevFrame.current = clip.lastFrame;
    built.current.add(id);
    const t = queuedAt.current.get(id);
    if (t) setBuildTimes((b) => [...b, (performance.now() - t) / 1000].slice(-20));
    setBuiltTick((n) => n + 1);
    showGenQueue();
    playNextGeneration();
  }

  function failGeneration(id: string, reason: string) {
    genOrder.current = genOrder.current.filter((x) => x !== id);
    genClips.current.delete(id);
    setAhead((a) => a.filter((x) => x !== id));
    setError(`A clip failed to build: ${reason}`);
    say(`FAILED  ${short(id)}: ${reason}`);
    showGenQueue();
    playNextGeneration();
  }

  /** Play the next clip in order once it is built. A later clip never jumps ahead of an earlier one. */
  function playNextGeneration() {
    const el = genVideo.current;
    const id = genOrder.current[0];
    if (!el || genPlaying.current || !id) return;
    const clip = genClips.current.get(id);
    if (!clip?.url || !built.current.has(id)) return;
    genOrder.current.shift();
    genPlaying.current = id;
    el.src = clip.url;
    void el.play().catch(() => {
      // No sound without a gesture: play muted and offer unmute.
      el.muted = true;
      setSoundBlocked(true);
      void el.play();
    });
    setPlaying({ id, prompt: clip.prompt, seconds: clip.seconds });
    setAired((n) => n + 1);
    setAhead((a) => a.filter((x) => x !== id));
    showGenQueue();
    say(`playing ${short(id)}`);
  }

  function onGenerationEnded() {
    const id = genPlaying.current;
    genPlaying.current = null;
    if (id) {
      say(`done    ${short(id)}`);
      const url = genClips.current.get(id)?.url;
      genClips.current.delete(id);
      // The element keeps showing the last frame until the next clip loads.
      if (url) setTimeout(() => URL.revokeObjectURL(url), 1000);
    }
    const next = genOrder.current[0];
    if (latest.current.live && (!next || !built.current.has(next))) setStalls((n) => n + 1);
    playNextGeneration();
  }

  /** The sidebar's Ready and Building lists, from the generation queue. */
  function showGenQueue() {
    const items = genOrder.current.map((clip_id) => ({ clip_id, ...genClips.current.get(clip_id)! })).filter((c) => c.prompt);
    setReady(items.filter((c) => built.current.has(c.clip_id)).map(({ clip_id, prompt, seconds }) => ({ clip_id, prompt, seconds })));
    setBuilding(items.filter((c) => !built.current.has(c.clip_id)).map(({ clip_id, prompt, seconds }) => ({ clip_id, prompt, seconds })));
  }

  /** Upload a file for the generation API once: reserve it, then PUT the bytes to the presigned URL. */
  async function uploadForGeneration(item: Item): Promise<string> {
    const cached = genUploads.current.get(item.id);
    if (cached) return cached;
    const { file } = item;
    const res = await fetch("/api/files", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...keyHeaders() },
      body: JSON.stringify({ name: file.name, size: file.size, mime_type: file.type }),
    });
    const data = (await res.json().catch(() => null)) as { file_id?: string; presigned_url?: string; error?: string } | null;
    if (!res.ok || !data?.file_id || !data.presigned_url) throw new Error(`Could not upload ${file.name}: ${data?.error ?? res.status}`);
    const put = await fetch(data.presigned_url, { method: "PUT", body: file, headers: { "Content-Type": file.type } });
    if (!put.ok) throw new Error(`Could not upload ${file.name}.`);
    genUploads.current.set(item.id, data.file_id);
    return data.file_id;
  }

  /** The generation input's file fields: frames, or references (never both). */
  async function generationFiles(next: Prepared): Promise<Record<string, unknown>> {
    const { format, inputs: i } = next;
    const fields: Record<string, unknown> = {};
    if (format.kind === "fl2va") {
      if (i.first) fields.first_frame_image = await uploadForGeneration(i.first);
      if (i.last) fields.last_frame_image = await uploadForGeneration(i.last);
      return fields;
    }
    if (format.kind === "t2va") return fields;
    const audios = i.audios.slice(0, format.audios);
    if (i.images.length) fields.reference_images = await Promise.all(i.images.map(uploadForGeneration));
    if (audios.length) fields.reference_audios = await Promise.all(audios.map(uploadForGeneration));
    if (i.videos.length) fields.reference_videos = await Promise.all(i.videos.map(uploadForGeneration));
    if (i.images.length + audios.length + i.videos.length) {
      fields.reference_order = [...i.images.map(() => "image"), ...audios.map(() => "audio"), ...i.videos.map(() => "video")];
    }
    return fields;
  }

  /** Upload an input once per session. */
  async function upload(m: Model, item: Item): Promise<FileRef> {
    const cached = uploads.current.get(item.id);
    if (cached) return cached;
    const ref = await m.uploadFile(item.file, { name: item.file.name });
    uploads.current.set(item.id, ref);
    return ref;
  }

  /**
   * The prompt format the next clip's inputs imply: frames (FL2VA, a fresh
   * start), references and/or a latent continuation (Ref2VA), or text only
   * (T2VA, the opening clip with nothing else).
   */
  function clipFormat(i: Inputs, withFrames: boolean, from: string | null): ClipFormat {
    if (withFrames) return { kind: "fl2va", frames: frameSet(i)! };
    if (!from && !i.images.length && !i.videos.length) return { kind: "t2va" };
    // Reference audio needs a picture, a video or a continuation beside it.
    const audios = i.images.length || i.videos.length || from ? i.audios.length : 0;
    return { kind: "ref2va", pictures: i.images.length, audios, videos: i.videos.length, tail: from ? i.videos.length + 1 : null };
  }

  /** The enqueue fields for a prepared clip, uploading its inputs as needed. */
  async function clipParams(m: Model, next: Prepared): Promise<Record<string, unknown>> {
    const { format, inputs: i, from } = next;
    const params: Record<string, unknown> = {};
    if (format.kind === "fl2va") {
      if (i.first) params.first_frame_image = await upload(m, i.first);
      if (i.last) params.last_frame_image = await upload(m, i.last);
      return params;
    }
    if (format.kind === "t2va") return params;
    const audios = i.audios.slice(0, format.audios);
    if (i.images.length) params.reference_images = await Promise.all(i.images.map((x) => upload(m, x)));
    if (audios.length) params.reference_audios = await Promise.all(audios.map((x) => upload(m, x)));
    if (i.videos.length) params.reference_videos = await Promise.all(i.videos.map((x) => upload(m, x)));
    // Audio before video keeps Audio N stable whether or not a video has sound.
    if (i.images.length + audios.length + i.videos.length) {
      params.reference_order = [...i.images.map(() => "image"), ...audios.map(() => "audio"), ...i.videos.map(() => "video")];
    }
    if (from) {
      params.continue_from_clip_id = from;
      params.continue_mode = "latent";
    }
    return params;
  }

  /** The next beat to film, moving on to the next episode when this one is done. */
  function takeBeat(): Plan | null {
    let s = storyRef.current;
    if (!s) return null;
    if (beatIndex.current >= s.beats.length) {
      if (!nextStory.current) {
        say("next episode not ready; improvising from the last beat");
        prefetchNext();
        return null;
      }
      s = nextStory.current;
      nextStory.current = null;
      showStory(s, episodeRef.current + 1, 0);
      say(`episode ${episodeRef.current}: ${s.title}`);
    }
    const plan = { story: s, episode: episodeRef.current, index: beatIndex.current };
    beatIndex.current += 1;
    setNextIndex(beatIndex.current);
    if (s.beats.length - beatIndex.current <= PREFETCH_AT) prefetchNext();
    return plan;
  }

  /** Have the showrunner write the next episode in the background. */
  function prefetchNext() {
    const s = storyRef.current;
    if (!s || fetchingNext.current || nextStory.current) return;
    fetchingNext.current = true;
    setWriting((w) => w ?? "next");
    void writeStory({ mode: "next", story: s }).then((next) => {
      fetchingNext.current = false;
      setWriting((w) => (w === "next" ? null : w));
      if (next && storyRef.current === s) {
        nextStory.current = next;
        say(`next episode ready: ${next.title}`);
      }
    });
  }

  /** The reference pictures (small JPEGs) and labels, for the story writer. */
  async function storyReferences(): Promise<{ images: string[]; references: string }> {
    const current = latest.current.inputs;
    const images = await Promise.all(
      current.images.map(async (item) => {
        const cached = thumbs.current.get(item.id);
        if (cached) return cached;
        const data = await thumbnail(item.file);
        thumbs.current.set(item.id, data);
        return data;
      }),
    );
    return { images, references: referenceList(current) };
  }

  async function writeStory(body: Record<string, unknown>): Promise<Story | null> {
    const started = performance.now();
    try {
      const refs = await storyReferences();
      const res = await fetch("/api/story", {
        method: "POST",
        headers: { "Content-Type": "application/json", ...keyHeaders() },
        body: JSON.stringify({ ...body, ...refs }),
      });
      const data = (await res.json()) as { story?: Story; error?: string };
      if (!res.ok || !data.story) {
        say(`story: ${data.error ?? res.status}`);
        return null;
      }
      say(`story (${((performance.now() - started) / 1000).toFixed(1)}s): ${data.story.title}, ${data.story.beats.length} beats`);
      return data.story;
    } catch {
      say("story: request failed");
      return null;
    }
  }

  // Ask the server-side director for the next prompt. Any failure falls back to
  // the steer text so the stream keeps going; a missing key turns Claude off.
  async function direct(steer: string, plan: Plan | null, format: ClipFormat, retake?: DirectorRequest["retake"]): Promise<string> {
    const started = performance.now();
    const body: DirectorRequest = {
      direction: steer,
      recent: recent.current,
      fresh: steer.trim() !== applied.current,
      format,
      independent: latest.current.api === "generations",
      retake,
    };
    if (plan) {
      const { story: s, index } = plan;
      const beat = s.beats[index];
      body.story = {
        title: s.title,
        bible: s.bible,
        beat,
        number: index + 1,
        total: s.beats.length,
        next: s.beats[index + 1]?.beat,
        cast: s.cast,
        props: s.props,
        place: s.places.find((p) => p.name === beat.place),
      };
    }
    try {
      const res = await fetch("/api/director", {
        method: "POST",
        headers: { "Content-Type": "application/json", ...keyHeaders() },
        body: JSON.stringify(body),
      });
      const data = (await res.json()) as { prompt?: string; error?: string };
      if (!res.ok || !data.prompt) {
        say(`director: ${data.error ?? res.status}; using the steer text`);
        if (res.status === 503) setEnhance(false);
        return steer.trim();
      }
      const secs = ((performance.now() - started) / 1000).toFixed(1);
      say(`director (${secs}s, ${format.kind}): ${data.prompt.slice(0, 110).replace(/\n/g, " ")}${data.prompt.length > 110 ? "…" : ""}`);
      return data.prompt;
    } catch {
      say("director: request failed; using the steer text");
      return steer.trim();
    }
  }

  // Video and audio arrive as separate tracks.
  useEffect(() => {
    const el = audio.current;
    if (!el) return;
    el.srcObject = audioTrack ? new MediaStream([audioTrack]) : null;
    if (audioTrack) {
      void el.play().then(() => setSoundBlocked(false)).catch(() => {
        setSoundBlocked(true);
        say("audio blocked by the browser; press unmute");
      });
    }
  }, [audioTrack]);

  /** Frames set in the panel go on the next clip queued, then leave the panel. */
  function armFrames() {
    if (!inputs.first && !inputs.last) return;
    armed.current = { first: inputs.first, last: inputs.last };
    setInputs((i) => ({ ...i, first: null, last: null }));
    say(`frames armed for the next clip: ${frameSet(armed.current)}`);
  }

  async function start() {
    const text = draft.trim();
    if (!text) return;
    setError(null);
    setStarting(true);
    setPrompt(text);
    setAired(0);
    setBuildTimes([]);
    setStalls(0);
    showStory(null, 1, 0);
    nextStory.current = null;
    try {
      // Write the first episode while the session connects.
      if (enhance) setWriting("first");
      const [first] = await Promise.all([
        enhance ? writeStory({ mode: "new", premise: text }) : Promise.resolve(null),
        api === "streaming" && model.status !== "ready" ? model.connect() : Promise.resolve(),
      ]);
      setWriting(null);
      if (enhance && !first) setError("The story writer failed, so clips follow your text directly.");
      showStory(first, 1, 0);
      armFrames();
      setLive(true);
      setLiveSince(Date.now());
      setNow(Date.now());
    } catch {
      // The reason is already on model.lastError, which shows above the controls.
      setWriting(null);
    } finally {
      setStarting(false);
    }
  }

  async function stop() {
    setLive(false);
    setLiveSince(null);
    if (api === "generations") {
      // Generations already started finish on Reactor's side; the page stops waiting for them.
      genRun.current += 1;
      genVideo.current?.pause();
      for (const clip of genClips.current.values()) if (clip.url) URL.revokeObjectURL(clip.url);
      genClips.current.clear();
      genOrder.current = [];
      genPlaying.current = null;
      genPrevFrame.current = undefined;
      recent.current = [];
      applied.current = null;
      prepared.current = null;
      armed.current = null;
      nextStory.current = null;
      built.current.clear();
      beatOf.current.clear();
      setAhead([]);
      setPlaying(null);
      setReady([]);
      setBuilding([]);
      say("stopped");
      return;
    }
    await model.disconnect();
    say("disconnected");
  }

  async function steer() {
    const text = draft.trim();
    if (!text) return;
    setPrompt(text);
    armFrames();
    say(`steer: ${text.slice(0, 100)}`);
    const s = storyRef.current;
    if (!enhance || !s || text === prompt.trim()) return;
    // Rewrite the rest of the episode around the direction; beats already filmed stay.
    setWriting("rewrite");
    const done = beatIndex.current;
    const rewritten = await writeStory({ mode: "steer", story: s, done, direction: text });
    setWriting(null);
    if (!rewritten) {
      setError("The story writer could not take that direction; try again.");
      return;
    }
    if (storyRef.current !== s) return; // the episode changed meanwhile
    nextStory.current = null;
    showStory(rewritten, episodeRef.current, beatIndex.current);
  }

  /** End the stream if it is live and start over: no story, no inputs, no log, the first example's text. */
  async function reset() {
    if (live || model.status !== "disconnected") await stop();
    for (const item of [...inputs.images, ...inputs.audios, ...inputs.videos]) URL.revokeObjectURL(item.url);
    setInputs(EMPTY_INPUTS);
    setDraft(DEFAULT_PROMPT);
    setPrompt(DEFAULT_PROMPT);
    setExampleId(EXAMPLES[0].id);
    setError(null);
    setLog([]);
    setAired(0);
    setBuildTimes([]);
    setStalls(0);
    setWriting(null);
    showStory(null, 1, 0);
    nextStory.current = null;
    uploads.current.clear();
    genUploads.current.clear();
    thumbs.current.clear();
    queuedAt.current.clear();
  }

  function changeKeys(k: Keys) {
    if (k.reactor !== keys.reactor) token = null;
    saveKeys(k);
    setKeys(k);
  }

  async function pickExample(example: Example) {
    setError(null);
    setLoadingExample(true);
    try {
      const next = await exampleInputs(example);
      for (const item of [...inputs.images, ...inputs.audios, ...inputs.videos]) URL.revokeObjectURL(item.url);
      setInputs(next);
      setDraft(example.concept);
      setExampleId(example.id);
      setTab("references");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load the example.");
    } finally {
      setLoadingExample(false);
    }
  }

  function submit(e?: React.FormEvent) {
    e?.preventDefault();
    if (live) void steer();
    else void start();
  }

  function unmute() {
    if (api === "generations") {
      const v = genVideo.current;
      if (v) v.muted = false;
      setSoundBlocked(false);
      return;
    }
    const el = audio.current;
    if (!el) return;
    void el.play().then(() => setSoundBlocked(false)).catch(() => setSoundBlocked(true));
  }

  const avgBuild = buildTimes.length ? buildTimes.reduce((a, b) => a + b, 0) / buildTimes.length : null;
  const elapsed = liveSince === null ? 0 : Math.max(0, Math.floor((now - liveSince) / 1000));
  const timer = `${String(Math.floor(elapsed / 60)).padStart(2, "0")}:${String(elapsed % 60).padStart(2, "0")}`;
  const busy = starting || (api === "streaming" && model.status === "connecting");
  const framesSet = Boolean(inputs.first || inputs.last);
  const pending = draft.trim() !== prompt.trim() || framesSet;
  const playingBeat = playing ? beatOf.current.get(playing.id) : undefined;
  const segmentOf = (id: string) => beatOf.current.get(id)?.segment;

  return (
    <main className="page">
      <div className="wrap">
        <header className="header">
          <div className="brand">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="/reactor-wordmark-white.svg" alt="Reactor" />
            <span className="divider" aria-hidden="true" />
            <h1>Pika Endless Livestream</h1>
          </div>

          <div className="controls">
            <div role="radiogroup" aria-label="Reactor API" className="tabs">
              {(["streaming", "generations"] as const).map((a) => (
                <button
                  key={a}
                  type="button"
                  role="radio"
                  aria-checked={api === a}
                  aria-selected={api === a}
                  className="tab mono"
                  disabled={live || busy}
                  title={a === "streaming" ? "One realtime session; clips continue each other" : "Each clip is its own generation; the page plays them back to back"}
                  onClick={() => setApi(a)}
                >
                  {a === "streaming" ? "Streaming API" : "Generations API"}
                </button>
              ))}
            </div>
            <dl className="stats">
              <div>
                <dt className="mono">Clips aired</dt>
                <dd>{aired}</dd>
              </div>
              <div>
                <dt className="mono">Avg build</dt>
                <dd>{avgBuild === null ? "—" : `${avgBuild.toFixed(1)} s`}</dd>
              </div>
              <div>
                <dt className="mono">Stalls</dt>
                <dd>{stalls}</dd>
              </div>
            </dl>
            <span className="pill mono" role="status">
              <span className={`dot${live ? " on" : ""}`} />
              {live ? `Live ${timer}` : busy ? "Starting" : "Offline"}
            </span>
            {live && (
              <button type="button" className="btn btn-ghost" onClick={stop}>End stream</button>
            )}
            <button type="button" className="btn btn-ghost" onClick={() => void reset()} disabled={busy} title="End the stream and start over with a clean slate">
              Reset
            </button>
          </div>
        </header>

        <div className="body">
          <section className="stream" aria-label="Stream">
            <div className="stage">
              {api === "streaming" ? (
                <>
                  <PikaH3FlexMainVideoView style={{ width: "100%", height: "100%" }} muted />
                  <audio ref={audio} autoPlay />
                </>
              ) : (
                <video ref={genVideo} className="gen-video" playsInline onEnded={onGenerationEnded} />
              )}
              {!playing && (
                <div className="stage-empty">
                  {writing === "first" ? (
                    <span className="mono">Writing the first episode…</span>
                  ) : busy || live ? (
                    <span className="mono">Starting the stream…</span>
                  ) : hasReferences(inputs) ? (
                    <div className="stage-ready">
                      <div className="stage-thumbs">
                        {inputs.images.slice(0, 5).map((i) => (
                          // eslint-disable-next-line @next/next/no-img-element
                          <img key={i.id} src={i.url} alt="" />
                        ))}
                      </div>
                      <strong>Ready to go live</strong>
                    </div>
                  ) : (
                    <div className="stage-ready">
                      <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><rect x="2" y="5" width="15" height="14" rx="2" /><path d="m17 10 5-3v10l-5-3" /></svg>
                      <strong>Nothing on air yet</strong>
                      <span>Describe what to stream below, then start the live stream.</span>
                      <span className="api-note">{API_NOTES[api]}</span>
                    </div>
                  )}
                </div>
              )}
              {playing && soundBlocked && (
                <button type="button" className="unmute" aria-label="Unmute" onClick={unmute}>
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M11 5 6 9H3v6h3l5 4V5Z" />
                    <path d="M15.5 8.5a5 5 0 0 1 0 7" />
                    <path d="M18.5 5.5a9 9 0 0 1 0 13" />
                  </svg>
                </button>
              )}
            </div>

            <form className="panel" onSubmit={submit}>
              <KeysBar
                keys={keys}
                server={serverKeys}
                onChange={changeKeys}
              />
              {!live && (
                <div className="examples" role="group" aria-label="Examples">
                  <span className="mono">Examples</span>
                  <div className="example-row">
                    {EXAMPLES.map((ex) => (
                      <button
                        key={ex.id}
                        type="button"
                        className="example"
                        aria-pressed={exampleId === ex.id}
                        disabled={loadingExample || busy}
                        onClick={() => void pickExample(ex)}
                      >
                        {ex.thumbnail ? (
                          // eslint-disable-next-line @next/next/no-img-element
                          <img src={ex.thumbnail} alt="" />
                        ) : (
                          <span className="example-blank" aria-hidden="true" />
                        )}
                        <span>{ex.name}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
              <textarea
                aria-label={live ? "Steer the stream" : "Describe the stream"}
                placeholder="A dog news show, a heist in a bakery, a nature documentary about one pigeon…"
                rows={2}
                value={draft}
                onChange={(e) => {
                  setDraft(e.target.value);
                  setExampleId(null);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit();
                }}
              />
              <InputsPanel inputs={inputs} onChange={setInputs} tab={tab} onTab={setTab} onError={setError} />
              {error && <p className="error" role="alert">{error}</p>}
              <div className="steer-row">
                <div className="writer">
                  <label className="toggle">
                    <input type="checkbox" checked={enhance} onChange={(e) => setEnhance(e.target.checked)} disabled={live} />
                    Enhance prompt with
                  </label>
                  <div role="radiogroup" aria-label="Prompt writer" className="tabs">
                    {(["claude", "openai"] as const).map((w) => (
                      <button
                        key={w}
                        type="button"
                        role="radio"
                        aria-checked={keys.writer === w}
                        aria-selected={keys.writer === w}
                        className="tab mono"
                        disabled={!enhance}
                        onClick={() => changeKeys({ ...keys, writer: w })}
                      >
                        {WRITERS[w].name}
                      </button>
                    ))}
                  </div>
                  <span className="hint">{enhance ? writerHint(keys, serverKeys) : "Off: your text goes to the model as written."}</span>
                </div>
                {live ? (
                  <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
                    {framesSet && <span className="hint">The next clip {inputs.first ? "starts" : "ends"} on this image.</span>}
                    <button type="submit" className="btn btn-primary" disabled={!pending || writing === "rewrite"}>
                      {writing === "rewrite" ? "Rewriting…" : "Steer"}
                    </button>
                  </div>
                ) : (
                  <button type="submit" className="btn btn-primary" disabled={busy || !draft.trim()}>
                    {writing === "first" ? "Writing…" : busy ? "Starting…" : "Start live stream"}
                  </button>
                )}
              </div>
            </form>
          </section>

          <aside className="queue" aria-label="Story and prompts">
            <div role="tablist" aria-label="Sidebar" className="tabs side-tabs">
              <button type="button" role="tab" aria-selected={side === "story"} className="tab mono" onClick={() => setSide("story")}>Story</button>
              <button type="button" role="tab" aria-selected={side === "prompts"} className="tab mono" onClick={() => setSide("prompts")}>Prompts</button>
            </div>

            {side === "story" ? (
              story ? (
                <section className="story" aria-label="Story">
                  <span className="mono">Episode {episode}{writing === "rewrite" ? " · rewriting" : writing === "next" ? " · writing the next" : ""}</span>
                  <h2>{story.title}</h2>
                  <ol className="beats">
                    {story.beats.map((b, i) => {
                      const onAir = playingBeat?.episode === episode && playingBeat.index === i;
                      const state = onAir ? "on-air" : playingBeat?.episode === episode && i < playingBeat.index ? "done" : i < nextIndex ? "queued" : "upcoming";
                      return (
                        <li key={`${episode}-${i}`} className={`beat ${state}`} aria-current={onAir ? "step" : undefined}>
                          <span className="mono">{onAir ? "On air · " : ""}{b.segment}{b.cut ? " · cut" : ""}</span>
                          <span>{b.beat}</span>
                        </li>
                      );
                    })}
                  </ol>
                </section>
              ) : (
                <SideEmpty text={writing === "first" ? "Writing the first episode…" : enhance ? "The episode's story arc shows up here once you start." : "Turn on Enhance prompt to have Claude plan episodes."} />
              )
            ) : playing || ready.length || building.length ? (
              <div className="prompts">
                {playing && (
                  <article className="card playing" aria-current="step">
                    <div className="card-head mono"><span>Playing{playingBeat ? ` · ${playingBeat.segment}` : ""}</span><span className="meta">{playing.seconds.toFixed(1)} s</span></div>
                    <pre className="raw">{playing.prompt}</pre>
                  </article>
                )}
                {ready.map((c) => (
                  <article key={c.clip_id} className="card ready">
                    <div className="card-head mono"><span className="state">Ready{segmentOf(c.clip_id) ? ` · ${segmentOf(c.clip_id)}` : ""}</span><span className="meta">{c.seconds.toFixed(1)} s</span></div>
                    <pre className="raw">{c.prompt}</pre>
                  </article>
                ))}
                {building.map((c) => (
                  <article key={c.clip_id} className="card building">
                    <div className="card-head mono"><span>Building{segmentOf(c.clip_id) ? ` · ${segmentOf(c.clip_id)}` : ""}</span><span className="meta">{c.seconds.toFixed(1)} s</span></div>
                    <pre className="raw">{c.prompt}</pre>
                  </article>
                ))}
              </div>
            ) : (
              <SideEmpty text="Each clip's prompt shows up here as it is written, built and played." />
            )}
          </aside>
        </div>

        <details className="debug">
          <summary className="mono">Debug log</summary>
          <pre>{log.join("\n") || "Nothing yet."}</pre>
        </details>
      </div>
    </main>
  );
}

function SideEmpty({ text }: { text: string }) {
  return (
    <>
      <div className="ghost-cards" aria-hidden="true"><div /><div /><div /></div>
      <p className="empty">{text}</p>
    </>
  );
}

// Who writes the stream when Enhance prompt is on. Matches app/lib/server/writer.ts.
const WRITERS: Record<Writer, { name: string; story: string; clip: string; key: "anthropic" | "openai" }> = {
  claude: { name: "Claude", story: "Claude Sonnet 5.5", clip: "Claude Haiku 5.5", key: "anthropic" },
  openai: { name: "OpenAI", story: "GPT-5.1", clip: "GPT-5.1 mini", key: "openai" },
};

function writerHint(keys: Keys, server: ServerKeys | null): string {
  const w = WRITERS[keys.writer];
  const has = (k: "anthropic" | "openai") => Boolean(keys[k] || server?.[k]);
  if (!has(w.key)) {
    const other = keys.writer === "claude" ? WRITERS.openai : WRITERS.claude;
    return has(other.key)
      ? `No ${w.name} key, so ${other.name} writes instead. Paste a ${w.name} key above.`
      : `No ${w.name} key: paste one above, or your text goes to the model as written.`;
  }
  return `${w.story} plans each episode; ${w.clip} writes every clip's prompt.`;
}

// What the header's API switch changes, shown on the empty stage.
const API_NOTES: Record<Api, string> = {
  streaming:
    "Streaming API: one realtime session. The model plays each clip as soon as it is built, and every clip continues the one before, as one continuous take.",
  generations:
    "Generations API: each clip is its own generation job. The page plays the finished videos in order, joined by cuts, and with Enhance prompt on it checks each clip's last frame and retakes a broken one before it plays.",
};
