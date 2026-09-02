"use client";

import {
  RiAddLine,
  RiArrowLeftSLine,
  RiArrowRightLine,
  RiArrowRightSLine,
  RiCheckLine,
  RiCloseLine,
  RiDeleteBinLine,
  RiErrorWarningLine,
  RiFullscreenLine,
  RiImageAddLine,
  RiLoader4Line,
  RiMicLine,
  RiPauseFill,
  RiPlayFill,
  RiSendPlane2Line,
  RiStopCircleLine,
  RiStopFill,
  RiVolumeMuteLine,
  RiVolumeUpLine,
} from "@remixicon/react";
import { FileRef, Reactor, type ReactorMessage, type ReactorStatus } from "@reactor-team/js-sdk";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type CSSProperties,
  type DragEvent,
  type FormEvent,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { CEREBRAS_STORY_MODEL, type StoryHistoryItem, type StoryPlan } from "@/lib/cerebras-contract";
import { COOKING_PROPS, type CookingProp } from "@/lib/cooking-props";
import {
  FAST_H3_CANVAS,
  FAST_H3_DEFAULT_CLIP_SECONDS,
  FAST_H3_DISPLAY_NAME,
  FAST_H3_MAX_PROMPT_LENGTH,
  FAST_H3_MODEL,
  clipFromMessageData,
  queueFromMessageData,
  type FastH3ClipInfo,
  type FastH3ClipSeconds,
  type FastH3Queue,
} from "@/lib/h3-contract";

const DEFAULT_PROMPT = "An eccentric chef hosts a continuous live cooking show. Preserve the established character, kitchen, lighting, camera language, and native sound while the action advances naturally.";
const IMAGE_DB_NAME = "h3-max-studio";
const IMAGE_STORE_NAME = "images";
const START_IMAGE_STORAGE_KEY = "h3-max:start-image";
const START_PROMPT_STORAGE_KEY = "h3-max:start-prompt";
const IMAGE_PROMPTS_STORAGE_KEY = "h3-max:image-prompts";
const AUDIO_STORAGE_KEY = "h3-max:audio-enabled";
const CHUNK_SECONDS_STORAGE_KEY = "h3-max:chunk-seconds";
const PREBUFFER_CHUNKS_STORAGE_KEY = "h3-max:prebuffer-chunks";
const DIRECTOR_COLLAPSED_STORAGE_KEY = "h3-max:director-collapsed";
const PROPS_PANE_HEIGHT_STORAGE_KEY = "h3-max:props-pane-height";
const GALTON_LEGACY_IMAGE_NAME = "exec-624d143b-1ecd-4469-b291-93b5a219c258.png";
const GALTON_MOUSTACHE_IMAGE_ID = "galton-ramshackle-moustache-v1";
const GALTON_MOUSTACHE_IMAGE_NAME = "galton-ramshackle-moustache.png";
const GALTON_MOUSTACHE_IMAGE_URL = "/characters/galton-ramshackle-moustache.png";
const GALTON_MOUSTACHE_PROMPT = "Galton Ramshackle, an eccentric hot-tempered British chef with a very visible curly moustache, hosts a chaotic live cooking show in his professional kitchen. He introduces himself, cooks with theatrical intensity, delivers funny sharp dialogue, and reacts intuitively to new ingredients. No visible camera or film crew. Preserve Galton's identity, moustache, kitchen, lighting, camera language, and native sound.";
const MAX_LIBRARY_IMAGES = 30;
const MAX_VISIBLE_CLIPS = 20;
const DEFAULT_PREBUFFER_CHUNKS = 3;
const MIN_IMAGE_LIBRARY_HEIGHT = 96;
const MIN_PROPS_PANE_HEIGHT = 160;
const IMAGE_TYPES = new Set(["image/jpeg", "image/png", "image/webp", "image/gif", "image/avif"]);

type StreamPhase = "idle" | "connecting" | "generating" | "playing" | "buffering" | "paused" | "stopped" | "error";
type PrebufferChunks = 0 | 1 | 2 | 3 | 4;
type ClipStatus = "queued" | "ready" | "playing" | "finished" | "failed";

type StudioSpeechRecognitionResult = { isFinal: boolean; 0: { transcript: string } };
type StudioSpeechRecognition = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((event: { results: ArrayLike<StudioSpeechRecognitionResult> }) => void) | null;
  onerror: ((event: { error: string; message?: string }) => void) | null;
  onend: (() => void) | null;
  onspeechend: (() => void) | null;
  start: () => void;
  stop: () => void;
  abort: () => void;
};
type StudioSpeechRecognitionConstructor = new () => StudioSpeechRecognition;

declare global {
  interface Window {
    SpeechRecognition?: StudioSpeechRecognitionConstructor;
    webkitSpeechRecognition?: StudioSpeechRecognitionConstructor;
  }
}

type LibraryImage = { id: string; name: string; dataUrl: string; createdAt: number; isDemo?: boolean };
const GALTON_DEMO_IMAGE: LibraryImage = {
  id: GALTON_MOUSTACHE_IMAGE_ID,
  name: GALTON_MOUSTACHE_IMAGE_NAME,
  dataUrl: GALTON_MOUSTACHE_IMAGE_URL,
  createdAt: 0,
  isDemo: true,
};
type StreamClip = FastH3ClipInfo & {
  ordinal: number;
  status: ClipStatus;
  sceneSummary: string;
  dialogue: string;
  storyModel: string | null;
};
type TextCue = { id: string; label: string; instruction: string };
type ActivityItem = { id: string; text: string; tone: "normal" | "good" | "bad" };

function openImageDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(IMAGE_DB_NAME, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(IMAGE_STORE_NAME)) db.createObjectStore(IMAGE_STORE_NAME, { keyPath: "id" });
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("Could not open the image library."));
  });
}

async function readStoredImages() {
  const db = await openImageDb();
  return new Promise<LibraryImage[]>((resolve, reject) => {
    const request = db.transaction(IMAGE_STORE_NAME, "readonly").objectStore(IMAGE_STORE_NAME).getAll();
    request.onsuccess = () => resolve((request.result as LibraryImage[]).sort((a, b) => b.createdAt - a.createdAt));
    request.onerror = () => reject(request.error ?? new Error("Could not read the image library."));
  }).finally(() => db.close());
}

async function storeImage(image: LibraryImage) {
  const db = await openImageDb();
  await new Promise<void>((resolve, reject) => {
    const request = db.transaction(IMAGE_STORE_NAME, "readwrite").objectStore(IMAGE_STORE_NAME).put(image);
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error ?? new Error("Could not store the image."));
  });
  db.close();
}

async function removeStoredImage(id: string) {
  const db = await openImageDb();
  await new Promise<void>((resolve, reject) => {
    const request = db.transaction(IMAGE_STORE_NAME, "readwrite").objectStore(IMAGE_STORE_NAME).delete(id);
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error ?? new Error("Could not remove the image."));
  });
  db.close();
}

function fileToDataUrl(file: File) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => typeof reader.result === "string" ? resolve(reader.result) : reject(new Error("Could not read the image."));
    reader.onerror = () => reject(reader.error ?? new Error("Could not read the image."));
    reader.readAsDataURL(file);
  });
}

async function dataUrlToFile(dataUrl: string, name: string) {
  const response = await fetch(dataUrl);
  const blob = await response.blob();
  return new File([blob], name, { type: blob.type || "image/png" });
}

function readImagePrompts(): Record<string, string> {
  try {
    const stored = window.localStorage.getItem(IMAGE_PROMPTS_STORAGE_KEY);
    return stored ? JSON.parse(stored) as Record<string, string> : {};
  } catch {
    return {};
  }
}

function cleanImageLabel(name: string) {
  return name
    .replace(/\.[a-z0-9]{2,5}$/i, "")
    .replace(/^(img|image|photo|screenshot|cleanshot)[-_ ]*/i, "")
    .replace(/[-_]+/g, " ")
    .replace(/\b[0-9a-f]{8,}\b/gi, "")
    .replace(/\s+/g, " ")
    .trim() || "the subject from the added image";
}

function imageCue(image: LibraryImage): TextCue {
  const savedPrompt = readImagePrompts()[image.id]?.trim();
  const label = cleanImageLabel(image.name);
  return {
    id: `image-${image.id}`,
    label,
    instruction: savedPrompt
      ? `Incorporate ${label} into the current scene naturally. Use this textual description: ${savedPrompt}`
      : `Incorporate ${label} into the current scene naturally and make it part of the ongoing action.`,
  };
}

function propDirection(props: CookingProp[], cues: TextCue[]) {
  const parts: string[] = [];
  if (props.length) {
    parts.push(`Persistent scene props: ${props.map((prop) => prop.name).join(", ")}.`);
    parts.push(...props.map((prop) => `${prop.name}: ${prop.instruction}.`));
    parts.push("Keep every named prop visibly or logically present in the established scene until the operator removes it.");
  }
  if (cues.length) parts.push(...cues.map((cue) => cue.instruction));
  if (parts.length) parts.push("Preserve the current character, location, camera language, dialogue continuity, and native sound.");
  return parts.join(" ");
}

function fitPrompt(value: string) {
  const clean = value.replace(/\s+/g, " ").trim();
  return clean.length <= FAST_H3_MAX_PROMPT_LENGTH ? clean : `${clean.slice(0, FAST_H3_MAX_PROMPT_LENGTH - 1).trimEnd()}…`;
}

function formatElapsed(milliseconds: number) {
  const totalSeconds = Math.floor(milliseconds / 1_000);
  return `${String(Math.floor(totalSeconds / 60)).padStart(2, "0")}:${String(totalSeconds % 60).padStart(2, "0")}`;
}

function phaseLabel(phase: StreamPhase) {
  if (phase === "connecting") return "Connecting";
  if (phase === "generating") return "Building opening";
  if (phase === "buffering") return "Building next clip";
  if (phase === "playing") return "Live";
  if (phase === "paused") return "Playback paused";
  if (phase === "stopped") return "Stream stopped";
  if (phase === "error") return "Needs attention";
  return "Ready";
}

function recordFrom(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? value as Record<string, unknown> : {};
}

export function H3Studio() {
  const [images, setImages] = useState<LibraryImage[]>([]);
  const [startImageId, setStartImageId] = useState<string | null>(null);
  const [startPrompt, setStartPrompt] = useState(DEFAULT_PROMPT);
  const [nextPrompt, setNextPrompt] = useState("");
  const [chunkSeconds, setChunkSeconds] = useState<FastH3ClipSeconds>(FAST_H3_DEFAULT_CLIP_SECONDS);
  const [prebufferChunks, setPrebufferChunks] = useState<PrebufferChunks>(DEFAULT_PREBUFFER_CHUNKS);
  const [hasCerebrasKey, setHasCerebrasKey] = useState(false);
  const [phase, setPhase] = useState<StreamPhase>("idle");
  const [reactorStatus, setReactorStatus] = useState<ReactorStatus>("disconnected");
  const [clips, setClips] = useState<StreamClip[]>([]);
  const [queue, setQueue] = useState<FastH3Queue>({ generation: [], playout: [], history: [] });
  const [currentClipId, setCurrentClipId] = useState<string | null>(null);
  const [hasMedia, setHasMedia] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [activity, setActivity] = useState<ActivityItem[]>([]);
  const [isPageDragging, setIsPageDragging] = useState(false);
  const [isStageDragging, setIsStageDragging] = useState(false);
  const [isPropDragging, setIsPropDragging] = useState(false);
  const [queuedProps, setQueuedProps] = useState<CookingProp[]>([]);
  const [textCues, setTextCues] = useState<TextCue[]>([]);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [audioEnabled, setAudioEnabled] = useState(true);
  const [isVoiceSupported, setIsVoiceSupported] = useState(false);
  const [isVoiceListening, setIsVoiceListening] = useState(false);
  const [isDirectorCollapsed, setIsDirectorCollapsed] = useState(false);
  const [propsPaneHeight, setPropsPaneHeight] = useState<number | null>(null);
  const [isPropsResizing, setIsPropsResizing] = useState(false);
  const [storageReady, setStorageReady] = useState(false);

  const stageRef = useRef<HTMLDivElement | null>(null);
  const imagePanelRef = useRef<HTMLElement | null>(null);
  const imageLibraryRef = useRef<HTMLDivElement | null>(null);
  const propsLibraryRef = useRef<HTMLDivElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const reactorRef = useRef<Reactor | null>(null);
  const runActiveRef = useRef(false);
  const phaseRef = useRef<StreamPhase>("idle");
  const promptRef = useRef(startPrompt);
  const chunkSecondsRef = useRef<FastH3ClipSeconds>(chunkSeconds);
  const prebufferChunksRef = useRef<PrebufferChunks>(prebufferChunks);
  const audioEnabledRef = useRef(audioEnabled);
  const queuedPropsRef = useRef<CookingProp[]>([]);
  const textCuesRef = useRef<TextCue[]>([]);
  const queueRef = useRef<FastH3Queue>({ generation: [], playout: [], history: [] });
  const clipsRef = useRef<StreamClip[]>([]);
  const lastQueuedClipIdRef = useRef<string | null>(null);
  const startingFrameRef = useRef<FileRef | null>(null);
  const queueFillActiveRef = useRef(false);
  const autoplayEnabledRef = useRef(false);
  const streamEpochRef = useRef(0);
  const nextOrdinalRef = useRef(1);
  const startedAtRef = useRef<number | null>(null);
  const storyHistoryRef = useRef<StoryHistoryItem[]>([]);
  const storyFallbackNotifiedRef = useRef(false);
  const dragDepthRef = useRef(0);
  const voiceRecognitionRef = useRef<StudioSpeechRecognition | null>(null);
  const voiceShouldResumeRef = useRef(false);
  const voicePromptBaseRef = useRef("");
  const voiceCapturedSpeechRef = useRef(false);
  const propsResizeRef = useRef<{ pointerId: number; startY: number; startHeight: number; maxHeight: number; height: number } | null>(null);
  const ensureQueueRef = useRef<() => Promise<void>>(async () => undefined);

  const startImage = images.find((image) => image.id === startImageId) ?? null;
  const isRunning = phase !== "idle" && phase !== "stopped" && phase !== "error";
  const queuedCount = queue.generation.length + queue.playout.length;
  const readyCount = queue.playout.length;
  const currentClip = clips.find((clip) => clip.clip_id === currentClipId) ?? null;

  const setPhaseState = useCallback((next: StreamPhase) => {
    phaseRef.current = next;
    setPhase(next);
  }, []);

  const replaceClips = useCallback((next: StreamClip[]) => {
    clipsRef.current = next;
    setClips(next.slice(-MAX_VISIBLE_CLIPS));
  }, []);

  const upsertClip = useCallback((clip: FastH3ClipInfo, status: ClipStatus, extra?: Partial<StreamClip>) => {
    const existing = clipsRef.current.find((item) => item.clip_id === clip.clip_id);
    const nextClip: StreamClip = {
      ...clip,
      ordinal: existing?.ordinal ?? nextOrdinalRef.current++,
      status,
      sceneSummary: existing?.sceneSummary ?? extra?.sceneSummary ?? "",
      dialogue: existing?.dialogue ?? extra?.dialogue ?? "",
      storyModel: existing?.storyModel ?? extra?.storyModel ?? null,
      ...extra,
    };
    replaceClips(existing
      ? clipsRef.current.map((item) => item.clip_id === clip.clip_id ? nextClip : item)
      : [...clipsRef.current, nextClip]);
  }, [replaceClips]);

  const pushActivity = useCallback((text: string, tone: ActivityItem["tone"] = "normal") => {
    setActivity((current) => [{ id: window.crypto.randomUUID(), text, tone }, ...current].slice(0, 7));
  }, []);

  useEffect(() => {
    void readStoredImages().then(async (stored) => {
      const storedStartId = window.localStorage.getItem(START_IMAGE_STORAGE_KEY);
      const storedStart = stored.find((image) => image.id === storedStartId) ?? null;
      const prompts = readImagePrompts();
      const retained = [
        GALTON_DEMO_IMAGE,
        ...stored.filter((image) => image.id !== GALTON_MOUSTACHE_IMAGE_ID).slice(0, MAX_LIBRARY_IMAGES - 1),
      ];
      const useGalton = !storedStartId || storedStart?.name === GALTON_LEGACY_IMAGE_NAME;
      const restoredStart = useGalton
        ? GALTON_DEMO_IMAGE
        : retained.find((image) => image.id === storedStartId) ?? GALTON_DEMO_IMAGE;
      setImages(retained);
      setStartImageId(restoredStart.id);
      setStartPrompt(
        prompts[restoredStart.id]
        || (restoredStart.isDemo ? GALTON_MOUSTACHE_PROMPT : null)
        || window.localStorage.getItem(START_PROMPT_STORAGE_KEY)
        || DEFAULT_PROMPT,
      );
      const storedAudio = window.localStorage.getItem(AUDIO_STORAGE_KEY) !== "false";
      audioEnabledRef.current = storedAudio;
      setAudioEnabled(storedAudio);
      const storedSecondsValue = window.localStorage.getItem(CHUNK_SECONDS_STORAGE_KEY);
      const storedSeconds = storedSecondsValue === null ? NaN : Number(storedSecondsValue);
      if ([6, 10, 14].includes(storedSeconds)) setChunkSeconds(storedSeconds as FastH3ClipSeconds);
      const storedBufferValue = window.localStorage.getItem(PREBUFFER_CHUNKS_STORAGE_KEY);
      const storedBuffer = storedBufferValue === null ? NaN : Number(storedBufferValue);
      if ([0, 1, 2, 3, 4].includes(storedBuffer)) setPrebufferChunks(storedBuffer as PrebufferChunks);
      setIsDirectorCollapsed(window.localStorage.getItem(DIRECTOR_COLLAPSED_STORAGE_KEY) === "true");
      const storedHeight = Number(window.localStorage.getItem(PROPS_PANE_HEIGHT_STORAGE_KEY));
      if (Number.isFinite(storedHeight) && storedHeight >= MIN_PROPS_PANE_HEIGHT) setPropsPaneHeight(storedHeight);
      setStorageReady(true);
    }).catch(() => setNotice("The local image library is unavailable in this browser."));
  }, []);

  useEffect(() => {
    void fetch("/api/story", { cache: "no-store" })
      .then((response) => response.json())
      .then((body: { enabled?: boolean }) => setHasCerebrasKey(Boolean(body.enabled)))
      .catch(() => setHasCerebrasKey(false));
  }, []);

  useEffect(() => {
    if (!storageReady) return;
    window.localStorage.setItem(START_PROMPT_STORAGE_KEY, startPrompt);
    if (startImageId) {
      window.localStorage.setItem(START_IMAGE_STORAGE_KEY, startImageId);
      window.localStorage.setItem(IMAGE_PROMPTS_STORAGE_KEY, JSON.stringify({ ...readImagePrompts(), [startImageId]: startPrompt }));
    } else window.localStorage.removeItem(START_IMAGE_STORAGE_KEY);
  }, [startImageId, startPrompt, storageReady]);

  useEffect(() => {
    audioEnabledRef.current = audioEnabled;
    if (audioRef.current) {
      audioRef.current.muted = !audioEnabled;
      audioRef.current.volume = 1;
    }
    if (storageReady) window.localStorage.setItem(AUDIO_STORAGE_KEY, String(audioEnabled));
  }, [audioEnabled, storageReady]);

  useEffect(() => {
    if (!storageReady) return;
    window.localStorage.setItem(CHUNK_SECONDS_STORAGE_KEY, String(chunkSeconds));
    window.localStorage.setItem(PREBUFFER_CHUNKS_STORAGE_KEY, String(prebufferChunks));
    window.localStorage.setItem(DIRECTOR_COLLAPSED_STORAGE_KEY, String(isDirectorCollapsed));
  }, [chunkSeconds, isDirectorCollapsed, prebufferChunks, storageReady]);

  useEffect(() => {
    setIsVoiceSupported(Boolean(window.SpeechRecognition || window.webkitSpeechRecognition));
    const handleFullscreen = () => setIsFullscreen(document.fullscreenElement === stageRef.current);
    document.addEventListener("fullscreenchange", handleFullscreen);
    return () => document.removeEventListener("fullscreenchange", handleFullscreen);
  }, []);

  useEffect(() => {
    if (!runActiveRef.current || !startedAtRef.current) return;
    const interval = window.setInterval(() => setElapsedMs(Date.now() - startedAtRef.current!), 1_000);
    return () => window.clearInterval(interval);
  }, [phase]);

  useEffect(() => () => {
    runActiveRef.current = false;
    streamEpochRef.current += 1;
    const reactor = reactorRef.current;
    reactorRef.current = null;
    void reactor?.disconnect();
  }, []);

  const planStoryBeat = useCallback(async (direction: string, history: StoryHistoryItem[], props: CookingProp[], cues: TextCue[]): Promise<StoryPlan | null> => {
    if (!hasCerebrasKey) return null;
    try {
      const response = await fetch("/api/story", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          direction,
          duration: chunkSecondsRef.current,
          history,
          images: [],
          props: [
            ...props.map((prop) => `${prop.name}: ${prop.instruction}`),
            ...cues.map((cue) => `${cue.label}: ${cue.instruction}`),
          ],
        }),
      });
      const body = await response.json().catch(() => ({})) as Partial<StoryPlan> & { error?: string };
      if (!response.ok || !body.videoPrompt) throw new Error(body.error || "Cerebras did not return a story plan.");
      storyFallbackNotifiedRef.current = false;
      return {
        videoPrompt: body.videoPrompt,
        sceneSummary: body.sceneSummary ?? "",
        dialogue: body.dialogue ?? "",
        model: body.model ?? CEREBRAS_STORY_MODEL,
        latencyMs: typeof body.latencyMs === "number" ? body.latencyMs : null,
      };
    } catch (storyError) {
      if (!storyFallbackNotifiedRef.current) {
        setNotice(`Cerebras is unavailable; continuing directly. ${storyError instanceof Error ? storyError.message : ""}`.trim());
        pushActivity("Cerebras skipped; direct prompt fallback is active.", "bad");
        storyFallbackNotifiedRef.current = true;
      }
      return null;
    }
  }, [hasCerebrasKey, pushActivity]);

  const maybeEnableAutoplay = useCallback(async () => {
    const reactor = reactorRef.current;
    if (!reactor || !runActiveRef.current || autoplayEnabledRef.current) return;
    const threshold = prebufferChunksRef.current;
    if (threshold > 0 && queueRef.current.playout.length < threshold) return;
    autoplayEnabledRef.current = true;
    await reactor.sendCommand("set_autoplay", { enabled: true });
    pushActivity(threshold ? `${threshold}-clip prebuffer ready. Playback started.` : "Playback starts on the first ready clip.", "good");
  }, [pushActivity]);

  const ensureQueue = useCallback(async () => {
    const reactor = reactorRef.current;
    if (!reactor || !runActiveRef.current || reactor.getStatus() !== "ready" || queueFillActiveRef.current) return;
    const activeEpoch = streamEpochRef.current;
    queueFillActiveRef.current = true;
    try {
      const targetDepth = Math.max(1, prebufferChunksRef.current + 1);
      let outstanding = queueRef.current.generation.length + queueRef.current.playout.length;
      while (runActiveRef.current && activeEpoch === streamEpochRef.current && outstanding < targetDepth) {
        const props = queuedPropsRef.current.slice(-4);
        const cues = textCuesRef.current.slice(-4);
        const context = propDirection(props, cues);
        const direction = [promptRef.current.trim(), context].filter(Boolean).join(" ");
        const storyPlan = await planStoryBeat(direction, storyHistoryRef.current.slice(-6), props, cues);
        if (!runActiveRef.current || activeEpoch !== streamEpochRef.current) return;
        const prompt = fitPrompt(storyPlan?.videoPrompt
          ? [storyPlan.videoPrompt, context].filter(Boolean).join(" ")
          : direction);
        const isFirst = lastQueuedClipIdRef.current === null;
        const payload: Record<string, unknown> = {
          prompt,
          seconds: chunkSecondsRef.current,
          metadata: JSON.stringify({ ordinal: nextOrdinalRef.current }),
        };
        if (isFirst && startingFrameRef.current) payload.starting_frame = startingFrameRef.current;
        else if (lastQueuedClipIdRef.current) payload.continue_from_clip_id = lastQueuedClipIdRef.current;
        const reply = await reactor.sendCommand("enqueue", payload);
        const clip = clipFromMessageData(reply?.data);
        if (!clip) {
          const lastError = reactor.getLastError();
          throw new Error(lastError?.message || "FastH3 did not accept the next clip.");
        }
        lastQueuedClipIdRef.current = clip.clip_id;
        startingFrameRef.current = null;
        upsertClip(clip, "queued", {
          sceneSummary: storyPlan?.sceneSummary ?? "",
          dialogue: storyPlan?.dialogue ?? "",
          storyModel: storyPlan?.model ?? null,
        });
        if (storyPlan) {
          storyHistoryRef.current = [...storyHistoryRef.current, {
            sceneSummary: storyPlan.sceneSummary,
            dialogue: storyPlan.dialogue,
            videoPrompt: prompt,
          }].slice(-8);
        }
        outstanding += 1;
        setPhaseState(currentClipId ? "playing" : "generating");
        pushActivity(`Clip ${nextOrdinalRef.current - 1} queued on ${FAST_H3_DISPLAY_NAME}.`);
      }
    } catch (queueError) {
      if (!runActiveRef.current || activeEpoch !== streamEpochRef.current) return;
      const message = queueError instanceof Error ? queueError.message : "The Reactor queue failed.";
      setError(message);
      setPhaseState("error");
      runActiveRef.current = false;
      pushActivity(message, "bad");
    } finally {
      queueFillActiveRef.current = false;
    }
  }, [currentClipId, planStoryBeat, pushActivity, setPhaseState, upsertClip]);

  useEffect(() => { ensureQueueRef.current = ensureQueue; }, [ensureQueue]);

  const handleReactorMessage = useCallback((message: ReactorMessage) => {
    const clip = clipFromMessageData(message.data);
    if (message.type === "queue_update") {
      const nextQueue = queueFromMessageData(message.data);
      if (nextQueue) {
        queueRef.current = nextQueue;
        setQueue(nextQueue);
        for (const item of nextQueue.generation) upsertClip(item, "queued");
        for (const item of nextQueue.playout) upsertClip(item, "ready");
        void maybeEnableAutoplay();
        window.setTimeout(() => void ensureQueueRef.current(), 0);
      }
      return;
    }
    if (message.type === "clip_generated" && clip) {
      upsertClip(clip, "ready");
      pushActivity(`Clip ${clipsRef.current.find((item) => item.clip_id === clip.clip_id)?.ordinal ?? ""} is ready.`, "good");
      void maybeEnableAutoplay();
      return;
    }
    if (message.type === "clip_started" && clip) {
      setCurrentClipId(clip.clip_id);
      setHasMedia(true);
      upsertClip(clip, "playing");
      setPhaseState("playing");
      return;
    }
    if (message.type === "clip_finished" && clip) {
      upsertClip(clip, "finished");
      setCurrentClipId((current) => current === clip.clip_id ? null : current);
      if (runActiveRef.current) setPhaseState(queueRef.current.playout.length ? "playing" : "buffering");
      window.setTimeout(() => void ensureQueueRef.current(), 0);
      return;
    }
    if (message.type === "clip_failed" && clip) {
      upsertClip(clip, "failed");
      setNotice(`Clip ${clipsRef.current.find((item) => item.clip_id === clip.clip_id)?.ordinal ?? ""} failed; the queue is continuing.`);
      pushActivity("A clip failed to build. Filling the queue again.", "bad");
      window.setTimeout(() => void ensureQueueRef.current(), 0);
      return;
    }
    if (message.type === "command_error") {
      const data = recordFrom(message.data);
      const reason = typeof data.reason === "string" ? data.reason : "Reactor rejected a command.";
      setNotice(reason);
      pushActivity(`${String(data.command ?? "Command")}: ${reason}`, "bad");
    }
  }, [maybeEnableAutoplay, pushActivity, setPhaseState, upsertClip]);

  const startStream = useCallback(async () => {
    if (!startPrompt.trim()) {
      setError("Write a starting direction before starting the stream.");
      return;
    }
    const activeEpoch = ++streamEpochRef.current;
    runActiveRef.current = false;
    const previous = reactorRef.current;
    reactorRef.current = null;
    await previous?.disconnect().catch(() => undefined);
    setError(null);
    setNotice(null);
    setActivity([]);
    setHasMedia(false);
    setCurrentClipId(null);
    setClips([]);
    clipsRef.current = [];
    setQueue({ generation: [], playout: [], history: [] });
    queueRef.current = { generation: [], playout: [], history: [] };
    nextOrdinalRef.current = 1;
    lastQueuedClipIdRef.current = null;
    startingFrameRef.current = null;
    autoplayEnabledRef.current = false;
    queueFillActiveRef.current = false;
    storyHistoryRef.current = [];
    promptRef.current = startPrompt.trim();
    chunkSecondsRef.current = chunkSeconds;
    prebufferChunksRef.current = prebufferChunks;
    startedAtRef.current = Date.now();
    setElapsedMs(0);
    setPhaseState("connecting");
    try {
      const tokenResponse = await fetch("/api/reactor/token", { method: "POST" });
      const tokenBody = await tokenResponse.json().catch(() => ({})) as { jwt?: string; error?: string };
      if (!tokenResponse.ok || !tokenBody.jwt) throw new Error(tokenBody.error || "Could not authenticate with Reactor.");
      if (activeEpoch !== streamEpochRef.current) return;
      const reactor = new Reactor({ modelName: FAST_H3_MODEL });
      reactorRef.current = reactor;
      reactor.on("statusChanged", (status) => {
        setReactorStatus(status);
        if (!runActiveRef.current && status !== "ready") return;
        if (status === "connecting" || status === "waiting") setPhaseState("connecting");
        if (status === "disconnected" && runActiveRef.current) {
          runActiveRef.current = false;
          setError("The Reactor session disconnected.");
          setPhaseState("error");
        }
      });
      reactor.on("error", (reactorError) => {
        setNotice(reactorError.message);
        pushActivity(reactorError.message, "bad");
      });
      reactor.on("message", handleReactorMessage);
      reactor.on("trackReceived", (name, _track, stream) => {
        if (name === "main_video" && videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.muted = true;
          void videoRef.current.play().catch(() => setNotice("Select play to begin the live video."));
        }
        if (name === "main_audio" && audioRef.current) {
          audioRef.current.srcObject = stream;
          audioRef.current.muted = !audioEnabledRef.current;
          audioRef.current.volume = 1;
          void audioRef.current.play().catch(() => setNotice("Select unmute to begin native audio."));
        }
      });
      await reactor.connect(tokenBody.jwt);
      if (activeEpoch !== streamEpochRef.current) {
        await reactor.disconnect();
        return;
      }
      setReactorStatus("ready");
      await reactor.sendCommand("set_canvas", { aspect: "16:9" });
      await reactor.sendCommand("set_clip_seconds", { seconds: chunkSeconds });
      await reactor.sendCommand("set_flush_on_clip_end", { enabled: false });
      if (startImage) {
        setPhaseState("connecting");
        startingFrameRef.current = await reactor.uploadFile(await dataUrlToFile(startImage.dataUrl, startImage.name));
        pushActivity(`${startImage.name} uploaded as the opening frame.`, "good");
      }
      runActiveRef.current = true;
      setPhaseState("generating");
      if (prebufferChunks === 0) await maybeEnableAutoplay();
      await ensureQueueRef.current();
    } catch (startError) {
      if (activeEpoch !== streamEpochRef.current) return;
      runActiveRef.current = false;
      const message = startError instanceof Error ? startError.message : "Could not start the Reactor session.";
      setError(message);
      setPhaseState("error");
      pushActivity(message, "bad");
      const reactor = reactorRef.current;
      reactorRef.current = null;
      await reactor?.disconnect().catch(() => undefined);
    }
  }, [chunkSeconds, handleReactorMessage, maybeEnableAutoplay, prebufferChunks, pushActivity, setPhaseState, startImage, startPrompt]);

  const stopStream = useCallback(() => {
    runActiveRef.current = false;
    streamEpochRef.current += 1;
    autoplayEnabledRef.current = false;
    voiceShouldResumeRef.current = false;
    voiceRecognitionRef.current?.abort();
    voiceRecognitionRef.current = null;
    setIsVoiceListening(false);
    const reactor = reactorRef.current;
    reactorRef.current = null;
    void (async () => {
      if (reactor) {
        await reactor.sendCommand("set_autoplay", { enabled: false });
        await reactor.sendCommand("reset");
        await reactor.disconnect();
      }
    })().catch(() => undefined);
    videoRef.current?.pause();
    audioRef.current?.pause();
    setReactorStatus("disconnected");
    setPhaseState("stopped");
    pushActivity("Reactor session stopped.");
  }, [pushActivity, setPhaseState]);

  const addFile = useCallback(async (file: File) => {
    if (!IMAGE_TYPES.has(file.type)) throw new Error("Use a JPEG, PNG, WebP, GIF, or AVIF image.");
    if (file.size > 12 * 1024 * 1024) throw new Error("Images must be 12 MB or smaller.");
    const image: LibraryImage = { id: window.crypto.randomUUID(), name: file.name || "Untitled image", dataUrl: await fileToDataUrl(file), createdAt: Date.now() };
    setImages((current) => [image, ...current].slice(0, MAX_LIBRARY_IMAGES));
    void storeImage(image).catch(() => setNotice("The image is available now but could not be saved locally."));
    return image;
  }, []);

  const scheduleImageCue = useCallback((image: LibraryImage) => {
    const cue = imageCue(image);
    const next = [...textCuesRef.current.filter((item) => item.id !== cue.id), cue].slice(-4);
    textCuesRef.current = next;
    setTextCues(next);
    const target = nextOrdinalRef.current;
    setNotice(`${cue.label} will be introduced textually in clip ${target}; queued clips stay untouched.`);
    pushActivity(`${cue.label} added as text-only scene context.`, "good");
  }, [pushActivity]);

  const selectImage = useCallback((image: LibraryImage) => {
    if (runActiveRef.current) {
      scheduleImageCue(image);
      return;
    }
    setStartImageId(image.id);
    setStartPrompt(readImagePrompts()[image.id] || (image.isDemo ? GALTON_MOUSTACHE_PROMPT : DEFAULT_PROMPT));
    setNotice(`${image.name} selected as the opening frame.`);
  }, [scheduleImageCue]);

  const handleFiles = useCallback(async (files: FileList | File[]) => {
    const selected = Array.from(files).filter((file) => IMAGE_TYPES.has(file.type));
    if (!selected.length) {
      setError("Drop a JPEG, PNG, WebP, GIF, or AVIF image.");
      return null;
    }
    try {
      const added = await Promise.all(selected.slice(0, 8).map(addFile));
      const first = added[0] ?? null;
      if (first) {
        if (runActiveRef.current) scheduleImageCue(first);
        else if (!startImageId) selectImage(first);
      }
      return first;
    } catch (fileError) {
      setError(fileError instanceof Error ? fileError.message : "Could not add the image.");
      return null;
    }
  }, [addFile, scheduleImageCue, selectImage, startImageId]);

  useEffect(() => {
    const handlePaste = (event: ClipboardEvent) => {
      const pasted = Array.from(event.clipboardData?.items ?? [])
        .filter((item) => item.kind === "file" && item.type.startsWith("image/"))
        .map((item) => item.getAsFile()).filter((file): file is File => file !== null);
      if (!pasted.length) return;
      event.preventDefault();
      void handleFiles(pasted).then((image) => {
        if (image) setNotice(`${image.name} pasted into the image library.`);
      });
    };
    window.addEventListener("paste", handlePaste);
    return () => window.removeEventListener("paste", handlePaste);
  }, [handleFiles]);

  const removeImage = useCallback((image: LibraryImage) => {
    if (image.isDemo) {
      setNotice("Galton Ramshackle is a permanent demo fixture.");
      return;
    }
    setImages((current) => current.filter((item) => item.id !== image.id));
    if (startImageId === image.id) {
      setStartImageId(null);
      setStartPrompt(DEFAULT_PROMPT);
      window.localStorage.removeItem(START_IMAGE_STORAGE_KEY);
    }
    const prompts = readImagePrompts();
    delete prompts[image.id];
    window.localStorage.setItem(IMAGE_PROMPTS_STORAGE_KEY, JSON.stringify(prompts));
    void removeStoredImage(image.id).catch(() => setNotice("The image was removed from this session only."));
  }, [startImageId]);

  const queueProp = useCallback((prop: CookingProp) => {
    const next = [...queuedPropsRef.current.filter((item) => item.id !== prop.id), prop].slice(-4);
    queuedPropsRef.current = next;
    setQueuedProps(next);
    const target = nextOrdinalRef.current;
    setNotice(`${prop.name} will enter as a text instruction in clip ${target} and persist from there.`);
    pushActivity(`${prop.name} added to persistent text context.`, "good");
  }, [pushActivity]);

  const removeQueuedProp = useCallback((propId: string) => {
    const removed = queuedPropsRef.current.find((prop) => prop.id === propId);
    const next = queuedPropsRef.current.filter((prop) => prop.id !== propId);
    queuedPropsRef.current = next;
    setQueuedProps(next);
    setNotice(`${removed?.name ?? "Prop"} will leave after already queued clips finish.`);
  }, []);

  const removeTextCue = useCallback((cueId: string) => {
    const next = textCuesRef.current.filter((cue) => cue.id !== cueId);
    textCuesRef.current = next;
    setTextCues(next);
  }, []);

  const handleStageDrop = useCallback(async (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    dragDepthRef.current = 0;
    setIsPageDragging(false);
    setIsStageDragging(false);
    setIsPropDragging(false);
    const prop = COOKING_PROPS.find((item) => item.id === event.dataTransfer.getData("application/x-h3-prop-id"));
    if (prop) return queueProp(prop);
    const libraryId = event.dataTransfer.getData("application/x-h3-image-id");
    const existing = images.find((image) => image.id === libraryId);
    if (existing) {
      if (runActiveRef.current) scheduleImageCue(existing);
      else selectImage(existing);
      return;
    }
    if (event.dataTransfer.files.length) await handleFiles(event.dataTransfer.files);
  }, [handleFiles, images, queueProp, scheduleImageCue, selectImage]);

  const dragHandlers = useMemo(() => ({
    onDragEnter: (event: DragEvent<HTMLDivElement>) => {
      const types = Array.from(event.dataTransfer.types);
      if (!types.some((type) => type === "Files" || type === "application/x-h3-image-id" || type === "application/x-h3-prop-id")) return;
      event.preventDefault();
      if (types.includes("application/x-h3-prop-id")) setIsPropDragging(true);
      else {
        dragDepthRef.current += 1;
        setIsPageDragging(true);
      }
    },
    onDragOver: (event: DragEvent<HTMLDivElement>) => { event.preventDefault(); event.dataTransfer.dropEffect = "copy"; },
    onDragLeave: () => {
      dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);
      if (!dragDepthRef.current) setIsPageDragging(false);
    },
    onDrop: (event: DragEvent<HTMLDivElement>) => {
      if ((event.target as Element).closest(".video-stage")) return;
      event.preventDefault();
      dragDepthRef.current = 0;
      setIsPageDragging(false);
      const prop = COOKING_PROPS.find((item) => item.id === event.dataTransfer.getData("application/x-h3-prop-id"));
      if (prop) queueProp(prop);
      else void handleFiles(event.dataTransfer.files);
    },
  }), [handleFiles, queueProp]);

  const handleNextPromptSubmit = useCallback((event: FormEvent) => {
    event.preventDefault();
    const direction = nextPrompt.trim();
    if (!direction) return;
    if (!runActiveRef.current) {
      setError("Start the stream before directing the next scene.");
      return;
    }
    promptRef.current = direction;
    setNextPrompt("");
    setError(null);
    setNotice(`Direction scheduled for clip ${nextOrdinalRef.current}; queued clips stay intact.`);
    pushActivity("New direction added after the committed queue.", "good");
  }, [nextPrompt, pushActivity]);

  const togglePause = useCallback(() => {
    const video = videoRef.current;
    const audio = audioRef.current;
    if (!video) return;
    if (phaseRef.current === "paused") {
      void video.play();
      void audio?.play();
      setPhaseState("playing");
    } else {
      video.pause();
      audio?.pause();
      setPhaseState("paused");
    }
  }, [setPhaseState]);

  const finishVoiceInput = useCallback(() => {
    const resume = voiceShouldResumeRef.current;
    voiceShouldResumeRef.current = false;
    voiceRecognitionRef.current = null;
    setIsVoiceListening(false);
    if (voiceCapturedSpeechRef.current) setNotice("Voice captured. Review it, then send it when ready.");
    voiceCapturedSpeechRef.current = false;
    if (resume && runActiveRef.current) {
      void videoRef.current?.play();
      void audioRef.current?.play();
      setPhaseState("playing");
    }
  }, [setPhaseState]);

  const toggleVoiceInput = useCallback(() => {
    if (isVoiceListening) return voiceRecognitionRef.current?.stop();
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Recognition) return setError("Voice input is not supported in this browser.");
    voiceShouldResumeRef.current = runActiveRef.current && phaseRef.current === "playing";
    if (voiceShouldResumeRef.current) {
      videoRef.current?.pause();
      audioRef.current?.pause();
      setPhaseState("paused");
    }
    const recognition = new Recognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = navigator.language || "en-US";
    voicePromptBaseRef.current = nextPrompt.trim();
    voiceCapturedSpeechRef.current = false;
    recognition.onresult = (event) => {
      let transcript = "";
      for (let index = 0; index < event.results.length; index += 1) transcript += `${event.results[index][0].transcript} `;
      voiceCapturedSpeechRef.current = Boolean(transcript.trim());
      setNextPrompt([voicePromptBaseRef.current, transcript.trim()].filter(Boolean).join(" ").slice(0, 2_000));
    };
    recognition.onerror = (event) => setError(event.message || `Voice input failed: ${event.error}.`);
    recognition.onspeechend = () => recognition.stop();
    recognition.onend = finishVoiceInput;
    voiceRecognitionRef.current = recognition;
    setIsVoiceListening(true);
    setNotice("Listening. Local playback is paused while you speak.");
    try { recognition.start(); } catch (voiceError) {
      setError(voiceError instanceof Error ? voiceError.message : "Could not start voice input.");
      finishVoiceInput();
    }
  }, [finishVoiceInput, isVoiceListening, nextPrompt, setPhaseState]);

  const toggleAudio = useCallback(() => {
    setAudioEnabled((current) => {
      const next = !current;
      audioEnabledRef.current = next;
      if (audioRef.current) audioRef.current.muted = !next;
      if (next) void audioRef.current?.play().catch(() => undefined);
      return next;
    });
  }, []);

  const toggleDirectorPanel = useCallback(() => setIsDirectorCollapsed((current) => !current), []);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey || event.repeat) return;
      if ((event.target instanceof Element) && event.target.closest("input, textarea, select, [contenteditable='true']")) return;
      if (event.key.toLowerCase() === "m") { event.preventDefault(); toggleAudio(); }
      if (event.key.toLowerCase() === "c") { event.preventDefault(); toggleDirectorPanel(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggleAudio, toggleDirectorPanel]);

  const propsPaneBounds = useCallback(() => {
    const panel = imagePanelRef.current;
    const library = imageLibraryRef.current;
    const propsLibrary = propsLibraryRef.current;
    const imageHeight = library?.getBoundingClientRect().height ?? MIN_IMAGE_LIBRARY_HEIGHT;
    const propsHeight = propsLibrary?.getBoundingClientRect().height ?? MIN_PROPS_PANE_HEIGHT;
    const fixedHeight = panel ? Array.from(panel.children).reduce((sum, child) => child === library || child === propsLibrary ? sum : sum + child.getBoundingClientRect().height, 0) : 0;
    const panelMax = panel ? panel.getBoundingClientRect().height - fixedHeight - MIN_IMAGE_LIBRARY_HEIGHT : imageHeight + propsHeight - MIN_IMAGE_LIBRARY_HEIGHT;
    return { min: MIN_PROPS_PANE_HEIGHT, max: Math.max(MIN_PROPS_PANE_HEIGHT, panelMax), current: propsHeight };
  }, []);

  const commitPropsPaneHeight = useCallback((height: number) => {
    const bounds = propsPaneBounds();
    const next = Math.round(Math.min(bounds.max, Math.max(bounds.min, height)));
    setPropsPaneHeight(next);
    if (storageReady) window.localStorage.setItem(PROPS_PANE_HEIGHT_STORAGE_KEY, String(next));
  }, [propsPaneBounds, storageReady]);

  const beginPropsResize = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 || window.matchMedia("(max-width: 700px)").matches) return;
    const bounds = propsPaneBounds();
    propsResizeRef.current = { pointerId: event.pointerId, startY: event.clientY, startHeight: bounds.current, maxHeight: bounds.max, height: bounds.current };
    event.currentTarget.setPointerCapture(event.pointerId);
    setIsPropsResizing(true);
    event.preventDefault();
  }, [propsPaneBounds]);

  const movePropsResize = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    const resize = propsResizeRef.current;
    if (!resize || resize.pointerId !== event.pointerId) return;
    resize.height = Math.round(Math.min(resize.maxHeight, Math.max(MIN_PROPS_PANE_HEIGHT, resize.startHeight + resize.startY - event.clientY)));
    setPropsPaneHeight(resize.height);
  }, []);

  const endPropsResize = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    const resize = propsResizeRef.current;
    if (!resize || resize.pointerId !== event.pointerId) return;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    propsResizeRef.current = null;
    setIsPropsResizing(false);
    commitPropsPaneHeight(resize.height);
  }, [commitPropsPaneHeight]);

  const resizePropsWithKeyboard = useCallback((event: ReactKeyboardEvent<HTMLDivElement>) => {
    if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return;
    event.preventDefault();
    const bounds = propsPaneBounds();
    commitPropsPaneHeight(bounds.current + (event.key === "ArrowUp" ? 24 : -24));
  }, [commitPropsPaneHeight, propsPaneBounds]);

  const pendingText = reactorStatus !== "ready"
    ? "Opening a secure Reactor session"
    : queue.generation.length
      ? `${queue.generation.length} clip${queue.generation.length === 1 ? "" : "s"} generating`
      : queue.playout.length
        ? `${queue.playout.length} clip${queue.playout.length === 1 ? "" : "s"} ready to play`
        : "Waiting for the next queued clip";

  return (
    <div className="studio-shell" {...dragHandlers}>
      {isPageDragging && !isPropDragging && (
        <div className="page-drop-overlay" aria-hidden="true"><RiImageAddLine /><strong>Drop into the image library</strong><span>Drop on the stream to add it as a text cue</span></div>
      )}

      <header className="studio-header">
        <div className="brand-cluster">
          <img className="reactor-logo" src="/brand/reactor-logo-white.svg" alt="Reactor" />
          <span className="header-rule" aria-hidden="true" />
          <div className="model-title"><strong>{FAST_H3_DISPLAY_NAME}</strong><span>INFINITE STREAM</span></div>
        </div>
        <div className="header-status">
          <span className={`stream-state state-${phase}`}><i aria-hidden="true" />{phaseLabel(phase)}</span>
          <span className="endpoint-name">{FAST_H3_MODEL}</span>
          <button type="button" className="header-stop" disabled={!isRunning} onClick={stopStream}><RiStopFill /> Stop stream</button>
        </div>
      </header>

      <main className={`workspace ${isDirectorCollapsed ? "director-collapsed" : ""}`}>
        <aside ref={imagePanelRef} className={`image-panel ${isPropsResizing ? "is-resizing-props" : ""}`} aria-label="Image library" style={propsPaneHeight === null ? undefined : ({ "--props-pane-height": `${propsPaneHeight}px` } as CSSProperties)}>
          <input ref={fileInputRef} className="visually-hidden" type="file" accept="image/jpeg,image/png,image/webp,image/gif,image/avif" multiple onChange={(event: ChangeEvent<HTMLInputElement>) => { if (event.target.files) void handleFiles(event.target.files); event.target.value = ""; }} />

          <section className="start-prompt-section reactor-start-prompt">
            <div className="start-prompt-heading"><label htmlFor="start-prompt">Starting prompt</label><span>{startImage ? startImage.name : "Text only"}</span></div>
            <textarea id="start-prompt" value={startPrompt} onChange={(event) => setStartPrompt(event.target.value.slice(0, 2_000))} rows={3} placeholder="Describe how the stream begins." />
            <div className="start-prompt-footer"><span>{startImage ? "Selected image opens the first clip" : "Select an image below if needed"}</span><button type="button" onClick={() => void startStream()}><RiPlayFill /> {clips.length ? "Restart" : "Start"}</button></div>
          </section>

          <div className="library-label"><span>Library</span><div className="library-label-actions"><small>{images.length} saved</small><button type="button" className="library-add-button" onClick={() => fileInputRef.current?.click()} aria-label="Add images"><RiAddLine /></button></div></div>
          <div ref={imageLibraryRef} className="image-library">
            {images.length === 0 ? (
              <button type="button" className="library-empty" onClick={() => fileInputRef.current?.click()}><RiImageAddLine /><strong>Add an opening frame</strong><span>Drop, paste, or browse files</span></button>
            ) : images.map((image) => (
              <article key={image.id} className={`image-card ${startImageId === image.id ? "is-start" : ""}`} draggable onDragStart={(event) => { event.dataTransfer.setData("application/x-h3-image-id", image.id); event.dataTransfer.effectAllowed = "copy"; }}>
                <button type="button" className="image-preview" onClick={() => selectImage(image)} title={isRunning ? "Add as text cue" : "Select as opening frame"}>
                  <img src={image.dataUrl} alt="" />
                  {startImageId === image.id && !isRunning && <span className="image-role"><RiCheckLine /> Selected</span>}
                </button>
                <div className="image-meta"><span title={image.name}>{image.isDemo ? "Galton Ramshackle · Demo" : image.name}</span><div><button type="button" onClick={() => selectImage(image)}>{isRunning ? "Add cue" : "Select"}</button>{!image.isDemo && <button type="button" onClick={() => removeImage(image)} aria-label={`Remove ${image.name}`}><RiDeleteBinLine /></button>}</div></div>
              </article>
            ))}
          </div>

          <div className="library-label props-label" role="separator" aria-label="Resize props panel" aria-orientation="horizontal" aria-valuemin={MIN_PROPS_PANE_HEIGHT} aria-valuenow={propsPaneHeight === null ? undefined : Math.round(propsPaneHeight)} tabIndex={0} title="Drag up or down to resize props" onPointerDown={beginPropsResize} onPointerMove={movePropsResize} onPointerUp={endPropsResize} onPointerCancel={endPropsResize} onKeyDown={resizePropsWithKeyboard} onDoubleClick={() => { setPropsPaneHeight(null); window.localStorage.removeItem(PROPS_PANE_HEIGHT_STORAGE_KEY); }}><span>Props</span><small>Text cues · drag into scene</small></div>
          <div ref={propsLibraryRef} className="props-library" aria-label="Cooking props">
            {COOKING_PROPS.map((prop) => (
              <button type="button" key={prop.id} className={`prop-card ${queuedProps.some((item) => item.id === prop.id) ? "is-queued" : ""}`} draggable title={`Add ${prop.name} as a scene instruction`} onClick={() => queueProp(prop)} onDragStart={(event) => { event.dataTransfer.setData("application/x-h3-prop-id", prop.id); event.dataTransfer.effectAllowed = "copy"; setIsPropDragging(true); }} onDragEnd={() => setIsPropDragging(false)}><span className="prop-image"><img src={prop.imageUrl} alt="" /></span><span>{prop.name}</span></button>
            ))}
          </div>
        </aside>

        <section className="stage-column" aria-label="Infinite video stream">
          <div className="stage-toolbar">
            <div className="stage-title"><span>LIVE OUTPUT</span><strong>Continuous world</strong></div>
            <div className="stage-metrics">
              <span><small>MODE</small><strong>{lastQueuedClipIdRef.current ? "Clip continuation" : startImage ? "Starting frame" : "Text"}</strong></span>
              <span><small>QUEUE</small><strong>{readyCount} ready · {queue.generation.length} building</strong></span>
              <span><small>RUN</small><strong>{formatElapsed(elapsedMs)}</strong></span>
              <span><small>AUDIO</small><strong>Native</strong></span>
              <button type="button" className={`stage-audio-toggle ${audioEnabled ? "" : "is-muted"}`} onClick={toggleAudio} aria-label={audioEnabled ? "Mute audio" : "Unmute audio"} aria-pressed={!audioEnabled} aria-keyshortcuts="M"><>{audioEnabled ? <RiVolumeUpLine /> : <RiVolumeMuteLine />}<span>{audioEnabled ? "Mute" : "Unmute"}</span><kbd>M</kbd></></button>
              <button type="button" className="icon-button" onClick={() => void stageRef.current?.requestFullscreen()} aria-label="Enter fullscreen"><RiFullscreenLine /></button>
            </div>
          </div>

          <div ref={stageRef} className={`video-stage ${hasMedia ? "has-video" : "is-empty"} ${isStageDragging ? "is-drop-target" : ""} ${isFullscreen ? "is-fullscreen" : ""}`} onDragEnter={(event) => { event.preventDefault(); setIsStageDragging(true); }} onDragOver={(event) => { event.preventDefault(); event.stopPropagation(); event.dataTransfer.dropEffect = "copy"; }} onDragLeave={(event) => { if (!event.currentTarget.contains(event.relatedTarget as Node)) setIsStageDragging(false); }} onDrop={(event) => void handleStageDrop(event)}>
            <video ref={videoRef} className="stream-video is-active" playsInline muted autoPlay onPlaying={() => runActiveRef.current && phaseRef.current !== "paused" && setPhaseState("playing")} onWaiting={() => runActiveRef.current && setPhaseState("buffering")} />
            <audio ref={audioRef} autoPlay muted={!audioEnabled} />

            {!hasMedia && startImage && <img className="stage-opening-preview" src={startImage.dataUrl} alt="" aria-hidden="true" />}
            {!hasMedia && (
              <div className={`stage-empty-state ${startImage ? "has-preview" : ""}`}>
                <div className="continuity-mark" aria-hidden="true"><span /><span /><span /></div>
                <span>{isRunning ? "REACTOR SESSION // LIVE" : "FASTH3 // READY"}</span>
                <strong>{isRunning ? `Building the first ${chunkSeconds} seconds` : "Start with a frame or just a direction"}</strong>
                <p>{isRunning ? "Native video and audio will appear as soon as the opening clip is ready." : startImage ? "The selected image will fill the first frame automatically." : "Select an image or begin from text."}</p>
                {isRunning && <div className="generation-skeleton"><i /><i /><i /></div>}
              </div>
            )}

            {isStageDragging && <div className={`stage-drop-message ${isPropDragging ? "is-prop" : ""}`}><RiImageAddLine /><strong>Add a textual scene cue</strong><span>The current queue will keep playing; this applies to the next uncommitted clip</span></div>}

            {(queuedProps.length > 0 || textCues.length > 0) && !isStageDragging && (
              <div className="queued-props" aria-label="Persistent scene cues">
                <span>In scene</span>
                {queuedProps.map((prop) => <button type="button" key={prop.id} onClick={() => removeQueuedProp(prop.id)} title={`Remove ${prop.name} from future prompts`}><img src={prop.imageUrl} alt="" /><strong>{prop.name}</strong><RiCloseLine /></button>)}
                {textCues.map((cue) => <button type="button" key={cue.id} onClick={() => removeTextCue(cue.id)} title={`Remove ${cue.label} from future prompts`}><strong>{cue.label}</strong><RiCloseLine /></button>)}
              </div>
            )}

            {hasMedia && (
              <div className="stage-overlay">
                <div className="chunk-readout"><span>CLIP</span><strong>{currentClip?.ordinal ?? "—"}</strong><small>REACTOR LIVE</small></div>
                <div className="stage-transport"><button type="button" onClick={toggleAudio} aria-label={audioEnabled ? "Mute audio" : "Enable audio"}>{audioEnabled ? <RiVolumeUpLine /> : <RiVolumeMuteLine />}</button><button type="button" onClick={togglePause} aria-label={phase === "paused" ? "Resume" : "Pause"}>{phase === "paused" ? <RiPlayFill /> : <RiPauseFill />}</button></div>
              </div>
            )}

            {hasMedia && (
              <form className="live-prompt-overlay" onSubmit={handleNextPromptSubmit}>
                <label className="visually-hidden" htmlFor="next-prompt">What happens next?</label>
                <input id="next-prompt" value={nextPrompt} onChange={(event) => setNextPrompt(event.target.value.slice(0, 2_000))} placeholder={isVoiceListening ? "Listening..." : "What happens next?"} autoComplete="off" />
                <button type="button" className={`voice-input-button ${isVoiceListening ? "is-listening" : ""}`} disabled={!isVoiceSupported} onClick={toggleVoiceInput} aria-label={isVoiceListening ? "Stop voice input" : "Start voice input"}>{isVoiceListening ? <RiStopCircleLine /> : <RiMicLine />}</button>
                <button type="submit" disabled={!nextPrompt.trim()} aria-label="Schedule direction"><RiSendPlane2Line /></button>
              </form>
            )}

            {(phase === "buffering" || phase === "generating" || phase === "connecting") && <div className="buffering-chip"><RiLoader4Line className="spin" /> {pendingText}</div>}
          </div>

          <div className="continuity-strip" aria-label="Reactor clip queue">
            <span className="strip-label">CLIP QUEUE</span>
            <div className="chunk-track">
              {clips.length === 0 ? <span className="track-empty">Queued and played clips will appear here.</span> : clips.map((clip) => (
                <div key={clip.clip_id} className={`chunk-frame clip-status status-${clip.status} ${clip.clip_id === currentClipId ? "active" : ""}`} title={clip.prompt}><span>{String(clip.ordinal).padStart(2, "0")}</span><small>{clip.status}</small></div>
              ))}
              {isRunning && <div className="pending-frame is-active" role="status" aria-live="polite"><RiLoader4Line className="spin" /><span className="pending-frame-copy"><small>Reactor queue</small><strong>{pendingText}</strong></span></div>}
            </div>
          </div>
        </section>

        <aside className={`director-panel ${isDirectorCollapsed ? "is-collapsed" : ""}`} aria-label="Stream settings">
          <div className="director-panel-bar"><button type="button" onClick={toggleDirectorPanel} aria-label={isDirectorCollapsed ? "Expand settings" : "Collapse settings"} aria-keyshortcuts="C">{isDirectorCollapsed ? <RiArrowLeftSLine /> : <RiArrowRightSLine />}{!isDirectorCollapsed && <><span>Controls</span><kbd>C</kbd></>}</button></div>
          {!isDirectorCollapsed && <>
            <section className="director-section credential-section">
              <div className="section-heading compact"><span>CONNECTION</span><strong>Server keys</strong></div>
              <div className="setting-row"><div><span>REACTOR_API_KEY</span><small>Required in .env.local</small></div><strong>Server only</strong></div>
              <p>The token route exchanges the server key for a short-lived token scoped to {FAST_H3_MODEL}. The raw key never reaches the browser.</p>
              <div className="credential-divider" />
              <div className="setting-row"><div><span>CEREBRAS_API_KEY</span><small>Optional in .env.local</small></div><strong>{hasCerebrasKey ? "Configured" : "Direct fallback"}</strong></div>
              <p>{CEREBRAS_STORY_MODEL} turns direction, persistent props, and story history into the next scene.</p>
            </section>

            <section className="director-section settings-section">
              <div className="section-heading compact"><span>GENERATION</span><strong>Stream format</strong></div>
              <div className="setting-row resolution-row"><div><span>Clip length</span><small>Applies to newly queued clips</small></div><div className="segmented">{([6, 10, 14] as const).map((seconds) => <button type="button" key={seconds} className={chunkSeconds === seconds ? "active" : ""} disabled={isRunning} onClick={() => setChunkSeconds(seconds)}>{seconds}s</button>)}</div></div>
              <div className="setting-row resolution-row"><div><span>Prebuffer window</span><small>{prebufferChunks === 0 ? "Play first clip immediately" : `Wait for ${prebufferChunks} ready clip${prebufferChunks === 1 ? "" : "s"}`}</small></div><div className="segmented prebuffer-segmented">{([0, 1, 2, 3, 4] as const).map((count) => <button type="button" key={count} className={prebufferChunks === count ? "active" : ""} disabled={isRunning} onClick={() => setPrebufferChunks(count)}>{count}</button>)}</div></div>
              <div className="setting-row"><div><span>Canvas</span><small>FastH3 launch format</small></div><strong>{FAST_H3_CANVAS}</strong></div>
              <div className="setting-row"><div><span>Audio</span><small>Generated with the video</small></div><strong>Native</strong></div>
              <div className="setting-row"><div><span>Session</span><small>WebRTC transport</small></div><strong>{reactorStatus}</strong></div>
            </section>

            <section className="director-section route-section">
              <div className="section-heading compact"><span>CONTEXT ROUTER</span><strong>Reactor clip chain</strong></div>
              <div className="route-map"><span className={!lastQueuedClipIdRef.current ? "active" : ""}>First frame</span><RiArrowRightLine /><span className={lastQueuedClipIdRef.current ? "active" : ""}>Clip ID</span><RiArrowRightLine /><span className={hasMedia ? "active" : ""}>Live track</span></div>
              <p>The selected library image is only the opening frame. Later images and props become text instructions; Reactor chains clips by ID.</p>
            </section>

            {(error || notice) && <div className={`message-box ${error ? "error" : "notice"}`} role={error ? "alert" : "status"}>{error ? <RiErrorWarningLine /> : <RiCheckLine />}<div><strong>{error ? "Stream needs attention" : "Queued"}</strong><span>{error ?? notice}</span></div><button type="button" onClick={() => { setError(null); setNotice(null); }} aria-label="Dismiss"><RiCloseLine /></button></div>}
            <div className="activity-log" aria-label="Recent stream activity">{activity.length === 0 ? <span>Reactor events will appear here.</span> : activity.map((item) => <span key={item.id} className={item.tone}>{item.text}</span>)}</div>
          </>}
        </aside>
      </main>

      <footer className="status-footer"><span>Reactor API</span><span>Native clip-ID continuity</span><span>{isRunning ? `${queuedCount} clips queued` : "Session idle"}</span></footer>
    </div>
  );
}
