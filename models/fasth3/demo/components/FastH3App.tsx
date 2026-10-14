"use client";

import {
  ReactorProvider,
  ReactorView,
  useReactor,
  useReactorMessage,
  type ReactorError,
} from "@reactor-team/js-sdk";
import Image from "next/image";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { DemoConfig } from "@/lib/config";

const TRACKS = [
  { name: "main_video", kind: "video", direction: "recvonly" },
  { name: "main_audio", kind: "audio", direction: "recvonly" },
] as const;

type PromptSource = "manual" | "bilibili" | "ai";

interface WorldState {
  type: "state_update";
  prompt: string | null;
  current_prompt: string | null;
  current_style_prompt: string | null;
  current_prompt_source: PromptSource | null;
  current_prompt_viewer_name: string | null;
  current_prompt_original_request: string | null;
  next_prompt: string | null;
  next_style_prompt: string | null;
  style_prompt: string;
  queued_prompts: string[];
  prompt_queue_depth: number;
  auto_story_enabled: boolean;
  auto_story_generating: boolean;
  auto_story_queue_target: number;
  live_chat_enabled: boolean;
  live_chat_connected: boolean;
  live_chat_room_id: number | null;
  live_prompt_pending: number;
  live_prompt_queue_depth: number;
  live_prompt_queue_limit: number;
  clip_seconds: number;
  clip_seconds_min: number;
  clip_seconds_max: number;
  seed: number;
  aspect: string;
  width: number;
  height: number;
  ready: boolean;
  running: boolean;
  paused: boolean;
  clip_index: number;
  clips_sent: number;
  seconds_sent: number;
  prompt_effective_clip_index: number;
  prompt_effective_in_seconds: number;
  valid_commands: string[];
}

interface ModelEvent {
  type?: string;
  [key: string]: unknown;
}

interface VisibleClip {
  clipIndex: number;
  prompt: string;
  source: PromptSource | null;
  viewerName: string | null;
  originalRequest: string | null;
}

interface LogItem {
  id: number;
  at: string;
  text: string;
  tone: "normal" | "error";
}

interface LivePromptItem {
  id: number;
  viewerName: string;
  request: string;
  status: "rewriting" | "queued";
  generationSeconds?: number;
  effectiveClipIndex?: number;
}

const TOKEN_REFRESH_SKEW_MS = 60_000;
let cachedToken: { jwt: string; expiresAtMs: number } | null = null;
let pendingToken: Promise<string> | null = null;

/**
 * Return one token for the whole session. The SDK asks before every request,
 * and a session only accepts the token that created it, so the token is
 * minted once and reused until shortly before it expires.
 */
async function fetchToken(): Promise<string> {
  if (cachedToken && Date.now() < cachedToken.expiresAtMs - TOKEN_REFRESH_SKEW_MS) {
    return cachedToken.jwt;
  }
  pendingToken ??= (async () => {
    try {
      const response = await fetch("/api/token", { cache: "no-store" });
      const body = (await response.json().catch(() => ({}))) as {
        jwt?: string;
        expires_at?: number;
        error?: string;
      };
      if (!response.ok || !body.jwt || !body.expires_at) {
        throw new Error(body.error ?? `Token request failed: ${response.status}`);
      }
      cachedToken = { jwt: body.jwt, expiresAtMs: body.expires_at * 1000 };
      return body.jwt;
    } finally {
      pendingToken = null;
    }
  })();
  return pendingToken;
}

function unwrap(raw: unknown): ModelEvent {
  const envelope = raw as {
    type?: string;
    data?: Record<string, unknown>;
  };
  if (
    envelope &&
    typeof envelope === "object" &&
    envelope.data &&
    typeof envelope.data === "object"
  ) {
    return { ...envelope.data, type: envelope.type };
  }
  return (raw ?? {}) as ModelEvent;
}

function statusLabel(status: string): string {
  return (
    {
      disconnected: "Not connected",
      connecting: "Connecting",
      waiting: "Waiting",
      ready: "Connected",
      error: "Connection failed",
    }[status] ?? status
  );
}

function sourceLabel(source: PromptSource | null): string {
  if (source === "bilibili") return "Bilibili";
  if (source === "ai") return "AI generated";
  if (source === "manual") return "Manual submission";
  return "Not playing yet";
}

function Button({
  children,
  onClick,
  disabled,
  primary = false,
  danger = false,
}: {
  children: React.ReactNode;
  onClick: () => void;
  disabled?: boolean;
  primary?: boolean;
  danger?: boolean;
}) {
  const color = primary
    ? "border-white bg-white text-black hover:bg-zinc-200"
    : danger
      ? "border-error/60 text-error hover:bg-error/10"
      : "border-edge bg-raised text-ink hover:border-faint";
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`h-9 rounded-md border px-3 text-xs font-semibold transition disabled:cursor-not-allowed disabled:opacity-35 ${color}`}
    >
      {children}
    </button>
  );
}

function Panel({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-lg border border-edge bg-panel p-4">
      <h2 className="mb-3 text-[11px] font-semibold uppercase tracking-[0.15em] text-dim">
        {title}
      </h2>
      {children}
    </section>
  );
}

function FastH3Workspace({ config }: { config: DemoConfig }) {
  const connect = useReactor((state) => state.connect);
  const disconnect = useReactor((state) => state.disconnect);
  const sendCommand = useReactor((state) => state.sendCommand);
  const status = useReactor((state) => state.status);
  const reactor = useReactor((state) => state.internal.reactor);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [world, setWorld] = useState<WorldState | null>(null);
  const [visibleClip, setVisibleClip] = useState<VisibleClip | null>(null);
  const [prompt, setPrompt] = useState("");
  const [seconds, setSeconds] = useState(14.375);
  const [seed, setSeed] = useState(1000);
  const [aspect, setAspect] = useState("16:9");
  const [logs, setLogs] = useState<LogItem[]>([]);
  const [livePrompts, setLivePrompts] = useState<LivePromptItem[]>([]);
  const [connectedAt, setConnectedAt] = useState<number | null>(null);
  const [connectedFor, setConnectedFor] = useState(0);
  const nextLogId = useRef(1);
  const nextLivePromptId = useRef(1);
  const mediaRoot = useRef<HTMLDivElement>(null);
  const visibleClipIndex = useRef(-1);
  const visibleSyncToken = useRef(0);

  const appendLog = useCallback((text: string, tone: LogItem["tone"] = "normal") => {
    setLogs((items) => [
      {
        id: nextLogId.current++,
        at: new Date().toLocaleTimeString(),
        text,
        tone,
      },
      ...items,
    ].slice(0, 40));
  }, []);

  const alignVisibleClip = useCallback((snapshot: WorldState) => {
    if (snapshot.current_prompt === null || snapshot.clip_index < 0) {
      visibleSyncToken.current += 1;
      visibleClipIndex.current = -1;
      setVisibleClip(null);
      return;
    }
    if (snapshot.clip_index === visibleClipIndex.current) return;

    const token = ++visibleSyncToken.current;
    const candidate: VisibleClip = {
      clipIndex: snapshot.clip_index,
      prompt: snapshot.current_prompt,
      source: snapshot.current_prompt_source,
      viewerName: snapshot.current_prompt_viewer_name,
      originalRequest: snapshot.current_prompt_original_request,
    };
    const commit = () => {
      if (token !== visibleSyncToken.current) return;
      visibleClipIndex.current = candidate.clipIndex;
      setVisibleClip(candidate);
    };
    const waitForVideo = (attempt = 0) => {
      if (token !== visibleSyncToken.current) return;
      const video = mediaRoot.current?.querySelector("video");
      if (!video || typeof video.requestVideoFrameCallback !== "function") {
        if (attempt < 20) {
          window.setTimeout(() => waitForVideo(attempt + 1), 50);
        } else {
          commit();
        }
        return;
      }

      let framesPresented = 0;
      const onFrame = () => {
        if (token !== visibleSyncToken.current) return;
        framesPresented += 1;
        if (framesPresented < 2) {
          video.requestVideoFrameCallback(onFrame);
        } else {
          commit();
        }
      };
      video.requestVideoFrameCallback(onFrame);
    };
    waitForVideo();
  }, []);

  useReactorMessage((raw: unknown) => {
    const message = unwrap(raw);
    if (message.type === "state_update") {
      const snapshot = message as unknown as WorldState;
      setWorld(snapshot);
      setSeconds(snapshot.clip_seconds);
      setSeed(snapshot.seed);
      setAspect(snapshot.aspect);
      alignVisibleClip(snapshot);
      return;
    }
    if (message.type === "prompt_accepted") {
      appendLog(
        `Prompt queued for clip ${String(message.effective_clip_index)} · queue ${String(message.queue_depth)}`,
      );
      return;
    }
    if (message.type === "auto_prompt_queued") {
      appendLog(
        `Story writer queued a scene · queue ${String(message.queue_depth)}${message.fallback_used ? " · fallback" : ""}`,
      );
      return;
    }
    if (message.type === "auto_story_accepted") {
      appendLog(`Infinite story ${message.enabled ? "enabled" : "disabled"}`);
      return;
    }
    if (message.type === "live_chat_status") {
      appendLog(
        `Bilibili chat ${message.connected ? "connected" : "disconnected"}${message.detail ? ` · ${String(message.detail)}` : ""}`,
        message.detail ? "error" : "normal",
      );
      return;
    }
    if (message.type === "live_prompt_received") {
      const viewerName = String(message.viewer_name);
      const request = String(message.request);
      setLivePrompts((items) => {
        const received: LivePromptItem = {
          id: nextLivePromptId.current++,
          viewerName,
          request,
          status: "rewriting",
        };
        return [received, ...items].slice(0, 8);
      });
      appendLog(
        `${viewerName} requested · ${request}`,
      );
      return;
    }
    if (message.type === "live_prompt_queued") {
      const viewerName = String(message.viewer_name);
      const request = String(message.request);
      setLivePrompts((items) => {
        let match = -1;
        for (let index = items.length - 1; index >= 0; index -= 1) {
          const item = items[index];
          if (
            item.status === "rewriting" &&
            item.viewerName === viewerName &&
            item.request === request
          ) {
            match = index;
            break;
          }
        }
        const queued: LivePromptItem = {
          id: match >= 0 ? items[match].id : nextLivePromptId.current++,
          viewerName,
          request,
          status: "queued",
          generationSeconds: Number(message.generation_seconds),
          effectiveClipIndex: Number(message.effective_clip_index),
        };
        if (match < 0) return [queued, ...items].slice(0, 8);
        return items.map((item, index) => (index === match ? queued : item));
      });
      appendLog(
        `Viewer scene queued for clip ${String(message.effective_clip_index)} · ${String(message.generation_seconds)}s${message.fallback_used ? " · fallback" : ""}`,
      );
      return;
    }
    if (message.type === "style_accepted") {
      appendLog(
        `Style set for clip ${String(message.effective_clip_index)} and later`,
      );
      return;
    }
    if (message.type === "clip_started") {
      appendLog(
        `Clip ${String(message.clip_index)} now playing · ${sourceLabel(message.source as PromptSource)}`,
      );
      return;
    }
    if (message.type === "clip_complete") {
      appendLog(`Clip ${String(message.clip_index)} complete`);
      return;
    }
    if (message.type === "channel_failed") {
      appendLog(String(message.reason ?? "Channel failed"), "error");
      return;
    }
    if (message.type) appendLog(message.type.replaceAll("_", " "));
  });

  useEffect(() => {
    if (status === "ready" && connectedAt === null) {
      setConnectedAt(Date.now());
      setConnectionError(null);
      appendLog("WebRTC connected");
    } else if (status !== "ready" && connectedAt !== null) {
      appendLog(`WebRTC disconnected after ${connectedFor}s`, "error");
      setConnectedAt(null);
      setWorld(null);
      visibleSyncToken.current += 1;
      visibleClipIndex.current = -1;
      setVisibleClip(null);
    }
  }, [appendLog, connectedAt, connectedFor, status]);

  useEffect(() => {
    if (connectedAt === null) {
      setConnectedFor(0);
      return;
    }
    const update = () => setConnectedFor(Math.floor((Date.now() - connectedAt) / 1000));
    update();
    const timer = window.setInterval(update, 1000);
    return () => window.clearInterval(timer);
  }, [connectedAt]);

  useEffect(() => {
    const onError = (error: ReactorError) => {
      // A refused command is reported by the call that sent it.
      if (error.operation === "sendCommand") return;
      setConnectionError(error.message);
      appendLog(error.message, "error");
    };
    reactor.on("error", onError);
    return () => reactor.off("error", onError);
  }, [appendLog, reactor]);

  const valid = useMemo(
    () => new Set(world?.valid_commands ?? []),
    [world?.valid_commands],
  );
  const connected = status === "ready";
  // `sendCommand` never rejects. A refused command resolves `undefined` and
  // emits its `ReactorError`, carrying the model's code, on the `error` event
  // while the call is in flight; only this client is told.
  const command = useCallback(
    async (name: string, params: Record<string, unknown> = {}) => {
      let refusal: ReactorError | null = null;
      const onError = (error: ReactorError) => {
        if (error.operation === "sendCommand" && error.message.startsWith(`command '${name}' failed`)) {
          refusal = error;
        }
      };
      reactor.on("error", onError);
      try {
        await sendCommand(name, params);
      } finally {
        reactor.off("error", onError);
      }
      const error = refusal as ReactorError | null;
      if (error) appendLog(error.message, "error");
    },
    [appendLog, reactor, sendCommand],
  );

  return (
    <div className="flex h-screen flex-col overflow-hidden">
      <header className="flex shrink-0 flex-wrap items-center gap-3 border-b border-edge px-4 py-3">
        <div className="flex items-center gap-3">
          <Image
            src="/brand/reactor-lockup-white.png"
            alt="Reactor"
            width={174}
            height={20}
            priority
            className="h-5 w-auto"
          />
          <span className="h-4 w-px bg-edge" aria-hidden="true" />
          <h1 className="text-sm font-semibold">FastH3 live channel</h1>
        </div>
        <span className="rounded-full border border-edge px-2.5 py-1 text-[11px] text-dim">
          {config.mode} · {config.apiUrl ?? config.modelName}
        </span>
        <span className="ml-auto flex items-center gap-2 text-xs text-dim">
          <span
            className={`size-2 rounded-full ${connected ? "bg-live" : connectionError ? "bg-error" : "bg-faint"}`}
          />
          {statusLabel(status)}
          {connected ? ` · ${connectedFor}s` : ""}
        </span>
        {connected ? (
          <Button onClick={() => void disconnect()}>Disconnect</Button>
        ) : (
          <Button
            primary
            disabled={status === "connecting" || status === "waiting"}
            onClick={() => void connect().catch((error: unknown) => appendLog(error instanceof Error ? error.message : "Connection failed", "error"))}
          >
            Connect
          </Button>
        )}
      </header>

      <main className="flex min-h-0 flex-1 flex-col lg:flex-row">
        <aside className="flex w-full shrink-0 flex-col gap-3 overflow-y-auto border-edge p-3 lg:w-[25rem] lg:border-r">
          <Panel title="Prompt queue">
            <textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              rows={5}
              maxLength={800}
              placeholder="Describe the next clip…"
              className="w-full resize-none rounded-md border border-edge bg-raised px-3 py-2 text-xs leading-relaxed outline-none focus:border-accent"
            />
            <div className="mt-2 flex gap-2">
              <Button
                primary
                disabled={!connected || !prompt.trim()}
                onClick={() => {
                  void command("set_prompt", { prompt: prompt.trim() });
                  setPrompt("");
                }}
              >
                Queue prompt
              </Button>
              <Button
                disabled={!connected}
                onClick={() => void command("set_prompt", { prompt: "" })}
              >
                Clear waiting
              </Button>
            </div>
            <div className="mt-3 rounded-md border border-edge bg-raised p-3 text-[11px]">
              <div className="flex items-center justify-between gap-3">
                <span className="font-semibold uppercase tracking-[0.12em] text-faint">
                  Playing now
                </span>
                <span className="text-dim">
                  {visibleClip ? `clip ${visibleClip.clipIndex}` : "waiting"}
                </span>
              </div>
              <dl className="mt-2 grid grid-cols-[4rem_1fr] gap-x-2 gap-y-1">
                <dt className="text-faint">Source</dt>
                <dd className="text-dim">{sourceLabel(visibleClip?.source ?? null)}</dd>
                {visibleClip?.source === "bilibili" ? (
                  <>
                    <dt className="text-faint">Viewer</dt>
                    <dd className="text-dim">{visibleClip.viewerName ?? "anonymous"}</dd>
                    <dt className="text-faint">Request</dt>
                    <dd className="break-words text-ink">
                      {visibleClip.originalRequest ?? "unavailable"}
                    </dd>
                  </>
                ) : null}
                <dt className="text-faint">Prompt</dt>
                <dd className="max-h-24 overflow-y-auto break-words text-dim">
                  {visibleClip?.prompt ?? "No clip has reached playback yet."}
                </dd>
              </dl>
            </div>
            <div className="mt-2 text-[11px] text-faint">
              Waiting for playback: {world?.prompt_queue_depth ?? 0}
            </div>
          </Panel>

          <Panel title="Infinite story">
            <p className="text-xs leading-relaxed text-dim">
              After 20 seconds, GPT-5.4 Mini uses the latest seven scenes to
              keep two future prompts ready. Prompts you submit keep FIFO priority.
            </p>
            <div className="mt-3 flex items-center justify-between gap-3">
              <span className="text-[11px] text-faint">
                {world?.auto_story_generating
                  ? "Writing next scene…"
                  : world?.auto_story_enabled
                    ? `On · target ${world.auto_story_queue_target}`
                    : "Off"}
              </span>
              <Button
                disabled={!connected || !valid.has("set_auto_story")}
                onClick={() =>
                  void command("set_auto_story", {
                    enabled: !(world?.auto_story_enabled ?? true),
                  })
                }
              >
                {world?.auto_story_enabled ? "Disable" : "Enable"}
              </Button>
            </div>
          </Panel>

          <Panel title="Bilibili live chat">
            <p className="text-xs leading-relaxed text-dim">
              Comments containing <span className="text-ink">!Prompt:</span>{" "}
              are translated into complete English FastH3 scenes and enter the
              viewer FIFO.
            </p>
            <div className="mt-3 space-y-1 text-[11px] text-faint">
              <p>
                Status:{" "}
                <span className={world?.live_chat_connected ? "text-live" : "text-dim"}>
                  {world?.live_chat_enabled
                    ? world.live_chat_connected
                      ? "receiving"
                      : "connecting"
                    : "disabled"}
                </span>
              </p>
              <p>
                Room:{" "}
                {world?.live_chat_room_id ? (
                  <a
                    className="text-dim underline decoration-edge underline-offset-2"
                    href={`https://live.bilibili.com/${world.live_chat_room_id}`}
                    target="_blank"
                    rel="noreferrer"
                  >
                    {world.live_chat_room_id}
                  </a>
                ) : (
                  <span className="text-dim">none</span>
                )}
              </p>
              <p>
                Waiting for rewrite:{" "}
                <span className="text-dim">{world?.live_prompt_pending ?? 0}</span>
              </p>
              <p>
                Bilibili backlog:{" "}
                <span className="text-dim">
                  {world?.live_prompt_queue_depth ?? 0} / {world?.live_prompt_queue_limit ?? 10}
                </span>
              </p>
            </div>
            <div className="mt-3 space-y-2 border-t border-edge pt-3">
              <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-faint">
                Recent viewer prompts
              </p>
              {livePrompts.length === 0 ? (
                <p className="text-[11px] text-faint">No matching comments received yet.</p>
              ) : livePrompts.map((item) => (
                <div key={item.id} className="rounded-md border border-edge bg-raised px-2.5 py-2 text-[11px]">
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate text-dim">{item.viewerName}</span>
                    <span className={item.status === "queued" ? "text-live" : "text-faint"}>
                      {item.status === "queued"
                        ? `queued · clip ${item.effectiveClipIndex} · ${item.generationSeconds?.toFixed(2)}s`
                        : "rewriting…"}
                    </span>
                  </div>
                  <p className="mt-1 break-words text-ink">{item.request}</p>
                </div>
              ))}
            </div>
          </Panel>

          <Panel title="Channel">
            <div className="grid grid-cols-2 gap-2">
              <Button
                primary
                disabled={!connected || !valid.has("start")}
                onClick={() => void command("start")}
              >
                Start
              </Button>
              <Button
                disabled={!connected || (!valid.has("pause") && !valid.has("resume"))}
                onClick={() => void command(world?.paused ? "resume" : "pause")}
              >
                {world?.paused ? "Resume" : "Pause"}
              </Button>
              <Button
                danger
                disabled={!connected || !valid.has("stop")}
                onClick={() => void command("stop")}
              >
                Stop
              </Button>
              <Button
                disabled={!connected || !valid.has("reset")}
                onClick={() => void command("reset")}
              >
                Reset
              </Button>
            </div>
            <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 text-[11px]">
              <dt className="text-faint">State</dt><dd className="text-right text-dim">{world?.running ? world.paused ? "paused" : "streaming" : world?.ready ? "ready" : "idle"}</dd>
              <dt className="text-faint">Clip</dt><dd className="text-right text-dim">{world?.clip_index ?? -1}</dd>
              <dt className="text-faint">Clips sent</dt><dd className="text-right text-dim">{world?.clips_sent ?? 0}</dd>
              <dt className="text-faint">Content</dt><dd className="text-right text-dim">{world?.seconds_sent ?? 0}s</dd>
            </dl>
          </Panel>

          <Panel title="Generation settings">
            <label className="block text-[11px] text-faint">
              Clip length · {seconds.toFixed(3)}s
              <input
                type="range"
                min={world?.clip_seconds_min ?? 5.167}
                max={world?.clip_seconds_max ?? 14.375}
                step="0.001"
                value={seconds}
                onChange={(event) => setSeconds(Number(event.target.value))}
                onPointerUp={() => void command("set_clip_seconds", { seconds })}
                disabled={!connected}
                className="mt-2 w-full"
              />
            </label>
            <div className="mt-4 grid grid-cols-[1fr_auto] gap-2">
              <input
                type="number"
                min="0"
                value={seed}
                onChange={(event) => setSeed(Number(event.target.value))}
                className="min-w-0 rounded-md border border-edge bg-raised px-3 text-xs outline-none focus:border-accent"
              />
              <Button disabled={!connected} onClick={() => void command("set_seed", { seed })}>Set seed</Button>
              <select
                value={aspect}
                onChange={(event) => setAspect(event.target.value)}
                disabled={!connected || !valid.has("set_canvas")}
                className="min-w-0 rounded-md border border-edge bg-raised px-3 text-xs outline-none"
              >
                {[
                  "16:9",
                  "1:1",
                  "9:16",
                  "4:3",
                ].map((value) => <option key={value}>{value}</option>)}
              </select>
              <Button disabled={!connected || !valid.has("set_canvas")} onClick={() => void command("set_canvas", { aspect })}>Set canvas</Button>
            </div>
          </Panel>
        </aside>

        <section className="flex min-h-0 min-w-0 flex-1 flex-col">
          <div ref={mediaRoot} className="relative min-h-[18rem] flex-1 bg-black">
            <ReactorView
              className="absolute inset-0 size-full"
              track="main_video"
              audioTrack="main_audio"
              muted={false}
              videoObjectFit="contain"
            />
            {!connected || !world?.running ? (
              <div className="pointer-events-none absolute inset-0 flex items-center justify-center p-8">
                <div className="max-w-md rounded-lg border border-white/10 bg-black/65 p-5 text-center backdrop-blur">
                  <p className="text-sm text-dim">
                    {!connected
                      ? "Connect to open a WebRTC session."
                      : world?.ready
                        ? "Press Start. The opening clip is generated before its first frame appears."
                        : "Queue a prompt, then press Start."}
                  </p>
                </div>
              </div>
            ) : null}
            {connected && world ? (
              <div className="absolute bottom-3 left-3 rounded-md border border-white/10 bg-black/60 px-3 py-2 text-[11px] text-white/70 backdrop-blur">
                {world.width}×{world.height} · 24 fps · playing clip {visibleClip?.clipIndex ?? "waiting"}
              </div>
            ) : null}
          </div>
          <div className="h-40 shrink-0 overflow-y-auto border-t border-edge bg-panel px-4 py-3">
            <h2 className="mb-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-faint">Model messages</h2>
            {logs.length === 0 ? (
              <p className="text-xs text-faint">No messages yet.</p>
            ) : logs.map((item) => (
              <p key={item.id} className={`mb-1 text-xs ${item.tone === "error" ? "text-error" : "text-dim"}`}>
                <span className="mr-2 text-faint">{item.at}</span>{item.text}
              </p>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}

/** Configure the same SDK connection path used by the Echo-WM demo. */
export function FastH3App({ config }: { config: DemoConfig }) {
  const local = config.mode === "local";
  return (
    <ReactorProvider
      modelName={config.modelName}
      modelTracks={[...TRACKS]}
      local={local}
      {...(config.apiUrl ? { apiUrl: config.apiUrl } : {})}
      {...(local ? {} : { jwtToken: fetchToken })}
      connectOptions={{ autoConnect: false }}
    >
      <FastH3Workspace config={config} />
    </ReactorProvider>
  );
}
