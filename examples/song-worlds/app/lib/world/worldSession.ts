/**
 * Step 2 — thin abstraction over the Reactor connection.
 *
 * `WorldSession` is the only surface the app talks to for world
 * generation, so the backing model can be swapped (Helios ↔
 * LingBot-World) via the WORLD_MODEL config flag, and the whole thing
 * can run in mock mode with zero live API access.
 *
 *   createWorldSession("reactor") → real Reactor session (WORLD_MODEL)
 *   createWorldSession("mock")    → no-network stub; the procedural
 *                                   <MockWorldCanvas> renders instead
 *
 * Image-conditioned generation — how and WHY (verified against the
 * installed @reactor-models/helios 0.9.14 schema):
 *
 *  - Seed images ride INLINE as `image_b64` on `set_image`. The
 *    presigned-URL upload path (uploadFile → FileRef) 404s on this
 *    deployment: the FileRef pointed at a file the model couldn't
 *    resolve, `set_conditioning` rejected it, its atomic rollback left
 *    no prompt committed, and `start` then failed with "No prompt set"
 *    → black stage. `image_b64` needs no upload endpoint at all.
 *    (It's marked deprecated but fully supported; if it's ever removed,
 *    the FileRef 404 needs investigating — see git history.)
 *  - `set_conditioning` is NOT used: the installed schema only accepts
 *    a FileRef image there (no image_b64 field). Its purpose was to
 *    close the race between `start` and async upload resolution — with
 *    an inline image there is no upload, and we get the same guarantee
 *    by AWAITING the model's acks (`image_accepted`, `conditions_ready`)
 *    before ever sending `start`.
 *  - prepareSeeds() runs during the loading phase: every distinct seed
 *    is fetched and downscaled to a base64 JPEG once (cache
 *    seedId → b64), so mid-song swaps never wait on fetch/encode.
 *  - Helios advances in 33-frame chunks (~CHUNK_SECONDS each); every
 *    command lands on a chunk boundary. After `start`, the FULL prompt
 *    series is pre-scheduled via `schedule_prompt({chunk, prompt})`
 *    (multiple queued prompts are supported; latest at-or-before the
 *    current chunk wins), so prompt changes track the generation clock
 *    exactly. During playback the score executor only hot-swaps seed
 *    IMAGES (set_image applies from the next chunk; the stream never
 *    tears down), with image_strength eased for morphs vs cuts.
 *
 * ┌────────────────────────────────────────────────────────────────────┐
 * │ REAL-API SWAP POINT                                                │
 * │ Everything needed for live generation is in ReactorWorldSession    │
 * │ below. It authenticates through GET /api/reactor/token (which      │
 * │ needs REACTOR_API_KEY in .env — see app/api/reactor/token/route.ts)│
 * │ and sends the model conditioning/start commands. When credentials  │
 * │ are live, the app picks it automatically; no code changes needed.  │
 * └────────────────────────────────────────────────────────────────────┘
 */

import { Reactor, type ReactorStatus } from "@reactor-team/js-sdk";
import { LingbotWorld2Model, type FileRef } from "@reactor-models/lingbot-world-2";
import type { CameraIntent } from "./audioMovement";
import {
  BEAT_PULSE_HOLD_MS,
  BEAT_PULSE_MIN_INTERVAL_MS,
  BEAT_PULSE_SUFFIX,
  CHUNK_SECONDS,
  IMAGE_STRENGTH,
  IMAGE_STRENGTH_MORPH,
  NO_CHARACTER_SUFFIX,
  STATIC_CAMERA_SUFFIX,
  SEED_JPEG_QUALITY,
  SEED_MAX_DIMENSION,
  USER_LOOK_PITCH_DEG,
  USER_LOOK_YAW_DEG,
  USER_MOVE_STEP,
  WORLD_MODEL,
} from "./config";

/**
 *  - "reactor" → Helios (continuous single stream; seeds hot-swap
 *    mid-stream via inline base64; the app's original path).
 *  - "lingbot" → Lingbot World 2 (navigable world; each seed change is
 *    a NEW run via reset→set_image→start, because Lingbot can't hot-swap
 *    a reference image; seeds MUST be uploaded via uploadFile — no
 *    base64 form exists, and start REQUIRES an image).
 *  - "mock"    → no network; procedural canvas.
 */
export type WorldSessionKind = "reactor" | "lingbot" | "mock";

/** What one score event asks of the world model: this seed, steered by
 *  this prompt, entered this way. The session decides which wire
 *  commands that needs. Structurally a subset of WorldEvent, so score
 *  events can be passed straight in. */
export interface WorldScene {
  seedId: string;
  prompt: string;
  transition: "cut" | "morph";
}

export interface WorldSession {
  readonly kind: WorldSessionKind;
  /** Resolves once the model is ready to accept commands. */
  connect(): Promise<void>;
  /**
   * Fires when the model REJECTS something after connect (command_error
   * on the data channel, or no video arriving after start). sendCommand
   * resolving only means "sent" — without this, a model-side rejection
   * is invisible and the stage just stays black.
   */
  onError(cb: (message: string) => void): void;
  /**
   * Fetch + downscale + base64-encode every seed the score uses
   * (seedId → b64 cache). Call during the loading phase — swaps at
   * event boundaries must never wait on fetch/encode work.
   */
  prepareSeeds(seeds: { id: string; imageUrl: string }[]): Promise<void>;
  /** Commit/apply one pre-authored score event (seed + prompt). */
  applyScene(scene: WorldScene): Promise<void>;
  /** Begin generation (after the first scene is committed). */
  start(): Promise<void>;
  /**
   * Apply a USER-driven camera intent (Lingbot only — the navigable
   * model). Diffs against the last intent and sends only the
   * movement/look setters that changed. No-op on Helios/mock (they have
   * no camera inputs). Safe to call every playback tick.
   */
  driveCamera(intent: CameraIntent): void;
  /**
   * Pulse the GENERATED world's activity on a downbeat (Lingbot only):
   * briefly intensify the active prompt so the generation surges on the
   * bar, then revert. No-op on Helios/mock. Throttled internally.
   */
  pulse(): void;
  /**
   * Pre-schedule the rest of the score's prompts onto the generation
   * clock (chunk indices). Call once, right after start(). Prompt
   * changes then fire model-side on exact chunk boundaries even if the
   * page's timers hiccup; playback only needs to send image swaps.
   */
  schedulePrompts(entries: { timestamp: number; prompt: string }[]): Promise<void>;
  /** Tear the session down cleanly. Safe to call more than once. */
  close(): Promise<void>;
  /** Fires with the live video track (reactor mode only). */
  onTrack(cb: (track: MediaStreamTrack) => void): void;
  onStatus(cb: (status: ReactorStatus) => void): void;
}

/** Never send scene commands closer together than this, whatever the
 *  caller does — last line of defense against flooding the session.
 *  (Sanitized events are already ≥2s apart, so this never fires in
 *  normal playback.) */
const MIN_SEND_INTERVAL_MS = 500;

/** How long to wait for a model ack before treating a command as
 *  silently dropped. Image validation includes a VAE encode, so it
 *  gets longer than a prompt ack. */
const IMAGE_ACK_TIMEOUT_MS = 12_000;
const PROMPT_ACK_TIMEOUT_MS = 5_000;

// Per-chunk camera-pose step sizes (radians / translation units), applied
// while a key is held. See config for the degree values + rationale.
const DEG2RAD = Math.PI / 180;
const POSE_YAW_RAD = USER_LOOK_YAW_DEG * DEG2RAD;
const POSE_PITCH_RAD = USER_LOOK_PITCH_DEG * DEG2RAD;
const POSE_MOVE = USER_MOVE_STEP;

type ModelMessage = Record<string, unknown> & { type?: string };

/** Fetch a seed image and re-encode it as a compact base64 JPEG (no
 *  data: prefix). Downscaling is safe: the model center-crops/resizes
 *  to its output resolution anyway (image_accepted reports the final
 *  size), and it keeps the data-channel payload small. */
async function fetchSeedAsBase64(imageUrl: string): Promise<string> {
  const res = await fetch(imageUrl);
  if (!res.ok) {
    throw new Error(`GET ${imageUrl} → ${res.status}`);
  }
  const bitmap = await createImageBitmap(await res.blob());
  const scale = Math.min(
    1,
    SEED_MAX_DIMENSION / Math.max(bitmap.width, bitmap.height),
  );
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("2d canvas unavailable");
  ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close();
  const dataUrl = canvas.toDataURL("image/jpeg", SEED_JPEG_QUALITY);
  return dataUrl.slice(dataUrl.indexOf(",") + 1);
}

class ReactorWorldSession implements WorldSession {
  readonly kind = "reactor" as const;
  private reactor: Reactor;
  private trackCbs: ((track: MediaStreamTrack) => void)[] = [];
  private statusCbs: ((status: ReactorStatus) => void)[] = [];
  private errorCbs: ((message: string) => void)[] = [];
  private waiters: {
    match: (m: ModelMessage) => boolean;
    resolve: (m: ModelMessage) => void;
  }[] = [];
  private lastSendAt = 0;
  private closed = false;
  private started = false;
  private gotVideoTrack = false;
  private trackWatchdog: ReturnType<typeof setTimeout> | null = null;
  /** seedId → base64 JPEG, filled by prepareSeeds. */
  private seedB64 = new Map<string, string>();
  private currentSeedId: string | null = null;
  private currentImageStrength = IMAGE_STRENGTH;

  constructor() {
    // Swap WORLD_MODEL in config.ts to A/B Helios vs LingBot-World —
    // both speak the same conditioning command surface.
    this.reactor = new Reactor({ modelName: WORLD_MODEL });

    this.reactor.on("trackReceived", (name: string, track: MediaStreamTrack) => {
      console.info(`[world] trackReceived: "${name}"`);
      if (name === "main_video") {
        this.gotVideoTrack = true;
        for (const cb of this.trackCbs) cb(track);
      }
    });
    this.reactor.on("statusChanged", (status: ReactorStatus) => {
      for (const cb of this.statusCbs) cb(status);
    });
    // Model data-channel messages. This is where rejections surface —
    // sendCommand only confirms transport, never acceptance. Everything
    // is logged so a black stage is diagnosable from the console.
    // Wire shape is an envelope `{ type, data: {...fields} }`; flatten
    // exactly like the typed Helios client's _unwrapMessage does.
    this.reactor.on("message", (raw: unknown) => {
      if (!raw || typeof raw !== "object") return;
      const env = raw as { type?: string; data?: Record<string, unknown> };
      const msg: ModelMessage =
        env.data && typeof env.data === "object"
          ? { ...env.data, type: env.type }
          : (raw as ModelMessage);

      // Feed anyone awaiting an ack (image_accepted / conditions_ready
      // / command_error / …) before generic logging.
      this.waiters = this.waiters.filter((w) => {
        if (!w.match(msg)) return true;
        w.resolve(msg);
        return false;
      });

      if (msg.type === "command_error") {
        const text = `Model rejected "${msg.command ?? "?"}": ${msg.reason ?? "no reason given"}`;
        console.error(`[world] ${text}`, msg);
        for (const cb of this.errorCbs) cb(text);
      } else if (msg.type === "chunk_complete") {
        // Once per ~1.4s — useful for eyeballing generation-clock vs
        // audio-clock drift, but too chatty for info.
        console.debug(
          `[world] chunk_complete #${msg.chunk_index} (prompt: ${String(msg.active_prompt).slice(0, 50)}…)`,
        );
      } else if (msg.type && msg.type !== "state") {
        // prompt_accepted / image_accepted / conditions_ready /
        // generation_started — the model lifecycle.
        console.info(`[world] model: ${msg.type}`, msg);
      }
    });
  }

  onTrack(cb: (track: MediaStreamTrack) => void) {
    this.trackCbs.push(cb);
  }
  onStatus(cb: (status: ReactorStatus) => void) {
    this.statusCbs.push(cb);
  }
  onError(cb: (message: string) => void) {
    this.errorCbs.push(cb);
  }

  /** Resolve with the first message matching `match`, or null on
   *  timeout (= the deployment silently dropped the command). */
  private waitFor(
    match: (m: ModelMessage) => boolean,
    timeoutMs: number,
  ): Promise<ModelMessage | null> {
    return new Promise((resolve) => {
      const waiter = { match, resolve: (m: ModelMessage) => resolve(m) };
      this.waiters.push(waiter);
      setTimeout(() => {
        this.waiters = this.waiters.filter((w) => w !== waiter);
        resolve(null);
      }, timeoutMs);
    });
  }

  async connect(): Promise<void> {
    const ready = new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(
        () => reject(new Error("Reactor session timed out waiting for ready")),
        90_000,
      );
      this.reactor.on("statusChanged", (status: ReactorStatus) => {
        if (status === "ready") {
          clearTimeout(timeout);
          resolve();
        }
      });
      this.reactor.on("error", (err: { message?: string }) => {
        clearTimeout(timeout);
        reject(new Error(err?.message ?? "Reactor connection error"));
      });
    });

    // getJwt resolver: minted server-side from REACTOR_API_KEY, browser-
    // cached via Cache-Control until expiry (see the token route).
    await this.reactor.connect(async () => {
      const r = await fetch("/api/reactor/token");
      if (!r.ok) {
        const body = (await r.json().catch(() => ({}))) as { error?: string };
        throw new Error(body.error ?? `Token fetch failed: ${r.status}`);
      }
      const { jwt } = (await r.json()) as { jwt: string };
      return jwt;
    });
    await ready;
  }

  async prepareSeeds(seeds: { id: string; imageUrl: string }[]): Promise<void> {
    await Promise.all(
      seeds.map(async ({ id, imageUrl }) => {
        if (this.seedB64.has(id)) return;
        try {
          this.seedB64.set(id, await fetchSeedAsBase64(imageUrl));
        } catch (e) {
          // Never block playback on one bad seed: its events degrade to
          // prompt-only (logged with the exact failing URL/status).
          console.warn(
            `[world] seed "${id}" fetch/encode failed: ${e instanceof Error ? e.message : e}`,
          );
        }
      }),
    );
    console.info(`[world] ${this.seedB64.size}/${seeds.length} seeds encoded (base64 inline)`);
  }

  /** Swap the reference image (inline base64 — the presigned-URL
   *  FileRef path is deliberately unused, it 404s on this deployment)
   *  and await the model's verdict. Returns true if accepted. */
  private async sendImage(seedId: string, b64: string): Promise<boolean> {
    await this.reactor.sendCommand("set_image", { image_b64: b64 });
    const verdict = await this.waitFor(
      (m) =>
        m.type === "image_accepted" ||
        (m.type === "command_error" && m.command === "set_image"),
      IMAGE_ACK_TIMEOUT_MS,
    );
    if (verdict?.type === "image_accepted") {
      this.currentSeedId = seedId;
      console.info(
        `[world] seed "${seedId}" accepted (${verdict.width}×${verdict.height})`,
      );
      return true;
    }
    console.warn(
      `[world] seed "${seedId}" NOT accepted (${verdict ? `rejected: ${verdict.reason}` : "no ack within timeout — command silently dropped?"})`,
    );
    return false;
  }

  private async setImageStrength(value: number): Promise<void> {
    if (value === this.currentImageStrength) return;
    this.currentImageStrength = value;
    await this.reactor.sendCommand("set_image_strength", { image_strength: value });
  }

  async applyScene({ seedId, prompt, transition }: WorldScene): Promise<void> {
    const now = Date.now();
    if (now - this.lastSendAt < MIN_SEND_INTERVAL_MS) return;
    this.lastSendAt = now;

    const b64 = this.seedB64.get(seedId);
    const seedChanged = seedId !== this.currentSeedId;

    if (!this.started) {
      // ── First scene, before start: image → ack → prompt → ack. ──
      // Sequential-with-acks is the race-free ordering here: nothing is
      // pending asynchronously (no upload), and `start` is only sent by
      // the caller after this resolves with conditions confirmed.
      if (b64) {
        const ok = await this.sendImage(seedId, b64);
        if (ok) await this.setImageStrength(IMAGE_STRENGTH);
      } else {
        console.warn(`[world] no encoded image for seed "${seedId}" — prompt-only start`);
      }
      await this.reactor.sendCommand("set_prompt", { prompt });
      const ready = await this.waitFor(
        (m) => m.type === "conditions_ready" || m.type === "prompt_accepted",
        PROMPT_ACK_TIMEOUT_MS,
      );
      if (ready?.type === "conditions_ready") {
        console.info(
          `[world] conditions_ready (has_image=${ready.has_image}, has_prompt=${ready.has_prompt})`,
        );
      } else if (!ready) {
        console.warn("[world] no prompt ack — start may be refused (watch for command_error)");
      }
      return;
    }

    // ── Mid-playback: prompts are already scheduled on the generation
    // clock (schedulePrompts); only seed image swaps go out live. ──
    if (seedChanged && b64) {
      // Cuts assert the new seed hard; a seed change on a morph eases
      // in with lower strength instead of snapping.
      await this.setImageStrength(
        transition === "cut" ? IMAGE_STRENGTH : IMAGE_STRENGTH_MORPH,
      );
      // Fire-and-forget on purpose: awaiting the ack here would stall
      // the rAF-driven executor; acceptance/rejection still logs via
      // the message handler.
      this.reactor.sendCommand("set_image", { image_b64: b64 }).catch((e) => {
        console.warn(`[world] set_image failed for "${seedId}"`, e);
      });
      this.currentSeedId = seedId;
    } else if (seedChanged) {
      console.warn(`[world] no encoded image for seed "${seedId}" — keeping previous seed`);
    }
  }

  async start(): Promise<void> {
    await this.reactor.sendCommand("start", {});
    const startedMsg = await this.waitFor(
      (m) =>
        m.type === "generation_started" ||
        (m.type === "command_error" && m.command === "start"),
      PROMPT_ACK_TIMEOUT_MS,
    );
    if (startedMsg?.type === "generation_started") {
      console.info(`[world] generation_started at chunk ${startedMsg.chunk_index}`);
    }
    this.started = true;
    // Watchdog: if no video track has arrived shortly after start,
    // generation almost certainly never began (look for a command_error
    // just above in the console).
    this.trackWatchdog = setTimeout(() => {
      if (this.gotVideoTrack || this.closed) return;
      const text =
        "No video track arrived within 20s of start — the model is " +
        "probably not generating (check for a rejected command in the console).";
      console.error(`[world] ${text}`);
      for (const cb of this.errorCbs) cb(text);
    }, 20_000);
  }

  async schedulePrompts(
    entries: { timestamp: number; prompt: string }[],
  ): Promise<void> {
    // Generation begins at chunk 0 when the song starts, so timestamp →
    // chunk index is a straight division. Prompts land model-side on
    // exact chunk boundaries — no rAF jitter, and if generation lags
    // real time the prompts stay in step with the VIDEO (the audio-
    // synced part of an event is its seed swap, sent by applyScene).
    for (const { timestamp, prompt } of entries) {
      const chunk = Math.max(1, Math.round(timestamp / CHUNK_SECONDS));
      await this.reactor.sendCommand("schedule_prompt", { chunk, prompt });
    }
    console.info(`[world] ${entries.length} prompts pre-scheduled on the chunk clock`);
  }

  driveCamera(): void {
    // Helios has no camera inputs — the world is a flat generated stream
    // and mouse-look is only a view transform (see WorldStage).
  }

  pulse(): void {
    // Helios drives beat activity via the timeline, not runtime pulses.
  }

  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    if (this.trackWatchdog) clearTimeout(this.trackWatchdog);
    try {
      await this.reactor.disconnect();
    } catch {
      // Teardown races (session already expired, tab closing) are fine.
    }
  }
}

/**
 * Lingbot World 2 session — a NAVIGABLE world model, wired to the same
 * WorldSession surface so the app orchestrator doesn't change.
 *
 * The fundamental difference from Helios: Lingbot CANNOT hot-swap a
 * reference image mid-stream ("changes during generation have no effect
 * until reset"). So the score is executed as MULTIPLE RUNS pasted into
 * one continuous-feeling experience:
 *   - same seed (morphs, prompt-led evolution) → set_prompt in-run,
 *     seamless (applies on the next chunk);
 *   - seed change (cuts) → reset → set_image(new) → set_prompt → start,
 *     i.e. a fresh run. That reconfigure has a visible gap; onSeam fires
 *     around it so the stage can freeze-cover it (wired in the UI).
 *
 * Seeds are uploaded via uploadFile (presigned-URL FileRef) — Lingbot
 * has NO base64 image form, and start REQUIRES an image. If uploadFile
 * 404s (the failure Helios dodged with inline base64), generation here
 * cannot start at all; prepareSeeds logs that loudly as the gate signal.
 */
class LingbotWorldSession implements WorldSession {
  readonly kind = "lingbot" as const;
  private model: LingbotWorld2Model;
  private trackCbs: ((track: MediaStreamTrack) => void)[] = [];
  private statusCbs: ((status: ReactorStatus) => void)[] = [];
  private errorCbs: ((message: string) => void)[] = [];
  private seamCbs: ((covering: boolean) => void)[] = [];
  private waiters: {
    match: (m: ModelMessage) => boolean;
    resolve: (m: ModelMessage) => void;
  }[] = [];
  private lastSendAt = 0;
  private closed = false;
  private started = false;
  private gotVideoTrack = false;
  private trackWatchdog: ReturnType<typeof setTimeout> | null = null;
  /** seedId → uploaded FileRef (reusable across set_image calls). */
  private seedRefs = new Map<string, FileRef>();
  private currentSeedId: string | null = null;
  /** Last [rx,ry,rz,tx,ty,tz] camera pose actually sent, for diffing — the
   *  pose persists per chunk, so we only resend when it changes. Reset to
   *  null after a cut (new run) to force a fresh pose push. */
  private lastPose: number[] | null = null;
  /** Base scene prompt currently driving generation (set by applyScene),
   *  restored after a beat pulse temporarily intensifies it. */
  private basePrompt = "";
  private lastPulseAt = 0;
  private pulseRevertTimer: ReturnType<typeof setTimeout> | null = null;

  constructor() {
    this.model = new LingbotWorld2Model();
    this.model.on("trackReceived", (name: string, track: MediaStreamTrack) => {
      console.info(`[lingbot] trackReceived: "${name}"`);
      if (name === "main_video") {
        this.gotVideoTrack = true;
        for (const cb of this.trackCbs) cb(track);
      }
    });
    this.model.on("statusChanged", (status: ReactorStatus) => {
      for (const cb of this.statusCbs) cb(status);
    });
    this.model.on("message", (raw: unknown) => {
      if (!raw || typeof raw !== "object") return;
      const env = raw as { type?: string; data?: Record<string, unknown> };
      const msg: ModelMessage =
        env.data && typeof env.data === "object"
          ? { ...env.data, type: env.type }
          : (raw as ModelMessage);
      this.waiters = this.waiters.filter((w) => {
        if (!w.match(msg)) return true;
        w.resolve(msg);
        return false;
      });
      if (msg.type === "command_error") {
        const text = `Model rejected "${msg.command ?? "?"}": ${msg.reason ?? "no reason given"}`;
        console.error(`[lingbot] ${text}`, msg);
        for (const cb of this.errorCbs) cb(text);
      } else if (msg.type === "chunk_complete") {
        console.debug(
          `[lingbot] chunk_complete #${msg.chunk_index} action=${msg.active_action} (${String(msg.active_prompt).slice(0, 40)}…)`,
        );
      } else if (msg.type && msg.type !== "state") {
        console.info(`[lingbot] model: ${msg.type}`, msg);
      }
    });
  }

  onTrack(cb: (track: MediaStreamTrack) => void) {
    this.trackCbs.push(cb);
  }
  onStatus(cb: (status: ReactorStatus) => void) {
    this.statusCbs.push(cb);
  }
  onError(cb: (message: string) => void) {
    this.errorCbs.push(cb);
  }
  /** Fires true when a cut's reset gap begins, false when the new run's
   *  frames are flowing — the stage uses this to freeze-cover the seam. */
  onSeam(cb: (covering: boolean) => void) {
    this.seamCbs.push(cb);
  }

  private waitFor(
    match: (m: ModelMessage) => boolean,
    timeoutMs: number,
  ): Promise<ModelMessage | null> {
    return new Promise((resolve) => {
      const waiter = { match, resolve: (m: ModelMessage) => resolve(m) };
      this.waiters.push(waiter);
      setTimeout(() => {
        this.waiters = this.waiters.filter((w) => w !== waiter);
        resolve(null);
      }, timeoutMs);
    });
  }

  async connect(): Promise<void> {
    const ready = new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(
        () => reject(new Error("Lingbot session timed out waiting for ready")),
        90_000,
      );
      this.model.on("statusChanged", (status: ReactorStatus) => {
        if (status === "ready") {
          clearTimeout(timeout);
          resolve();
        }
      });
      this.model.on("error", (err: { message?: string }) => {
        clearTimeout(timeout);
        reject(new Error(err?.message ?? "Lingbot connection error"));
      });
    });
    // Same account token as Helios — the JWT is model-agnostic.
    await this.model.connect(async () => {
      const r = await fetch("/api/reactor/token");
      if (!r.ok) {
        const body = (await r.json().catch(() => ({}))) as { error?: string };
        throw new Error(body.error ?? `Token fetch failed: ${r.status}`);
      }
      const { jwt } = (await r.json()) as { jwt: string };
      return jwt;
    });
    await ready;
  }

  /**
   * THE GATE: upload every seed via the presigned-URL protocol and cache
   * its FileRef. This is the exact path that 404'd under Helios; if it
   * fails here it is fatal (no base64 fallback, image is mandatory), so
   * every failure is logged prominently with the seed id.
   */
  async prepareSeeds(seeds: { id: string; imageUrl: string }[]): Promise<void> {
    await Promise.all(
      seeds.map(async ({ id, imageUrl }) => {
        if (this.seedRefs.has(id)) return;
        try {
          const res = await fetch(imageUrl);
          if (!res.ok) throw new Error(`GET ${imageUrl} → ${res.status}`);
          const blob = await res.blob();
          const ref = await this.model.uploadFile(blob, { name: id });
          this.seedRefs.set(id, ref);
          console.info(`[lingbot] seed "${id}" uploaded → FileRef ok`);
        } catch (e) {
          console.error(
            `[lingbot] seed "${id}" UPLOAD FAILED (fatal for lingbot — no base64 fallback): ${e instanceof Error ? e.message : e}`,
          );
        }
      }),
    );
    const ok = this.seedRefs.size;
    console.info(`[lingbot] ${ok}/${seeds.length} seeds uploaded`);
    if (ok === 0) {
      const text =
        "No seed image uploaded — Lingbot cannot start without an image " +
        "(uploadFile is failing; this is the presigned-URL 404). See console.";
      for (const cb of this.errorCbs) cb(text);
    }
  }

  /** set_image(FileRef) → await image_accepted. Returns true if accepted. */
  private async sendImage(seedId: string): Promise<boolean> {
    const ref = this.seedRefs.get(seedId);
    if (!ref) {
      console.warn(`[lingbot] no FileRef for seed "${seedId}"`);
      return false;
    }
    await this.model.setImage({ image: ref });
    const verdict = await this.waitFor(
      (m) =>
        m.type === "image_accepted" ||
        (m.type === "command_error" && m.command === "set_image"),
      IMAGE_ACK_TIMEOUT_MS,
    );
    if (verdict?.type === "image_accepted") {
      this.currentSeedId = seedId;
      console.info(`[lingbot] seed "${seedId}" accepted (${verdict.width}×${verdict.height})`);
      return true;
    }
    console.warn(
      `[lingbot] seed "${seedId}" NOT accepted (${verdict ? `rejected: ${verdict.reason}` : "no ack — dropped?"})`,
    );
    return false;
  }

  /**
   * STRICT, wire-level guarantee: no prompt ever reaches Lingbot without
   * the no-visible-character clause, regardless of what Nemotron wrote.
   * This is enforced HERE — not just as an instruction in the composer's
   * system prompt — because an LLM instruction can be missed on any
   * given event; concatenating the clause onto every literal string sent
   * over the wire cannot be. `basePrompt` stays the raw scene text (for
   * logging/reuse by pulse()); only the wire payload gets the suffix.
   */
  private setScenePrompt(rawPrompt: string): Promise<void> {
    this.basePrompt = rawPrompt;
    return this.model.setPrompt({
      prompt: rawPrompt + NO_CHARACTER_SUFFIX + STATIC_CAMERA_SUFFIX,
    });
  }

  async applyScene({ seedId, prompt }: WorldScene): Promise<void> {
    const now = Date.now();
    if (now - this.lastSendAt < MIN_SEND_INTERVAL_MS) return;
    this.lastSendAt = now;

    if (!this.started) {
      // First scene, before start: image → ack → prompt. Caller then
      // calls start(). Lingbot start REQUIRES both image and prompt.
      await this.sendImage(seedId);
      await this.setScenePrompt(prompt);
      const ready = await this.waitFor(
        (m) => m.type === "conditions_ready" || m.type === "prompt_accepted",
        PROMPT_ACK_TIMEOUT_MS,
      );
      if (ready?.type === "conditions_ready") {
        console.info(
          `[lingbot] conditions_ready (has_image=${ready.has_image}, has_prompt=${ready.has_prompt})`,
        );
      }
      return;
    }

    // Mid-playback. A seed change is the ONLY thing that needs a new run
    // (Lingbot can't hot-swap the image); same-seed events just move the
    // prompt within the current run.
    if (seedId !== this.currentSeedId) {
      if (!this.seedRefs.has(seedId)) {
        console.warn(`[lingbot] no FileRef for seed "${seedId}" — keeping current run`);
        await this.setScenePrompt(prompt);
        return;
      }
      // ── CUT: reset → new run. Cover the gap. ──
      for (const cb of this.seamCbs) cb(true);
      await this.model.reset();
      const ok = await this.sendImage(seedId);
      await this.setScenePrompt(prompt);
      if (ok) {
        await this.applyImmersionDefaults(); // reset() cleared it — re-arm
        await this.model.start();
        await this.waitFor(
          (m) =>
            m.type === "generation_started" ||
            (m.type === "command_error" && m.command === "start"),
          PROMPT_ACK_TIMEOUT_MS,
        );
        this.lastPose = null; // force a fresh pose push into the new run
      }
      for (const cb of this.seamCbs) cb(false);
    } else {
      // Same seed (morph / prompt-led evolution) — seamless in-run.
      await this.setScenePrompt(prompt);
    }
  }

  /**
   * Let the world DEVIATE from the static seed (Lingbot has no
   * image_strength knob). "manual" kv-cache reset stops the model from
   * auto-snapping its context back to the seed, so the scene evolves and
   * drifts within a section instead of looking like a frozen postcard;
   * we re-anchor deliberately only at cuts (new runs). attn_window "auto"
   * balances stability against responsiveness to the camera motion we
   * now drive. Re-applied after each new run since reset clears state.
   */
  private async applyImmersionDefaults(): Promise<void> {
    try {
      await this.model.setKvCacheReset({ mode: "manual" });
      await this.model.setAttnWindow({ attn_window: "auto" });
    } catch (e) {
      console.warn("[lingbot] immersion defaults failed", e);
    }
  }

  async start(): Promise<void> {
    await this.applyImmersionDefaults();
    await this.model.start();
    const startedMsg = await this.waitFor(
      (m) =>
        m.type === "generation_started" ||
        (m.type === "command_error" && m.command === "start"),
      PROMPT_ACK_TIMEOUT_MS,
    );
    if (startedMsg?.type === "generation_started") {
      console.info(`[lingbot] generation_started (${startedMsg.chunk_num} chunks)`);
    }
    this.started = true;
    this.trackWatchdog = setTimeout(() => {
      if (this.gotVideoTrack || this.closed) return;
      const text =
        "No video track within 20s of start — check for a rejected command " +
        "(most likely no image was uploaded; see the console).";
      console.error(`[lingbot] ${text}`);
      for (const cb of this.errorCbs) cb(text);
    }, 20_000);
  }

  async schedulePrompts(): Promise<void> {
    // Lingbot has no schedule_prompt command; prompts are driven live
    // per-event by applyScene (set_prompt applies on the next chunk).
    console.info("[lingbot] schedulePrompts: no-op (prompts driven live per event)");
  }

  /**
   * Downbeat activity in the GENERATED world: briefly append the pulse
   * suffix to the active prompt so the model surges on the bar, then
   * revert. set_prompt applies on the NEXT chunk, so this lands at bar
   * rate (the model can't react per beat — that's the overlay's job).
   * Throttled so dense downbeats don't stack pulses on top of reverts.
   */
  pulse(): void {
    if (!this.started || this.closed || !this.basePrompt) return;
    const now = Date.now();
    if (now - this.lastPulseAt < BEAT_PULSE_MIN_INTERVAL_MS) return;
    this.lastPulseAt = now;

    // Bypasses setScenePrompt (must not overwrite this.basePrompt — the
    // pulse is a transient overlay on it) but still carries the strict
    // no-character AND static-camera clauses on the wire, same as every
    // other prompt (the surge is scene energy, never a camera move).
    this.model
      .setPrompt({
        prompt:
          this.basePrompt +
          BEAT_PULSE_SUFFIX +
          NO_CHARACTER_SUFFIX +
          STATIC_CAMERA_SUFFIX,
      })
      .catch((e) => console.warn("[lingbot] pulse failed", e));
    if (this.pulseRevertTimer) clearTimeout(this.pulseRevertTimer);
    this.pulseRevertTimer = setTimeout(() => {
      if (this.closed) return;
      // Revert to whatever the current base is (a scene change may have
      // updated it in the meantime).
      this.model
        .setPrompt({
          prompt: this.basePrompt + NO_CHARACTER_SUFFIX + STATIC_CAMERA_SUFFIX,
        })
        .catch(() => {});
    }, BEAT_PULSE_HOLD_MS);
  }

  /**
   * Push an audio/user camera intent to the model. Diffs against the
   * last sent intent and only emits the setters that changed (rotation
   * bucketed to nearest 2°), throttled so sub-chunk churn never floods
   * the data channel. Idempotent to call every playback tick.
   */
  driveCamera(intent: CameraIntent): void {
    if (!this.started || this.closed) return;

    // Explicit per-chunk camera pose [rx, ry, rz, tx, ty, tz]. Rotation is
    // RELATIVE to the current orientation, so an ALL-ZERO pose holds the
    // camera perfectly still and OVERRIDES the model's own look-drift — this
    // is what makes autonomous rotation impossible. A held key applies a
    // fixed rotation/translation each chunk (a turn / a walk); releasing
    // returns to all-zero = frozen. The pose persists across chunks, so we
    // only resend when it changes.
    //
    // Sign convention (flip a term here if a direction comes out inverted):
    //   ry (yaw)   +ve = look RIGHT   ← lookH = +1 (ArrowRight)
    //   rx (pitch) +ve = look UP      ← lookV = +1 (ArrowUp)
    //   tz         +ve = move FORWARD ← lon   = +1 (W)
    //   tx         +ve = strafe RIGHT ← lat   = +1 (D)
    const rx = intent.lookV * POSE_PITCH_RAD;
    const ry = intent.lookH * POSE_YAW_RAD;
    const tx = intent.lat * POSE_MOVE;
    const tz = intent.lon * POSE_MOVE;
    const pose = [rx, ry, 0, tx, 0, tz];

    const prev = this.lastPose;
    if (prev && prev.every((v, i) => v === pose[i])) return; // unchanged
    this.lastPose = pose;
    void this.model
      .setCameraPose({ camera_pose: pose })
      .catch((e) => console.warn("[lingbot] camera pose failed", e));
  }

  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    if (this.trackWatchdog) clearTimeout(this.trackWatchdog);
    if (this.pulseRevertTimer) clearTimeout(this.pulseRevertTimer);
    try {
      await this.model.disconnect();
    } catch {
      // Teardown races are fine.
    }
  }
}

/**
 * Mock session: the exact same call sequence the real session receives,
 * but nothing leaves the browser. The procedural <MockWorldCanvas>
 * renders the world (with the active event's seed image as backdrop)
 * from the same score data instead of a video track. Commands are
 * logged so the score execution is observable.
 */
class MockWorldSession implements WorldSession {
  readonly kind = "mock" as const;
  private statusCbs: ((status: ReactorStatus) => void)[] = [];
  private started = false;
  private currentSeedId: string | null = null;

  onTrack(): void {
    // No video track in mock mode — MockWorldCanvas renders instead.
  }
  onError(): void {
    // Nothing to reject in mock mode.
  }
  onStatus(cb: (status: ReactorStatus) => void) {
    this.statusCbs.push(cb);
  }

  async connect(): Promise<void> {
    // Simulate a short connect so the UI's connecting state is visible.
    for (const cb of this.statusCbs) cb("connecting");
    await new Promise((r) => setTimeout(r, 400));
    for (const cb of this.statusCbs) cb("ready");
  }

  async prepareSeeds(seeds: { id: string; imageUrl: string }[]): Promise<void> {
    console.info(
      `[mock world] prepareSeeds (${seeds.length}):`,
      seeds.map((s) => s.id).join(", "),
    );
  }

  async applyScene({ seedId, prompt, transition }: WorldScene): Promise<void> {
    const swap = seedId !== this.currentSeedId;
    this.currentSeedId = seedId;
    console.info(
      `[mock world] ${this.started ? (swap ? `set_image (${transition})` : "prompt-led") : "commit image+prompt"}` +
        ` seed=${seedId}: ${prompt.slice(0, 60)}…`,
    );
  }

  async start(): Promise<void> {
    this.started = true;
    console.info("[mock world] start");
  }

  driveCamera(): void {
    // Mock canvas has no navigable camera — mouse-look handles the view.
  }

  pulse(): void {
    // No live generation to pulse in mock mode.
  }

  async schedulePrompts(
    entries: { timestamp: number; prompt: string }[],
  ): Promise<void> {
    console.info(`[mock world] schedule_prompt ×${entries.length} (chunk clock)`);
  }

  async close(): Promise<void> {
    for (const cb of this.statusCbs) cb("disconnected");
  }
}

export function createWorldSession(kind: WorldSessionKind): WorldSession {
  switch (kind) {
    case "lingbot":
      return new LingbotWorldSession();
    case "reactor":
      return new ReactorWorldSession();
    default:
      return new MockWorldSession();
  }
}
