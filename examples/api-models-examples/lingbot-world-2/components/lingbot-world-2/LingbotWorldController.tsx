"use client";

import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import {
  useLingbotWorld2,
  useLingbotWorld2Message,
  type FileRef,
  type LingbotWorld2Message,
} from "@reactor-models/lingbot-world-2";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import {
  EXAMPLES,
  STRUCTURED_EXAMPLES,
  composePrompt,
  cloneScene,
  emptyScene,
  scenesEqual,
  type StructuredExample,
  type StructuredScene,
} from "@/lib/lingbot-world-prompts";

// Sentinel id used in the overrides map for the user's custom
// layered scene. Custom has no pristine constant to fall back to; an
// "edited" state simply means a non-empty override exists.
const CUSTOM_SCENE_ID = "__custom__";
import { LayeredSceneEditor } from "@/components/lingbot-world-2/LayeredSceneEditor";
import { LivePromptInspector } from "@/components/lingbot-world-2/LivePromptInspector";

type MoveL = "idle" | "forward" | "back";
type MoveLat = "idle" | "strafe_left" | "strafe_right";
type LookH = "idle" | "left" | "right";
type LookV = "idle" | "up" | "down";

// Per-example user overrides. Each example id maps to the user's edited
// StructuredScene. Clicking an example loads its override (if any) rather
// than the pristine constant, so edits survive across re-clicks and
// reloads. "Reset to example" inside the editor deletes the override.
const OVERRIDES_STORAGE_KEY = "lingbot-world-2:overrides:v1";
const MAX_EVENTS = 9;

// Validate a parsed object is a StructuredScene shape — used when
// hydrating from localStorage.
function isStructuredScene(v: unknown): v is StructuredScene {
  if (!v || typeof v !== "object") return false;
  const s = v as Record<string, unknown>;
  return (
    !!s.base &&
    typeof s.base === "object" &&
    !!s.camera &&
    typeof s.camera === "object" &&
    !!s.movement &&
    typeof s.movement === "object" &&
    Array.isArray(s.events)
  );
}

const KEY_TO_MOVE_L: Record<string, Exclude<MoveL, "idle">> = {
  w: "forward",
  W: "forward",
  s: "back",
  S: "back",
};
const KEY_TO_MOVE_LAT: Record<string, Exclude<MoveLat, "idle">> = {
  a: "strafe_left",
  A: "strafe_left",
  d: "strafe_right",
  D: "strafe_right",
};

const KEY_TO_LOOK_H: Record<string, Exclude<LookH, "idle">> = {
  ArrowLeft: "left",
  ArrowRight: "right",
};

const KEY_TO_LOOK_V: Record<string, Exclude<LookV, "idle">> = {
  ArrowUp: "up",
  ArrowDown: "down",
};

// ---- Native camera-pose layer (set_camera_pose) ----
//
// The backend's camera_pose input takes a flat list of per-frame motion
// DELTAS: [rx, ry, rz, tx, ty, tz] per frame (Euler-radian rotation +
// translation, camera-local). Rotation OVERRIDES the arrow keys, translation
// ADDS to WASD. We send one delta per chunk (6 floats; the backend repeats it
// across its chunk_size) — no absolute pose, no matrices, no client state to
// maintain. The backend sanitizes any payload, so this is hard to misuse.
const MOUSE_SENS_DEFAULT = 0.0003; // per-frame radians of look per pixel (user-adjustable)
const MOUSE_SENS_MIN = 0.00005;
const MOUSE_SENS_MAX = 0.0012;
const MOUSE_MAX_ROT = 0.2; // hard ceiling on per-chunk |yaw|/|pitch| (the HUD
// outer ring) — a fast fling can't over-rotate
const ROLL_SPEED = 0.08; // per-frame radians of roll (Q/E) — matched to the
// keyboard arrow rate (~4.6°/frame) so it's visible
// Arrow-key look is routed through the SAME camera_pose yaw/pitch as mouse-look
// (rather than the backend's separate discrete look_horizontal/look_vertical
// state), so it's just a steady, fixed-rate version of mouse movement — it
// stacks with the mouse and drives everything mouse-driven yaw does, incl. orbit.
const ARROW_LOOK_SPEED = ROLL_SPEED; // per-frame radians of yaw/pitch while an arrow is held
const JUMP_SPEED = 1.0; // per-frame up-translation while held (magnitude
// is washed out by per-chunk normalization)
const JOY_SPEED = 1.0; // per-frame translation magnitude at full joystick deflect
// Camera-local "up" sign for Jump. Empirically verified against the backend:
// characters dove DOWN with +1, so up is -1 in this model's local-Y convention.
const JUMP_UP_SIGN = -1;

// ---- Orbit mode (Phase 1: horizontal) ----
// Orbit couples the YAW (mouse OR arrow-key look — both land in the same `ry`,
// see ARROW_LOOK_SPEED) with a proportional camera-local strafe (+ a slight
// forward dolly) so the point R ahead stays centered while the camera circles it,
// instead of rotating about its own optical center. For a per-frame yaw θ:
//   tx += -R·sin θ   (strafe to slide along the arc)
//   tz += R·(1-cos θ) (the tiny forward sagitta)
// R is the coupling RATIO (strafe per unit yaw) = the orbit radius in the model's
// local units; R = 0 reproduces today's rotate-in-place. Because the backend
// max-norms each chunk (only the WITHIN-chunk tx:ry ratio survives, absolute scale
// is washed out), R is a RELATIVE "arc width", not metres — and the whole scheme
// only works if rotation + translation are normalized TOGETHER (verify live: if the
// subject spirals out instead of staying centered, they're normed separately).
// Signs assume +X = right, +Z = forward, positive ry = yaw right (see mouse onMove).
const ORBIT_RADIUS_DEFAULT = 6;
const ORBIT_RADIUS_STEP = 0.5; // per-click nudge from the up/down stepper buttons

// Jump has three selectable modes (the "Jump" switch by the pad):
//   "hold"   — hold to translate straight UP for as long as held (no descent).
//   "prompt" — jump toggles ONLY the scene's jumpPrompt; no camera_pose at all.
//   "charge" — hold to charge a meter (steps through discrete levels, no motion yet);
//              release to fire that level's per-latent arc. camera_pose is
//              controlled at LATENT granularity (the backend takes one delta per
//              latent, 3 latents per chunk), so each level's arc is a hand-editable
//              per-latent up/down/still plan (set in the grid popup). The backend
//              max-norms each chunk together (one max over its 3 latents), which
//              keeps the WITHIN-chunk shape (relative magnitude survives) but
//              discards cross-chunk absolute scale — so the arc is expressed per
//              latent, and its size is its latent count.
type JumpMode = "hold" | "prompt" | "charge";

// Crouch modes (the "Crouch" switch), mirroring Jump:
//   "hold"   — C held → sustained straight-DOWN translation for as long as held
//              (mirror of jump "hold"; the way to walk vertically downward).
//   "prompt" — C held → inject the scene's crouchPrompt, no camera_pose.
//   "camera" — C press → a one-shot downward camera dip (see below).
// Both the jump and crouch sentences are per-scene, editable in the scene editor
// (the "vertical" tab) — never hardcoded here.
type CrouchMode = "hold" | "prompt" | "camera";

// DiT self-attention window override (the backend `set_attn_window` event):
//   "auto"  — motion-based still-window trigger (default): small window when the
//             camera is still, full window when moving.
//   "small" — force the still (small) window always.
//   "large" — force the moving (large/full) window always.
type AttnWindow = "auto" | "small" | "large";

// KV-cache / RoPE reset mode (backend `set_kv_cache_reset`):
//   "off"    — no reset at all.
//   "auto"   — periodic window reset (~every 88 latent frames) + manual trigger.
//   "manual" — no periodic reset; only the manual `trigger_kv_cache_reset` fires.
type KvResetMode = "off" | "auto" | "manual";

// One-shot crouch dip (camera mode): applied to the FIRST chunk after C press,
// additive to WASD/forward (the backend sums the action + camera_pose), then the
// pose reverts so the camera holds its new, slightly-lower height. CROUCH_DIP tunes
// the dip depth vs forward when moving.
const CROUCH_DIP = 0.5;
// Sustained per-frame DOWN translation while C is held in crouch "hold" mode
// (mirror of JUMP_SPEED for jump "hold"). Magnitude is normalized per-chunk on the
// backend, so this mainly sets the down:forward ratio when walking down while W.
const CROUCH_SPEED = 1.0;
const CROUCH_PATTERN_STORAGE = "lingbot-world-2:crouch-patterns:v1";
// Crouch is a press+release action. Two hand-editable one-chunk patterns (grid
// popup, CHUNK_LATENTS cells each, +1 up / 0 still / -1 down): "press" fires on
// C-down (a downward dip), "release" fires on C-up (the reverse, standing
// back up). Defaults: down-then-still on press, up-then-still on release, so a
// press+release nets back to the original height.
type CrouchPhase = "press" | "release";
type CrouchPatterns = { press: number[]; release: number[] };
function defaultCrouchPatterns(): CrouchPatterns {
  const still = Array<number>(CHUNK_LATENTS - 1).fill(0);
  return { press: [-1, ...still], release: [1, ...still] };
}
function isCrouchLatents(v: unknown): v is number[] {
  return (
    Array.isArray(v) &&
    v.length === CHUNK_LATENTS &&
    v.every((x) => x === 1 || x === 0 || x === -1)
  );
}
function isValidCrouchPatterns(v: unknown): v is CrouchPatterns {
  const o = v as CrouchPatterns | null;
  return (
    !!o &&
    typeof o === "object" &&
    isCrouchLatents(o.press) &&
    isCrouchLatents(o.release)
  );
}

// Latents per chunk on the backend (lingbot-v2 config.yml chunk_size = 3). The
// client sends CHUNK_LATENTS deltas per chunk (18 floats); the backend uses them
// one-to-one (k == target_len), so each latent is steered independently.
const CHUNK_LATENTS = 3;
// Charge is DISCRETE to match the backend: NUM_CHARGE_LEVELS stages, level k →
// k chunks (k * CHUNK_LATENTS latents). The meter STEPS through levels (dwelling
// LEVEL_DWELL_MS on each, bouncing up/down) instead of filling continuously, so
// what you see is exactly what you fire. Release at level k → a k-chunk arc.
const NUM_CHARGE_LEVELS = 3; // 1, 2, or 3 chunks
const LEVEL_DWELL_MS = 400; // time held on each level before stepping
const CHARGE_PATTERNS_STORAGE = "lingbot-world-2:charge-patterns:v1";

// Each charge level's arc is a hand-editable per-latent plan (edited in the
// per-level grid popup): an array of length level*CHUNK_LATENTS where each latent
// is +1 up / -1 down / 0 still (a hold/pause). Defaults are SYMMETRIC (equal up
// and down) so the character returns to its launch height:
//   L1 (1 chunk):  up · down            -> [1, 0, -1]
//   L2 (2 chunks): up up null | down down null -> [1,1,0, -1,-1,0]
//   L3 (3 chunks): up×4 · down×4        -> [1,1,1,1, 0, -1,-1,-1,-1]
function defaultChargePattern(level: number): number[] {
  if (level === 2) return [1, 1, 0, -1, -1, 0]; // symmetric per-chunk (up up null; down down null)
  const L = level * CHUNK_LATENTS;
  const still = 1; // one pause latent at the peak
  const up = Math.floor((L - still) / 2); // odd L → symmetric up == down
  const down = L - still - up;
  return [
    ...Array<number>(up).fill(1),
    ...Array<number>(still).fill(0),
    ...Array<number>(down).fill(-1),
  ];
}
function defaultChargePatterns(): number[][] {
  return Array.from({ length: NUM_CHARGE_LEVELS }, (_, i) =>
    defaultChargePattern(i + 1),
  );
}
// Guard against a stale/garbage localStorage payload.
function isValidChargePatterns(v: unknown): v is number[][] {
  return (
    Array.isArray(v) &&
    v.length === NUM_CHARGE_LEVELS &&
    v.every(
      (p, i) =>
        Array.isArray(p) &&
        p.length === (i + 1) * CHUNK_LATENTS &&
        p.every((x) => x === 1 || x === 0 || x === -1),
    )
  );
}

function keyToHoldSlot(key: string): number | undefined {
  if (key.length !== 1) return undefined;
  const code = key.charCodeAt(0);
  if (code >= 49 && code <= 57) return code - 49;
  return undefined;
}

function PadButton({
  label,
  pressed,
  disabled,
  onPress,
  onRelease,
  className,
}: {
  label: React.ReactNode;
  pressed: boolean;
  disabled?: boolean;
  onPress: () => void;
  onRelease: () => void;
  className?: string;
}) {
  const handlers = disabled
    ? {}
    : {
        onPointerDown: (e: React.PointerEvent) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          onPress();
        },
        onPointerUp: onRelease,
        onPointerCancel: onRelease,
        onPointerLeave: (e: React.PointerEvent) => {
          if (e.buttons !== 0) onRelease();
        },
      };

  return (
    <button
      type="button"
      disabled={disabled}
      {...handlers}
      className={cn(
        "h-10 w-10 rounded border font-mono text-xs select-none transition-all",
        "disabled:opacity-30 disabled:cursor-not-allowed",
        pressed
          ? "bg-amber-300/20 border-amber-300/60 text-amber-200 scale-95"
          : "bg-white/5 border-white/15 text-white/80 hover:bg-white/10 active:scale-95",
        className,
      )}
    >
      {label}
    </button>
  );
}

// Small hold-to-activate button (roll / jump / crouch) — fires onDown while
// held, onUp on release. Sits beside the WASD pad.
function HoldBtn({
  label,
  lit,
  disabled,
  title,
  onDown,
  onUp,
  className,
}: {
  label: React.ReactNode;
  lit: boolean;
  disabled?: boolean;
  title?: string;
  onDown: () => void;
  onUp: () => void;
  className?: string;
}) {
  const handlers = disabled
    ? {}
    : {
        onPointerDown: (e: React.PointerEvent) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          onDown();
        },
        onPointerUp: onUp,
        onPointerCancel: onUp,
        onPointerLeave: (e: React.PointerEvent) => {
          if (e.buttons !== 0) onUp();
        },
      };
  return (
    <button
      type="button"
      disabled={disabled}
      title={title}
      {...handlers}
      className={cn(
        "h-7 w-11 rounded border font-mono text-[10px] select-none transition-all",
        "disabled:opacity-30 disabled:cursor-not-allowed",
        lit
          ? "bg-amber-300/20 border-amber-300/60 text-amber-200 scale-95"
          : "bg-white/5 border-white/15 text-white/80 hover:bg-white/10 active:scale-95",
        className,
      )}
    >
      {label}
    </button>
  );
}

function HoldChip({
  slot,
  name,
  empty,
  pressed,
  disabled,
  onPress,
  onRelease,
}: {
  slot: number;
  name: string;
  empty: boolean;
  pressed: boolean;
  disabled: boolean;
  onPress: () => void;
  onRelease: () => void;
}) {
  const handlers =
    disabled || empty
      ? {}
      : {
          onPointerDown: (e: React.PointerEvent) => {
            e.currentTarget.setPointerCapture(e.pointerId);
            onPress();
          },
          onPointerUp: onRelease,
          onPointerCancel: onRelease,
          onPointerLeave: (e: React.PointerEvent) => {
            if (e.buttons !== 0) onRelease();
          },
        };
  const displayName = name.trim() || `event ${slot + 1}`;
  return (
    <div
      {...handlers}
      title={empty ? `Slot ${slot + 1} is empty` : displayName}
      aria-disabled={disabled || empty}
      className={cn(
        "group flex items-center gap-1.5 rounded-full border px-2.5 py-1 font-mono text-[10px] transition-colors text-left max-w-full select-none",
        (disabled || empty) && "opacity-40 cursor-not-allowed",
        !disabled && !empty && "cursor-pointer",
        pressed
          ? "border-amber-300/80 bg-amber-300/25 text-amber-100"
          : "border-white/15 bg-white/5 text-white/70 hover:bg-white/10 hover:border-white/25",
      )}
    >
      <span
        className={cn(
          "inline-flex h-4 min-w-4 items-center justify-center rounded border px-0.5 text-[9px] font-bold",
          pressed
            ? "border-amber-300/80 bg-amber-300/30 text-amber-100"
            : "border-white/25 bg-white/10 text-white/80",
        )}
      >
        {slot + 1}
      </span>
      <span className="truncate">
        {empty ? <em className="text-white/30">empty</em> : displayName}
      </span>
    </div>
  );
}

export function LingbotWorldController({ className }: { className?: string }) {
  // Typed LingBot World 2 surface. The setter methods (lw2.setPrompt, …) are
  // per-render wrappers around the store's stable `sendCommand`, so callbacks
  // keep `sendCommand` as their dependency anchor — a stale `lw2` closure
  // still drives the same underlying store. Every command the app sends is
  // declared in the published schema, so the raw `sendCommand` escape hatch
  // is no longer used directly. Commands with a declared reply resolve with
  // it (`setPrompt` → `prompt_accepted`, `setImage` → `image_accepted`,
  // `pause`/`resume`/`reset` → `generation_*`); the rest resolve `undefined`
  // once the model's handler has run. None of them reject — failures land
  // on the `command_error` broadcast (→ toast) and `lastError`.
  const lw2 = useLingbotWorld2();
  const { status, sendCommand, uploadFile } = lw2;

  const isReady = status === "ready";

  const [sentImagePreview, setSentImagePreview] = useState<string | null>(null);
  const [imageInfo, setImageInfo] = useState<{ w: number; h: number } | null>(
    null,
  );
  const [pendingImage, setPendingImage] = useState<{
    file: File;
    previewUrl: string;
    label: string;
    presetSrc?: string;
  } | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [loadingExampleId, setLoadingExampleId] = useState<string | null>(null);

  const [hasPrompt, setHasPrompt] = useState(false);
  const [hasImage, setHasImage] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [chunkIndex, setChunkIndex] = useState(0);
  const [chunkNum, setChunkNum] = useState(0);
  const [activeAction, setActiveAction] = useState<string>("still");
  const [tspSize, setTspSize] = useState<number | null>(null);
  const [errorToast, setErrorToast] = useState<string | null>(null);

  const [moveL, setMoveL] = useState<MoveL>("idle");
  const [moveLat, setMoveLat] = useState<MoveLat>("idle");
  const [lookH, setLookH] = useState<LookH>("idle");
  const [lookV, setLookV] = useState<LookV>("idle");
  const [cameraPoseActive, setCameraPoseActive] = useState(false);

  // --- Native camera-pose layer (mouse-look + jump) ---
  const [mouseLook, setMouseLook] = useState(false); // pointer-lock engaged
  const [vertDir, setVertState] = useState(0); // vertical: -1 crouch (C). Jump no longer uses this.
  const [rollDir, setRollDir] = useState(0); // roll: -1 (Q) / 0 / +1 (E)
  const [jumpMode, setJumpMode] = useState<JumpMode>("charge"); // "Jump" switch: hold / prompt / charge
  const [crouchMode, setCrouchMode] = useState<CrouchMode>("camera"); // "Crouch" switch: hold / prompt / camera
  const [jumpLit, setJumpLit] = useState(false); // Space button highlight (held or arc in flight)
  const [charging, setCharging] = useState(false); // charge mode: meter stepping while held
  const [chargeLevel, setChargeLevel] = useState(0); // discrete level 0..NUM (0 = empty)
  const [verticalPrompt, setVerticalPrompt] = useState(""); // live jump/crouch sentence (for the inspector)
  // Hand-editable per-latent arc for each charge level (grid popup). Persisted.
  const [chargePatterns, setChargePatterns] = useState<number[][]>(
    defaultChargePatterns,
  );
  const [editingLevel, setEditingLevel] = useState<number | null>(null); // which level's grid is open
  // Hand-editable one-chunk crouch dip patterns (press + release; grid popup). Persisted.
  const [crouchPatterns, setCrouchPatterns] = useState<CrouchPatterns>(
    defaultCrouchPatterns,
  );
  const [editingCrouch, setEditingCrouch] = useState(false); // crouch grid popup open?
  const [mouseSens, setMouseSens] = useState(MOUSE_SENS_DEFAULT); // look sensitivity
  const [joy, setJoy] = useState({ x: 0, y: 0 }); // joystick knob (normalized −1..1)
  // Orbit: mouse-look circles a point R ahead instead of rotating in place. R is
  // the SOLE control — 0 = normal rotate-in-place (identical to no orbit), >0 =
  // orbit (bigger = wider/farther). No separate on/off flag: R=0 already IS "off".
  const [orbitRadius, setOrbitRadius] = useState(0);
  // DiT self-attention window override (backend `set_attn_window`). Default
  // "auto" mirrors the backend default (motion-based still-window trigger).
  const [attnWindow, setAttnWindow] = useState<AttnWindow>("auto");
  // KV-cache / RoPE reset mode (backend `set_kv_cache_reset`). Default "auto"
  // mirrors the shipped config default (kv_cache_reset_enable: true -> "auto").
  const [kvCacheResetMode, setKvCacheResetMode] = useState<KvResetMode>("auto");
  // Jump/crouch sentences live on the active scene (scene.jumpPrompt /
  // scene.crouchPrompt), edited in the scene editor — no outer state here.
  // Live mouse-signal HUD: a decaying recent-motion vector (screen px).
  const [mouseViz, setMouseViz] = useState({ x: 0, y: 0 });
  // Mouse movement accumulated (in pixels) since the last per-chunk send.
  // Converted to a per-frame rotation delta at send time, then zeroed.
  const pendingDYawRef = useRef(0); // += movementX (mouse right → yaw right)
  const pendingDPitchRef = useRef(0); // += movementY (mouse up → pitch up)
  const mouseVizRef = useRef({ x: 0, y: 0 }); // decaying viz vector (rAF-driven)
  const mouseLookRef = useRef(false);
  const vertDirRef = useRef(0); // -1 crouch / 0  (jump is tracked separately below)
  const rollDirRef = useRef(0); // -1 / 0 / +1
  // --- Jump (decoupled from vertDir so it can be prompt-only or camera) ---
  const jumpModeRef = useRef<JumpMode>("charge");
  useEffect(() => {
    jumpModeRef.current = jumpMode;
  }, [jumpMode]);
  const crouchModeRef = useRef<CrouchMode>("camera");
  useEffect(() => {
    crouchModeRef.current = crouchMode;
  }, [crouchMode]);
  // Camera mode: one-shot dips queued for the next chunk (press = down, release = up).
  const crouchPressDipRef = useRef(false);
  const crouchReleaseDipRef = useRef(false);
  const crouchPatternsRef = useRef<CrouchPatterns>(crouchPatterns);
  useEffect(() => {
    crouchPatternsRef.current = crouchPatterns;
  }, [crouchPatterns]);
  const jumpHeldRef = useRef(false); // is the jump key/button physically held?
  // Charge-arc state (a per-latent vertical-intent plan, consumed CHUNK_LATENTS
  // at a time). arc[i] ∈ {+1 up, 0 still, -1 down}; empty = no arc in flight.
  const jumpArcRef = useRef<number[]>([]);
  const jumpArcPosRef = useRef(0); // index of the current chunk's first latent
  const chargeLevelRef = useRef(0); // live discrete level 1..NUM while charging
  const chargeLevelDirRef = useRef(1); // meter step direction (+1/-1)
  const chargePatternsRef = useRef<number[][]>(chargePatterns);
  useEffect(() => {
    chargePatternsRef.current = chargePatterns;
  }, [chargePatterns]);
  // Restore saved per-level patterns (validated) so edits survive reloads.
  useEffect(() => {
    try {
      const saved = localStorage.getItem(CHARGE_PATTERNS_STORAGE);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (isValidChargePatterns(parsed)) setChargePatterns(parsed);
      }
      const savedCrouch = localStorage.getItem(CROUCH_PATTERN_STORAGE);
      if (savedCrouch) {
        const parsed = JSON.parse(savedCrouch);
        if (isValidCrouchPatterns(parsed)) setCrouchPatterns(parsed);
      }
    } catch {
      /* localStorage unavailable / bad JSON */
    }
  }, []);
  const mouseSensRef = useRef(MOUSE_SENS_DEFAULT);
  useEffect(() => {
    mouseSensRef.current = mouseSens;
  }, [mouseSens]);
  // Orbit: mirror to a ref so sendCameraPoseChunk reads the latest without re-subscribing.
  const orbitRadiusRef = useRef(0);
  useEffect(() => {
    orbitRadiusRef.current = orbitRadius;
  }, [orbitRadius]);
  // Last non-zero radius, so the O key can mute (→0) and un-mute back to it without losing the setting.
  const lastOrbitRadiusRef = useRef(ORBIT_RADIUS_DEFAULT);
  useEffect(() => {
    if (orbitRadius > 0) lastOrbitRadiusRef.current = orbitRadius;
  }, [orbitRadius]);
  const joyRef = useRef({ x: 0, y: 0 }); // normalized joystick vector (−1..1)
  const joyAreaRef = useRef<HTMLDivElement>(null);
  const poseSentActiveRef = useRef(false); // did we last send a non-empty pose?
  // Ref indirection so the message handler (defined above the callback) can
  // invoke the latest sendCameraPoseChunk without a declaration-order issue.
  const sendCameraPoseChunkRef = useRef<() => void>(() => {});

  const [rotationSpeed, setRotationSpeed] = useState(5.0);
  const [seed, setSeed] = useState(42);
  const [advancedOpen, setAdvancedOpen] = useState(false);

  // The active layered scene = the scene currently driving set_prompt.
  // Mirrors overrides[activeExampleId] (or the pristine example constant)
  // and lives separately because the user may pre-edit a different
  // example while one is running.
  const [scene, setScene] = useState<StructuredScene | null>(null);
  const [activeExampleId, setActiveExampleId] = useState<string | null>(null);

  // Per-example user edits. Reading the override during applyExample is
  // what makes edits survive re-clicks; clicking an example without an
  // override gets the pristine constant.
  const [overrides, setOverrides] = useState<Record<string, StructuredScene>>(
    {},
  );
  const [overridesLoaded, setOverridesLoaded] = useState(false);

  // Which example the editor modal is open for. Independent of
  // activeExampleId so the user can pre-edit example B while example A
  // is generating in the background.
  const [editingExampleId, setEditingExampleId] = useState<string | null>(null);
  const [heldSlots, setHeldSlots] = useState<number[]>([]);

  // Live read-only inspector — shown alongside the running generation
  // so the user can audit the current composed prompt and per-layer
  // contributions without leaving the running session.
  const [inspectorOpen, setInspectorOpen] = useState(false);

  // Restore overrides from localStorage on mount. Prune any whose
  // content is byte-equal to the pristine constant — those can happen
  // if a default ships with text the user had previously typed by hand,
  // and they'd otherwise leave a misleading "edited" badge on a card
  // whose effective prompt is identical to default.
  useEffect(() => {
    try {
      const raw = localStorage.getItem(OVERRIDES_STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw) as unknown;
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
          const out: Record<string, StructuredScene> = {};
          for (const [id, val] of Object.entries(
            parsed as Record<string, unknown>,
          )) {
            if (!isStructuredScene(val)) continue;
            const pristine = STRUCTURED_EXAMPLES[id]?.scene;
            if (pristine && scenesEqual(val, pristine)) continue;
            out[id] = val;
          }
          setOverrides(out);
        }
      }
    } catch {}
    setOverridesLoaded(true);
  }, []);

  useEffect(() => {
    if (!overridesLoaded) return;
    try {
      if (Object.keys(overrides).length > 0) {
        localStorage.setItem(OVERRIDES_STORAGE_KEY, JSON.stringify(overrides));
      } else {
        localStorage.removeItem(OVERRIDES_STORAGE_KEY);
      }
    } catch {}
  }, [overrides, overridesLoaded]);

  const overridesRef = useRef<Record<string, StructuredScene>>({});
  useEffect(() => {
    overridesRef.current = overrides;
  }, [overrides]);

  // Returns the user-edited scene for an example if one exists, else a
  // freshly cloned copy of the pristine constant. Always returns a fresh
  // clone so the caller can mutate it without touching the source. For
  // CUSTOM_SCENE_ID there is no pristine constant — only the override.
  const effectiveSceneFor = useCallback(
    (id: string): StructuredScene | null => {
      const override = overridesRef.current[id];
      if (override) return cloneScene(override);
      if (id === CUSTOM_SCENE_ID) return null;
      const structured = STRUCTURED_EXAMPLES[id];
      return structured ? cloneScene(structured.scene) : null;
    },
    [],
  );

  const moveLStackRef = useRef<Array<Exclude<MoveL, "idle">>>([]);
  const moveLatStackRef = useRef<Array<Exclude<MoveLat, "idle">>>([]);
  const lastSentMoveLRef = useRef<MoveL>("idle");
  const lastSentMoveLatRef = useRef<MoveLat>("idle");
  const lookHDirRef = useRef(0); // -1 left / 0 / +1 right (feeds camera_pose yaw)
  const lookVDirRef = useRef(0); // -1 down / 0 / +1 up (feeds camera_pose pitch)
  const fileInputRef = useRef<HTMLInputElement>(null);
  const isReadyRef = useRef(isReady);
  useEffect(() => {
    isReadyRef.current = isReady;
  }, [isReady]);
  // Mirror of the store's `lastError`, readable mid-await. The store field
  // captured in an async closure is a render-time snapshot; the ref follows
  // re-renders, so a comparison against a pre-call snapshot sees errors that
  // arrived during the call.
  const lastErrorRef = useRef(lw2.lastError);
  useEffect(() => {
    lastErrorRef.current = lw2.lastError;
  }, [lw2.lastError]);
  const isApplyingExampleRef = useRef(false);

  const lastSentPromptRef = useRef<string>("");
  // Mirror of the React `scene` state. Key handlers and recompose run
  // synchronously and need a ref-current value rather than the captured
  // stale state from their closure.
  const sceneRef = useRef<StructuredScene | null>(null);
  useEffect(() => {
    sceneRef.current = scene;
  }, [scene]);
  const heldSlotsRef = useRef<number[]>([]);

  const recomputePromptAndSend = useCallback(() => {
    // Vertical (jump/crouch) sentence appended to the prose so it matches the
    // motion. Jump: for hold/prompt while held; for charge only once the arc is
    // actually playing (not while merely charging up). Crouch: while C held.
    const jumpEngaged =
      (jumpModeRef.current !== "charge" && jumpHeldRef.current) ||
      jumpArcRef.current.length > 0;
    let vp = "";
    if (jumpEngaged) {
      vp = sceneRef.current?.jumpPrompt ?? "";
    } else if (vertDirRef.current < 0) {
      // C held → the scene's (editable) crouch line, in BOTH modes ("prompt" =
      // line only, "camera" = line alongside the dip), for the whole hold.
      vp = (sceneRef.current?.crouchPrompt ?? "").trim();
    } else if (crouchReleaseDipRef.current) {
      // Just released (camera mode): the "stands back up" line during the
      // release chunk, dropped once the release dip is consumed.
      vp = (sceneRef.current?.standPrompt ?? "").trim();
    }
    // Mirror it into state so the "Show prompt" inspector reflects it live.
    setVerticalPrompt(vp);

    // No active scene → nothing to compose (prompts only flow from scenes).
    if (!sceneRef.current) return;
    const isMoving =
      moveLStackRef.current.length > 0 || moveLatStackRef.current.length > 0;
    const next = composePrompt(
      sceneRef.current,
      isMoving,
      heldSlotsRef.current,
      vp,
    ).trim();
    if (!next) return;
    if (next === lastSentPromptRef.current) return;
    lastSentPromptRef.current = next;
    if (isReadyRef.current) {
      // Resolves with the correlated `prompt_accepted` reply (or
      // `undefined` when the send fails — the model's `command_error`
      // broadcast carries the reason to the toast).
      void lw2.setPrompt({ prompt: next }).then((reply) => {
        if (reply) setHasPrompt(true);
      });
    }
  }, [sendCommand]);
  // Ref indirection so the per-chunk message handler can drop the jump
  // sentence when the arc ends without capturing a stale callback.
  const recomputePromptAndSendRef = useRef<() => void>(() => {});
  useEffect(() => {
    recomputePromptAndSendRef.current = recomputePromptAndSend;
  }, [recomputePromptAndSend]);

  // ---- Messages from backend ----

  // Broadcasts only. On runtime 3.2+ the success acknowledgements
  // (`prompt_accepted`, `image_accepted`, `generation_paused`/`resumed`/
  // `reset`) are correlated replies delivered to the calling connection —
  // they resolve the awaited command instead of arriving here. The
  // handlers below cover what the model still `self.send`s to every
  // connection: shared state, per-chunk progress, and error reports.
  useLingbotWorld2Message((raw) => {
    // The published schema doesn't declare `workers_ready`, so widen the
    // union locally; every other branch narrows to its typed shape.
    const msg = raw as
      | LingbotWorld2Message
      | { type: "workers_ready"; tsp_size?: number };
    if (!msg?.type) return;
    switch (msg.type) {
      case "workers_ready":
        setTspSize(msg.tsp_size ?? null);
        break;
      case "conditions_ready":
        setHasPrompt(msg.has_prompt);
        setHasImage(msg.has_image);
        break;
      case "state":
        setHasPrompt(msg.has_prompt);
        setHasImage(msg.has_image);
        setIsGenerating(msg.running && msg.started);
        setIsPaused(msg.paused);
        setCameraPoseActive(msg.camera_pose_active);
        break;
      case "generation_started":
        setIsGenerating(true);
        setIsPaused(false);
        setChunkNum(msg.chunk_num);
        setChunkIndex(0);
        break;
      case "chunk_complete":
        setChunkIndex(msg.chunk_index);
        setActiveAction(msg.active_action || "still");
        // The crouch dips each last a single chunk. When the release dip ends,
        // drop the "stands up" line too.
        crouchPressDipRef.current = false;
        if (crouchReleaseDipRef.current) {
          crouchReleaseDipRef.current = false;
          recomputePromptAndSendRef.current();
        }
        // Advance the charge-mode jump arc by one chunk (CHUNK_LATENTS latents).
        // The arc auto-plays after release — no hold required during flight.
        if (jumpModeRef.current === "charge" && jumpArcRef.current.length > 0) {
          jumpArcPosRef.current += CHUNK_LATENTS;
          if (jumpArcPosRef.current >= jumpArcRef.current.length) {
            jumpArcRef.current = []; // landed
            jumpArcPosRef.current = 0;
            setJumpLit(false);
            recomputePromptAndSendRef.current(); // drop the jump sentence
          }
        }
        // Drive the native camera-pose layer one chunk at a time.
        sendCameraPoseChunkRef.current();
        break;
      case "generation_complete":
        setIsGenerating(false);
        setIsPaused(false);
        break;
      case "command_error":
        setErrorToast(
          `${msg.command || "?"}: ${msg.reason || "unknown error"}`,
        );
        break;
    }
  });

  useEffect(() => {
    if (!errorToast) return;
    const id = setTimeout(() => setErrorToast(null), 4000);
    return () => clearTimeout(id);
  }, [errorToast]);

  useEffect(() => {
    if (status === "disconnected") {
      setHasPrompt(false);
      setHasImage(false);
      setIsGenerating(false);
      setIsPaused(false);
      setChunkIndex(0);
      setChunkNum(0);
      setActiveAction("still");
      setTspSize(null);
      moveLStackRef.current = [];
      moveLatStackRef.current = [];
      lastSentMoveLRef.current = "idle";
      lastSentMoveLatRef.current = "idle";
      lookHDirRef.current = 0;
      lookVDirRef.current = 0;
      lastSentPromptRef.current = "";
      sceneRef.current = null;
      heldSlotsRef.current = [];
      setHeldSlots([]);
      setMoveL("idle");
      setMoveLat("idle");
      setLookH("idle");
      setLookV("idle");
      setCameraPoseActive(false);
      // Reset the camera-pose layer + release pointer lock.
      if (typeof document !== "undefined" && document.pointerLockElement)
        document.exitPointerLock();
      pendingDYawRef.current = 0;
      pendingDPitchRef.current = 0;
      mouseVizRef.current = { x: 0, y: 0 };
      setMouseViz({ x: 0, y: 0 });
      joyRef.current = { x: 0, y: 0 };
      setJoy({ x: 0, y: 0 });
      mouseLookRef.current = false;
      vertDirRef.current = 0;
      rollDirRef.current = 0;
      jumpHeldRef.current = false;
      jumpArcRef.current = [];
      jumpArcPosRef.current = 0;
      chargeLevelRef.current = 0;
      crouchPressDipRef.current = false;
      crouchReleaseDipRef.current = false;
      poseSentActiveRef.current = false;
      setMouseLook(false);
      setVertState(0);
      setRollDir(0);
      setJumpLit(false);
      setCharging(false);
      setChargeLevel(0);
      setScene(null);
      setActiveExampleId(null);
      setSentImagePreview(null);
      setImageInfo(null);
      setLoadingExampleId(null);
      setPendingImage((prev) => {
        if (prev && prev.previewUrl.startsWith("blob:")) {
          URL.revokeObjectURL(prev.previewUrl);
        }
        return null;
      });
      // Note: do NOT clear overrides on disconnect; they're a presets
      // store that should persist across sessions.
    }
  }, [status]);

  // Sync initial rotation speed to backend on connect
  useEffect(() => {
    if (isReady)
      void lw2.setRotationSpeedDeg({ rotation_speed_deg: rotationSpeed });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isReady]);

  // Re-assert the DiT attention-window + KV-cache-reset selections on connect,
  // so a fresh session reflects the UI state rather than only the config default.
  useEffect(() => {
    if (!isReady) return;
    void lw2.setAttnWindow({ attn_window: attnWindow });
    void lw2.setKvCacheReset({ mode: kvCacheResetMode });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isReady]);

  // ---- Native camera-pose layer (mouse-look + jump) ----

  // Send one per-frame camera-pose DELTA for the next chunk, reflecting input
  // accumulated since the last send. Called on every chunk_complete (and once
  // when a control engages). When nothing is active, send [] once to hand
  // rotation back to the arrow keys / translation back to WASD.
  const sendCameraPoseChunk = useCallback(() => {
    if (!isReadyRef.current) return;
    const joyActive = joyRef.current.x !== 0 || joyRef.current.y !== 0;
    // Jump contributes translation in "hold" (up while held) and "charge"
    // (the per-latent up→down arc while it plays). "prompt" never touches pose.
    const arcActive =
      jumpModeRef.current === "charge" && jumpArcRef.current.length > 0;
    const jumpMoving =
      (jumpModeRef.current === "hold" && jumpHeldRef.current) || arcActive;
    // Crouch camera mode contributes a one-shot dip (press = down / release = up),
    // each lasting exactly one chunk.
    const crouchDip = crouchPressDipRef.current
      ? crouchPatternsRef.current.press
      : crouchReleaseDipRef.current
        ? crouchPatternsRef.current.release
        : null;
    // Crouch "hold" mode contributes a sustained DOWN translation for as long as
    // C is held (mirror of jump "hold" up) — uniform on every latent, re-sent each
    // chunk so the descent continues, no return.
    const crouchHolding =
      crouchModeRef.current === "hold" && vertDirRef.current < 0;
    const arrowLooking = lookHDirRef.current !== 0 || lookVDirRef.current !== 0;
    const active =
      mouseLookRef.current ||
      crouchDip !== null ||
      crouchHolding ||
      rollDirRef.current !== 0 ||
      joyActive ||
      jumpMoving ||
      arrowLooking;
    if (!active) {
      if (poseSentActiveRef.current) {
        void lw2.setCameraPose({ camera_pose: [] });
        poseSentActiveRef.current = false;
      }
      pendingDYawRef.current = 0;
      pendingDPitchRef.current = 0;
      return;
    }
    // Rotation + horizontal translation are UNIFORM across the chunk's latents:
    // yaw (mouse X + arrow left/right) + pitch (mouse Y + arrow up/down) + roll
    // (Q/E), clamped to ±MOUSE_MAX_ROT so a fling can't over-rotate; joystick
    // strafe/forward on tx/tz.
    const clampRot = (v: number) =>
      Math.max(-MOUSE_MAX_ROT, Math.min(MOUSE_MAX_ROT, v));
    const ry = clampRot(
      pendingDYawRef.current * mouseSensRef.current +
        lookHDirRef.current * ARROW_LOOK_SPEED,
    ); // yaw
    const rx = clampRot(
      -pendingDPitchRef.current * mouseSensRef.current +
        lookVDirRef.current * ARROW_LOOK_SPEED,
    ); // pitch
    const rz = rollDirRef.current * ROLL_SPEED; // roll
    pendingDYawRef.current = 0;
    pendingDPitchRef.current = 0;
    // Orbit (Phase 1, horizontal): pair the yaw θ we're already sending with a
    // camera-local strafe (−R·sinθ) + forward sagitta (R·(1−cosθ)) so the point
    // R ahead stays centered while the camera circles it. θ is the clamped ry,
    // so a fling can't over-strafe either; adds to any joystick translation.
    // Zero yaw → zero orbit motion, so this only moves while you mouse-look.
    let orbitTx = 0,
      orbitTz = 0;
    const R = orbitRadiusRef.current;
    if (R > 0 && ry !== 0) {
      orbitTx = -R * Math.sin(ry);
      orbitTz = R * (1 - Math.cos(ry));
    }
    const tx = joyRef.current.x * JOY_SPEED + orbitTx; // joystick strafe + orbit
    const tz = -joyRef.current.y * JOY_SPEED + orbitTz; // joystick forward (up = +Z) + orbit
    // Vertical (ty) is the only PER-LATENT component. Jump: "hold" = up on every
    // latent; "charge" = the arc's per-latent intent. Crouch (camera mode) = a
    // one-shot small down on the FIRST latent only, additive to WASD/forward
    // (the backend sums the action + camera_pose translations). +up = JUMP_UP_SIGN.
    const uniformJumpTy =
      jumpModeRef.current === "hold" && jumpHeldRef.current
        ? JUMP_SPEED * JUMP_UP_SIGN
        : 0;
    // Crouch "hold": sustained DOWN, opposite sign to jump's up, uniform per latent.
    const uniformCrouchTy = crouchHolding ? CROUCH_SPEED * -JUMP_UP_SIGN : 0;
    const arc = jumpArcRef.current;
    const pos = jumpArcPosRef.current;
    const camera_pose: number[] = [];
    for (let j = 0; j < CHUNK_LATENTS; j++) {
      const intent = arcActive && pos + j < arc.length ? arc[pos + j] : 0; // +1/0/-1
      const jumpTy = arcActive
        ? intent * JUMP_SPEED * JUMP_UP_SIGN
        : uniformJumpTy;
      // Crouch: camera-mode dip (editable per-latent +1 up/0 still/-1 down), else the
      // hold-mode sustained down. Dips and hold are mutually exclusive (distinct modes).
      const crouchTy = crouchDip
        ? crouchDip[j] * CROUCH_DIP * JUMP_UP_SIGN
        : uniformCrouchTy;
      camera_pose.push(rx, ry, rz, tx, jumpTy + crouchTy, tz);
    }
    void lw2.setCameraPose({ camera_pose });
    poseSentActiveRef.current = true;
  }, [sendCommand]);
  useEffect(() => {
    sendCameraPoseChunkRef.current = sendCameraPoseChunk;
  }, [sendCameraPoseChunk]);

  // Pointer-lock mouse look: click the toggle to engage, move the mouse to
  // rotate (yaw from movementX, pitch from movementY), Esc / M to release.
  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!mouseLookRef.current) return;
      pendingDYawRef.current += e.movementX; // right → yaw right
      pendingDPitchRef.current += e.movementY; // converted (negated) at send
      // Feed the live HUD: accumulate recent motion (decayed each rAF frame).
      mouseVizRef.current.x += e.movementX;
      mouseVizRef.current.y += e.movementY;
    };
    const onLockChange = () => {
      const locked = document.pointerLockElement != null;
      mouseLookRef.current = locked;
      setMouseLook(locked);
      if (!locked) {
        // released → clear the HUD vector
        mouseVizRef.current.x = 0;
        mouseVizRef.current.y = 0;
        setMouseViz({ x: 0, y: 0 });
      }
      // Push promptly so the next chunk reflects the new state (engage or release).
      sendCameraPoseChunk();
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("pointerlockchange", onLockChange);
    return () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("pointerlockchange", onLockChange);
    };
  }, [sendCameraPoseChunk]);

  // HUD animation: while mouse-look is on, decay the viz vector toward 0 each
  // frame so the arrow shows recent motion direction + strength and settles
  // when the mouse stops.
  useEffect(() => {
    if (!mouseLook) return;
    let raf = 0;
    const tick = () => {
      const v = mouseVizRef.current;
      v.x *= 0.8;
      v.y *= 0.8;
      if (Math.abs(v.x) < 0.1) v.x = 0;
      if (Math.abs(v.y) < 0.1) v.y = 0;
      setMouseViz({ x: v.x, y: v.y });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [mouseLook]);

  const toggleMouseLook = useCallback(() => {
    if (mouseLookRef.current) {
      document.exitPointerLock();
    } else {
      try {
        document.body.requestPointerLock();
      } catch {
        /* needs user gesture */
      }
    }
  }, []);

  // Crouch control (C): -1 = held, 0 = off. Jump is separate.
  const setVert = useCallback(
    (dir: number) => {
      if (vertDirRef.current === dir) return;
      const wasIdle = vertDirRef.current === 0;
      vertDirRef.current = dir;
      setVertState(dir);
      // "hold" mode has no dip flags — the sustained DOWN is emitted every chunk by
      // sendCameraPoseChunk while vertDir < 0 (see `crouchHolding` there).
      // "camera" mode fires one-shot dips (consumed after one chunk): the PRESS
      // pattern (down) on C-down, the RELEASE pattern (stand back up) on C-up.
      // Prompt lines ride along via recomputePromptAndSend (crouch while held,
      // stand on release).
      if (crouchModeRef.current === "camera") {
        if (dir < 0 && wasIdle)
          crouchPressDipRef.current = true; // pressed → dip down
        else if (dir === 0) crouchReleaseDipRef.current = true; // released → stand up
      }
      sendCameraPoseChunk(); // engage the dip promptly (camera mode)
      recomputePromptAndSend(); // crouch line while held / stand line on release
    },
    [sendCameraPoseChunk, recomputePromptAndSend],
  );

  // Jump press. Behaviour depends on the "Jump" mode switch:
  //   hold   — engage a sustained UP translation while held.
  //   prompt — just append the scene's jumpPrompt (no camera_pose).
  //   charge — start the charge meter stepping through levels; no motion/prompt yet.
  const onJumpDown = useCallback(() => {
    // Block re-triggers: ignore key-repeat while held, AND ignore a fresh press
    // while a charge arc is still airborne — no re-jump until it lands (no
    // double-jump), mirroring how a real jump can't restart mid-flight.
    if (jumpHeldRef.current || jumpArcRef.current.length > 0) return;
    jumpHeldRef.current = true;
    setJumpLit(true);
    if (jumpModeRef.current === "charge") {
      jumpArcPosRef.current = 0; // arc is guaranteed empty here (guarded above)
      chargeLevelRef.current = 1;
      chargeLevelDirRef.current = 1;
      setChargeLevel(1);
      setCharging(true); // meter starts stepping; jump fires on release
      return; // no translation / no prompt while charging
    }
    sendCameraPoseChunk(); // hold: engage the up translation; prompt: no-op
    recomputePromptAndSend(); // append the jump sentence
  }, [sendCameraPoseChunk, recomputePromptAndSend]);

  // Jump release.
  const onJumpUp = useCallback(() => {
    if (!jumpHeldRef.current) return;
    jumpHeldRef.current = false;
    if (jumpModeRef.current === "charge") {
      // Freeze the meter at its discrete level and fire that level's hand-authored
      // per-latent pattern (edited in the grid popup; +1 up / -1 down / 0 still).
      // Auto-plays via the chunk driver.
      setCharging(false);
      const level = chargeLevelRef.current || 1;
      jumpArcRef.current = [...(chargePatternsRef.current[level - 1] ?? [])];
      jumpArcPosRef.current = 0;
      setChargeLevel(0); // meter empties; the arc is now in flight
      // keep lit + prompt through the flight (dropped when it lands)
      sendCameraPoseChunk(); // engage the first chunk's latents promptly
      recomputePromptAndSend(); // append the jump sentence for the arc
      return;
    }
    setJumpLit(false);
    sendCameraPoseChunk(); // hold: release the up translation; prompt: no-op
    recomputePromptAndSend(); // drop the jump sentence
  }, [sendCameraPoseChunk, recomputePromptAndSend]);

  // Flip the Jump mode. Cancel any in-flight jump / charge so the switch is clean.
  const changeJumpMode = useCallback(
    (mode: JumpMode) => {
      if (jumpModeRef.current === mode) return;
      jumpModeRef.current = mode;
      setJumpMode(mode);
      jumpHeldRef.current = false;
      jumpArcRef.current = [];
      jumpArcPosRef.current = 0;
      setCharging(false);
      chargeLevelRef.current = 0;
      setChargeLevel(0);
      setJumpLit(false);
      sendCameraPoseChunk(); // clear any lingering up/down pose
      recomputePromptAndSend(); // drop the jump sentence if it was appended
    },
    [sendCameraPoseChunk, recomputePromptAndSend],
  );

  // Flip the Crouch mode; release any held crouch so nothing sticks.
  const changeCrouchMode = useCallback(
    (mode: CrouchMode) => {
      if (crouchModeRef.current === mode) return;
      crouchModeRef.current = mode;
      setCrouchMode(mode);
      setVert(0); // release any held crouch (clears vertDir, re-sends pose)
      crouchPressDipRef.current = false;
      crouchReleaseDipRef.current = false; // after setVert, so no stray dip
    },
    [setVert],
  );

  // --- Charge-level grid editing (persisted) ---
  const persistChargePatterns = (next: number[][]) => {
    try {
      localStorage.setItem(CHARGE_PATTERNS_STORAGE, JSON.stringify(next));
    } catch {
      /* ignore */
    }
  };
  // Cycle one latent cell: up (+1) → down (-1) → still (0) → up.
  const cycleChargeCell = useCallback((level: number, idx: number) => {
    setChargePatterns((prev) => {
      const next = prev.map((p) => [...p]);
      const cur = next[level - 1][idx];
      next[level - 1][idx] = cur === 1 ? -1 : cur === -1 ? 0 : 1;
      persistChargePatterns(next);
      return next;
    });
  }, []);
  const resetChargeLevel = useCallback((level: number) => {
    setChargePatterns((prev) => {
      const next = prev.map((p) => [...p]);
      next[level - 1] = defaultChargePattern(level);
      persistChargePatterns(next);
      return next;
    });
  }, []);
  // Crouch press/release pattern editing (persisted).
  const persistCrouchPatterns = (next: CrouchPatterns) => {
    try {
      localStorage.setItem(CROUCH_PATTERN_STORAGE, JSON.stringify(next));
    } catch {
      /* ignore */
    }
  };
  const cycleCrouchCell = useCallback((phase: CrouchPhase, idx: number) => {
    setCrouchPatterns((prev) => {
      const arr = [...prev[phase]];
      const cur = arr[idx];
      arr[idx] = cur === 1 ? -1 : cur === -1 ? 0 : 1; // up → down → still → up
      const next = { ...prev, [phase]: arr };
      persistCrouchPatterns(next);
      return next;
    });
  }, []);
  const resetCrouchPatterns = useCallback(() => {
    const next = defaultCrouchPatterns();
    setCrouchPatterns(next);
    persistCrouchPatterns(next);
  }, []);

  // Charge meter: while charging, step through the discrete levels (1..NUM),
  // dwelling LEVEL_DWELL_MS on each and bouncing at the ends.
  useEffect(() => {
    if (!charging) return;
    let raf = 0;
    let lastStep = 0; // timestamp of the last level change
    const tick = (t: number) => {
      if (lastStep === 0) lastStep = t;
      if (t - lastStep >= LEVEL_DWELL_MS) {
        lastStep = t;
        let lvl = chargeLevelRef.current + chargeLevelDirRef.current;
        if (lvl >= NUM_CHARGE_LEVELS) {
          lvl = NUM_CHARGE_LEVELS;
          chargeLevelDirRef.current = -1;
        } else if (lvl <= 1) {
          lvl = 1;
          chargeLevelDirRef.current = 1;
        }
        chargeLevelRef.current = lvl;
        setChargeLevel(lvl);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [charging]);

  // Roll (the 3rd rotation DOF the mouse can't reach): -1 = Q, +1 = E, 0 = off.
  const setRoll = useCallback(
    (dir: number) => {
      if (rollDirRef.current === dir) return;
      rollDirRef.current = dir;
      setRollDir(dir);
      sendCameraPoseChunk();
    },
    [sendCameraPoseChunk],
  );

  // Drag joystick → continuous translation (camera_pose tx/tz). Unlike the
  // discrete WASD buttons, this gives a continuous direction + magnitude. The
  // knob position is normalized to the unit disk; releasing snaps back to 0.
  const onJoyPointer = useCallback(
    (e: React.PointerEvent, kind: "down" | "move" | "up") => {
      if (kind === "up") {
        joyRef.current = { x: 0, y: 0 };
        setJoy({ x: 0, y: 0 });
        sendCameraPoseChunk();
        return;
      }
      const el = joyAreaRef.current;
      if (!el) return;
      const r = el.getBoundingClientRect();
      let x = (e.clientX - (r.left + r.width / 2)) / (r.width / 2);
      let y = (e.clientY - (r.top + r.height / 2)) / (r.height / 2);
      const m = Math.hypot(x, y);
      if (m > 1) {
        x /= m;
        y /= m;
      } // clamp to unit disk
      joyRef.current = { x, y };
      setJoy({ x, y });
      if (kind === "down") sendCameraPoseChunk(); // engage promptly
    },
    [sendCameraPoseChunk],
  );

  // ---- Movement / look emitters ----

  const pushMoveL = useCallback(
    (next: MoveL) => {
      setMoveL(next);
      if (lastSentMoveLRef.current === next) return;
      lastSentMoveLRef.current = next;
      if (isReadyRef.current)
        void lw2.setMoveLongitudinal({ move_longitudinal: next });
    },
    [sendCommand],
  );

  const pushMoveLat = useCallback(
    (next: MoveLat) => {
      setMoveLat(next);
      if (lastSentMoveLatRef.current === next) return;
      lastSentMoveLatRef.current = next;
      if (isReadyRef.current) void lw2.setMoveLateral({ move_lateral: next });
    },
    [sendCommand],
  );

  const pushLookH = useCallback(
    (next: LookH) => {
      setLookH(next);
      const dir = next === "right" ? 1 : next === "left" ? -1 : 0;
      if (lookHDirRef.current === dir) return;
      lookHDirRef.current = dir;
      sendCameraPoseChunk(); // engage promptly, like roll/joystick
    },
    [sendCameraPoseChunk],
  );

  const pushLookV = useCallback(
    (next: LookV) => {
      setLookV(next);
      const dir = next === "up" ? 1 : next === "down" ? -1 : 0;
      if (lookVDirRef.current === dir) return;
      lookVDirRef.current = dir;
      sendCameraPoseChunk(); // engage promptly, like roll/joystick
    },
    [sendCameraPoseChunk],
  );

  const applyMovementStack = useCallback(() => {
    const topL = moveLStackRef.current.at(-1);
    const topLat = moveLatStackRef.current.at(-1);
    pushMoveL(topL ?? "idle");
    pushMoveLat(topLat ?? "idle");
    // Structured prompts depend on whether anything is held; recompose so
    // base + movement[isMoving] + events stays in sync with the input state.
    if (sceneRef.current) recomputePromptAndSend();
  }, [pushMoveL, pushMoveLat, recomputePromptAndSend]);

  // Zero every held-input control (WASD/joystick/look/roll/vert + mouse viz).
  // Called on reset: a key still physically held when Reset is clicked can lose
  // its keyup (focus jumps to the Reset button), which would otherwise leave the
  // joystick mirror — and the move state it reflects — stuck deflected.
  const clearMovementInputs = useCallback(() => {
    moveLStackRef.current = [];
    moveLatStackRef.current = [];
    joyRef.current = { x: 0, y: 0 };
    setJoy({ x: 0, y: 0 });
    pendingDYawRef.current = 0;
    pendingDPitchRef.current = 0;
    mouseVizRef.current = { x: 0, y: 0 };
    setMouseViz({ x: 0, y: 0 });
    setVert(0);
    setRoll(0);
    jumpHeldRef.current = false;
    jumpArcRef.current = [];
    jumpArcPosRef.current = 0;
    crouchPressDipRef.current = false;
    crouchReleaseDipRef.current = false;
    setCharging(false);
    chargeLevelRef.current = 0;
    setChargeLevel(0);
    setJumpLit(false);
    applyMovementStack(); // pushes moveL/moveLat back to "idle"
    pushLookH("idle");
    pushLookV("idle");
  }, [applyMovementStack, pushLookH, pushLookV, setVert, setRoll]);

  // Full local cleanup after a user-initiated Reset. On runtime 3.2+ the
  // model's `generation_reset` is the correlated reply to `lw2.reset()`
  // (not a broadcast), so `sendLifecycle` awaits the call and invokes
  // this on the ack. `applyScene` does its own narrower cleanup inline,
  // matching what the old broadcast handler skipped while
  // `isApplyingExampleRef` was set.
  const handleGenerationReset = useCallback(() => {
    setIsGenerating(false);
    setIsPaused(false);
    clearMovementInputs(); // never leave a held control stuck after a reset
    setHasPrompt(false);
    setHasImage(false);
    setLoadingExampleId(null);
    setSentImagePreview(null);
    setImageInfo(null);
    setChunkIndex(0);
    lastSentPromptRef.current = "";
    heldSlotsRef.current = [];
    setHeldSlots([]);
    sceneRef.current = null;
    setScene(null);
    setActiveExampleId(null);
    // Overrides persist across resets (they're per-example presets,
    // not session state).
    setPendingImage((p) => {
      if (p && p.previewUrl.startsWith("blob:")) {
        URL.revokeObjectURL(p.previewUrl);
      }
      return null;
    });
  }, [clearMovementInputs]);

  // ---- Prompt handlers ----

  const holdPress = useCallback(
    (slot: number) => {
      const events = sceneRef.current?.events;
      if (!events || slot < 0 || slot >= events.length) return;
      if (!heldSlotsRef.current.includes(slot)) {
        heldSlotsRef.current = [...heldSlotsRef.current, slot];
      }
      setHeldSlots(heldSlotsRef.current);
      recomputePromptAndSend();
    },
    [recomputePromptAndSend],
  );

  const holdRelease = useCallback(
    (slot: number) => {
      if (!heldSlotsRef.current.includes(slot)) return;
      heldSlotsRef.current = heldSlotsRef.current.filter((x) => x !== slot);
      setHeldSlots(heldSlotsRef.current);
      recomputePromptAndSend();
    },
    [recomputePromptAndSend],
  );

  useEffect(() => {
    const onBlur = () => {
      moveLStackRef.current = [];
      moveLatStackRef.current = [];
      applyMovementStack();
      pushLookH("idle");
      pushLookV("idle");
      heldSlotsRef.current = [];
      setHeldSlots([]);
      recomputePromptAndSend();
    };
    window.addEventListener("blur", onBlur);
    return () => window.removeEventListener("blur", onBlur);
  }, [applyMovementStack, pushLookH, pushLookV, recomputePromptAndSend]);

  // ---- Keyboard ----

  useEffect(() => {
    const isTypingTarget = (el: EventTarget | null): boolean => {
      if (!(el instanceof HTMLElement)) return false;
      const tag = el.tagName;
      return tag === "INPUT" || tag === "TEXTAREA" || el.isContentEditable;
    };

    // When a movement / look key is pressed but a button (or other
    // focusable, non-typing element) inside a scrollable ancestor still
    // holds focus, the browser may scroll that ancestor before our window
    // handler runs preventDefault. Move focus off such elements before
    // dispatching, so the only effect of the key is the controller action.
    const blurFocusedNonTyping = () => {
      const el = document.activeElement as HTMLElement | null;
      if (!el || el === document.body) return;
      if (isTypingTarget(el)) return;
      el.blur?.();
    };

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.repeat) return;
      if (isTypingTarget(e.target)) return;

      const mvL = KEY_TO_MOVE_L[e.key];
      if (mvL) {
        e.preventDefault();
        blurFocusedNonTyping();
        const stack = moveLStackRef.current;
        if (!stack.includes(mvL)) stack.push(mvL);
        applyMovementStack();
        return;
      }
      const mvLat = KEY_TO_MOVE_LAT[e.key];
      if (mvLat) {
        e.preventDefault();
        blurFocusedNonTyping();
        const stack = moveLatStackRef.current;
        if (!stack.includes(mvLat)) stack.push(mvLat);
        applyMovementStack();
        return;
      }
      // Arrow look feeds the same camera_pose yaw/pitch as mouse-look (see
      // ARROW_LOOK_SPEED / sendCameraPoseChunk) — a steady, fixed-rate look,
      // so it also drives orbit. An arrow key while mouse-look (pointer lock)
      // is engaged still preempts it (release the lock, same as Esc) so the
      // arrows read as a clean, deliberate handoff from free-look.
      const lh = KEY_TO_LOOK_H[e.key];
      if (lh) {
        e.preventDefault();
        blurFocusedNonTyping();
        if (mouseLookRef.current) document.exitPointerLock();
        pushLookH(lh);
        return;
      }
      const lv = KEY_TO_LOOK_V[e.key];
      if (lv) {
        e.preventDefault();
        blurFocusedNonTyping();
        if (mouseLookRef.current) document.exitPointerLock();
        pushLookV(lv);
        return;
      }

      // Esc / M release mouse-look (the cursor is hidden under pointer lock,
      // so the toggle button can't be clicked — a key is the way out).
      if (
        (e.key === "Escape" || e.key === "m" || e.key === "M") &&
        mouseLookRef.current
      ) {
        e.preventDefault();
        document.exitPointerLock();
        return;
      }

      // Space (and J) = Jump (up); C = Crouch (down) — hold controls, like
      // WASD. (Pause/resume is the on-screen button now; Space-to-pause made
      // no sense in a WASD game.) Crouch is on C, NOT Ctrl: macOS reserves
      // Ctrl+arrows for Spaces / Mission Control and grabs them at the OS level
      // before the page sees the keydown, so a Ctrl-held crouch would silently
      // swallow arrow-look (rotation) — an unfixable collision from JS.
      if (e.code === "Space" || e.key === "j" || e.key === "J") {
        e.preventDefault();
        blurFocusedNonTyping();
        onJumpDown();
        return;
      }
      if (e.key === "c" || e.key === "C") {
        e.preventDefault();
        blurFocusedNonTyping();
        setVert(-1);
        return;
      }

      // Q / E = roll (3rd rotation DOF, around the view axis).
      if (e.key === "q" || e.key === "Q") {
        e.preventDefault();
        blurFocusedNonTyping();
        setRoll(-1);
        return;
      }
      if (e.key === "e" || e.key === "E") {
        e.preventDefault();
        blurFocusedNonTyping();
        setRoll(1);
        return;
      }

      // O = mute / un-mute orbit: toggle R between 0 (rotate in place) and the last non-zero radius.
      if (e.key === "o" || e.key === "O") {
        e.preventDefault();
        blurFocusedNonTyping();
        setOrbitRadius((r) =>
          r > 0 ? 0 : lastOrbitRadiusRef.current || ORBIT_RADIUS_DEFAULT,
        );
        return;
      }

      const slot = keyToHoldSlot(e.key);
      if (slot !== undefined) {
        e.preventDefault();
        holdPress(slot);
        return;
      }
    };

    const onKeyUp = (e: KeyboardEvent) => {
      const mvL = KEY_TO_MOVE_L[e.key];
      if (mvL) {
        moveLStackRef.current = moveLStackRef.current.filter((m) => m !== mvL);
        applyMovementStack();
        return;
      }
      const mvLat = KEY_TO_MOVE_LAT[e.key];
      if (mvLat) {
        moveLatStackRef.current = moveLatStackRef.current.filter(
          (m) => m !== mvLat,
        );
        applyMovementStack();
        return;
      }
      if (KEY_TO_LOOK_H[e.key]) {
        pushLookH("idle");
        return;
      }
      if (KEY_TO_LOOK_V[e.key]) {
        pushLookV("idle");
        return;
      }
      if (e.code === "Space" || e.key === "j" || e.key === "J") {
        onJumpUp();
        return;
      }
      if (e.key === "c" || e.key === "C") {
        if (vertDirRef.current < 0) setVert(0);
        return;
      }
      if (e.key === "q" || e.key === "Q" || e.key === "e" || e.key === "E") {
        setRoll(0);
        return;
      }
      const slot = keyToHoldSlot(e.key);
      if (slot !== undefined) {
        holdRelease(slot);
        return;
      }
    };

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [
    applyMovementStack,
    pushLookH,
    pushLookV,
    holdPress,
    holdRelease,
    setVert,
    setRoll,
    onJumpDown,
    onJumpUp,
  ]);

  const onMoveLPress = (mv: Exclude<MoveL, "idle">) => {
    const stack = moveLStackRef.current;
    if (!stack.includes(mv)) stack.push(mv);
    applyMovementStack();
  };
  const onMoveLRelease = (mv: Exclude<MoveL, "idle">) => {
    moveLStackRef.current = moveLStackRef.current.filter((m) => m !== mv);
    applyMovementStack();
  };
  const onMoveLatPress = (mv: Exclude<MoveLat, "idle">) => {
    const stack = moveLatStackRef.current;
    if (!stack.includes(mv)) stack.push(mv);
    applyMovementStack();
  };
  const onMoveLatRelease = (mv: Exclude<MoveLat, "idle">) => {
    moveLatStackRef.current = moveLatStackRef.current.filter((m) => m !== mv);
    applyMovementStack();
  };

  // ---- Image selection + send ----

  useEffect(() => {
    return () => {
      if (pendingImage && pendingImage.previewUrl.startsWith("blob:")) {
        URL.revokeObjectURL(pendingImage.previewUrl);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingImage?.previewUrl]);

  const clearPendingImage = () => {
    if (pendingImage && pendingImage.previewUrl.startsWith("blob:")) {
      URL.revokeObjectURL(pendingImage.previewUrl);
    }
    setPendingImage(null);
  };

  const selectFile = (file: File) => {
    clearPendingImage();
    const previewUrl = URL.createObjectURL(file);
    setPendingImage({ file, previewUrl, label: file.name });
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const sendPendingImage = async () => {
    if (!pendingImage || !isReady || isUploading) return;
    setIsUploading(true);
    try {
      const ref = await uploadFile(pendingImage.file);
      // Resolves with the correlated `image_accepted` reply carrying the
      // decoded dimensions; `undefined` means the send failed (the
      // model's `command_error` broadcast carries the reason).
      const accepted = await lw2.setImage({ image: ref });
      if (!accepted) return;
      setSentImagePreview(pendingImage.previewUrl);
      setHasImage(true);
      setImageInfo({ w: accepted.width, h: accepted.height });
      setPendingImage((p) =>
        p && p.previewUrl === pendingImage.previewUrl ? null : p,
      );
    } catch (err) {
      console.error(err);
      setErrorToast(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setIsUploading(false);
    }
  };

  // ---- Apply a scene: uploads image, sends prompt, starts generation ----

  // Shared flow behind both the Quick Start examples and the custom scene:
  // clear held inputs, reset a running generation, upload the starting image,
  // send the composed prompt, and auto-start.
  const applyScene = useCallback(
    async (opts: {
      id: string;
      scene: StructuredScene;
      image:
        | { kind: "url"; src: string; name: string }
        | { kind: "file"; file: File; previewUrl: string }
        | { kind: "keep" }; // reuse the already-sent image
      errorLabel: string;
    }) => {
      // Blur whatever was just clicked so subsequent arrow-key presses don't
      // scroll the sidebar's overflow container. Without this, focus stays on
      // the clicked button and the browser treats arrow-key presses inside the
      // scrollable ancestor as scroll input alongside our window-level look
      // handler.
      if (typeof document !== "undefined") {
        (document.activeElement as HTMLElement | null)?.blur?.();
      }

      isApplyingExampleRef.current = true;
      try {
        // Clear any held control state so stale refs don't linger across the switch
        moveLStackRef.current = [];
        moveLatStackRef.current = [];
        pushMoveL("idle");
        pushMoveLat("idle");
        pushLookH("idle");
        pushLookV("idle");
        heldSlotsRef.current = [];
        setHeldSlots([]);
        sceneRef.current = null;

        // If currently generating or paused, reset first so the new scene
        // starts clean. The awaited call resolves with the correlated
        // `generation_reset` reply once the model's handler has run — no
        // arbitrary sleep needed before sending new data.
        if (isGenerating || isPaused) {
          await lw2.reset();
          setIsGenerating(false);
          setIsPaused(false);
          setHasPrompt(false);
          setHasImage(false);
          if (opts.image.kind !== "keep") {
            setSentImagePreview(null);
            setImageInfo(null);
          }
          lastSentPromptRef.current = "";
        }

        setLoadingExampleId(opts.id);

        // A reply-declaring command resolving `undefined` either failed or
        // completed with a bodyless acknowledgement. `lastError` is a
        // persistent record (success never clears it), so only an error
        // that appeared SINCE the pre-call snapshot counts as this call's
        // failure. Report which one happened — nothing more; the cause is
        // for whoever reads it to determine.
        const sendImage = async (ref: FileRef) => {
          const errorBefore = lastErrorRef.current;
          const accepted = await lw2.setImage({ image: ref });
          if (accepted) return accepted;
          const errorNow = lastErrorRef.current;
          throw new Error(
            errorNow && errorNow !== errorBefore
              ? `set_image failed: [${errorNow.code}] ${errorNow.message}`
              : "set_image completed but returned no reply payload",
          );
        };

        try {
          // Upload the starting image ("keep" assumes the previously-sent
          // image is still in place).
          if (opts.image.kind === "url") {
            const res = await fetch(opts.image.src);
            if (!res.ok)
              throw new Error(`Failed to load image (${res.status})`);
            const blob = await res.blob();
            const file = new File([blob], opts.image.name, {
              type: blob.type || "image/jpeg",
            });
            const ref = await uploadFile(file);
            const accepted = await sendImage(ref);
            setSentImagePreview(opts.image.src);
            setHasImage(true);
            setImageInfo({ w: accepted.width, h: accepted.height });
          } else if (opts.image.kind === "file") {
            const ref = await uploadFile(opts.image.file);
            const accepted = await sendImage(ref);
            setSentImagePreview(opts.image.previewUrl);
            setHasImage(true);
            setImageInfo({ w: accepted.width, h: accepted.height });
            // Don't revoke the previewUrl — it was just promoted to
            // sentImagePreview, so the URL is still in use.
            setPendingImage(null);
          }

          // Send the scene's composed prompt
          sceneRef.current = opts.scene;
          setScene(opts.scene);
          setActiveExampleId(opts.id);
          const p = composePrompt(opts.scene, false, []).trim();
          lastSentPromptRef.current = p;
          const promptAck = await lw2.setPrompt({ prompt: p });
          if (promptAck) setHasPrompt(true);

          // Auto-start. The image/prompt acks above already guarantee
          // their handlers ran, so no settle delay is needed; `start`
          // declares no reply — the await is the completion barrier, and
          // the `generation_started` broadcast confirms it took.
          await lw2.start();
          setIsGenerating(true);
        } catch (err) {
          console.error(err);
          setErrorToast(err instanceof Error ? err.message : opts.errorLabel);
        } finally {
          setLoadingExampleId(null);
        }
      } finally {
        isApplyingExampleRef.current = false;
      }
    },
    [
      isGenerating,
      isPaused,
      uploadFile,
      sendCommand,
      pushMoveL,
      pushMoveLat,
      pushLookH,
      pushLookV,
    ],
  );

  const applyExample = useCallback(
    async (ex: StructuredExample) => {
      if (!isReady || isUploading) return;
      // Resolve the example's scene through the override store: if the user
      // has edited this example before, those edits are applied from the
      // start. Otherwise the pristine constant is used.
      const effective = effectiveSceneFor(ex.id);
      if (!effective) return;
      await applyScene({
        id: ex.id,
        scene: effective,
        image: { kind: "url", src: ex.image.src, name: `${ex.id}.jpg` },
        errorLabel: "Failed to apply example",
      });
    },
    [isReady, isUploading, effectiveSceneFor, applyScene],
  );

  // ---- Lifecycle ----

  const canStart =
    isReady && hasPrompt && hasImage && !isGenerating && !isPaused;
  const canPauseResume = isReady && (isGenerating || isPaused);
  const canReset = isReady;

  const startBlockerReason = useMemo(() => {
    if (canStart) return null;
    if (!isReady) return "Not connected.";
    if (isGenerating) return "Already generating — Reset to start over.";
    const missing: string[] = [];
    if (!hasPrompt) missing.push("send a prompt");
    if (!hasImage) missing.push("send a starting image");
    if (missing.length === 0) return null;
    return `Need to ${missing.join(" and ")}.`;
  }, [canStart, isReady, isGenerating, hasPrompt, hasImage]);

  // Lifecycle buttons set state from the command's correlated reply
  // rather than optimistically: `pause`/`resume`/`reset` resolve with
  // `generation_paused`/`generation_resumed`/`generation_reset` once the
  // handler has run (or `undefined` on failure, with the reason arriving
  // as a `command_error` broadcast → toast). `start` declares no reply;
  // its `generation_started` broadcast flips the state.
  const sendLifecycle = (cmd: "start" | "pause" | "resume" | "reset") => {
    switch (cmd) {
      case "start":
        void lw2.start();
        break;
      case "pause":
        void lw2.pause().then((reply) => {
          if (reply) setIsPaused(true);
        });
        break;
      case "resume":
        void lw2.resume().then((reply) => {
          if (reply) setIsPaused(false);
        });
        break;
      case "reset":
        void lw2.reset().then((reply) => {
          if (reply) handleGenerationReset();
        });
        break;
    }
  };

  const pushRotationSpeed = (v: number) => {
    setRotationSpeed(v);
    if (isReady) void lw2.setRotationSpeedDeg({ rotation_speed_deg: v });
  };

  const pushSeed = (v: number) => {
    setSeed(v);
    if (isReady) void lw2.setSeed({ seed: v });
  };

  // DiT attention-window selector (auto / small / large).
  const pushAttnWindow = (w: AttnWindow) => {
    if (attnWindow === w) return;
    setAttnWindow(w);
    if (isReady) void lw2.setAttnWindow({ attn_window: w });
  };

  // KV-cache / RoPE reset mode selector (off / auto / manual).
  const pushKvCacheResetMode = (mode: KvResetMode) => {
    if (kvCacheResetMode === mode) return;
    setKvCacheResetMode(mode);
    if (isReady) void lw2.setKvCacheReset({ mode });
  };

  // One-shot forced KV-cache reset. The backend honors this in "auto" and
  // "manual" modes but rejects it with command_error when the mode is "off",
  // so the button is disabled in the UI when the mode is "off".
  const triggerKvCacheReset = () => {
    if (isReady && kvCacheResetMode !== "off") {
      void lw2.triggerKvCacheReset();
    }
  };

  // ---- Scene editor helpers ----

  // Scene shown in the editor modal. Reads from the override store if
  // present; for built-in examples falls back to the pristine constant,
  // for the custom slot falls back to a blank scene so the user has
  // something concrete to start editing.
  const editingScene = useMemo<StructuredScene | null>(() => {
    if (!editingExampleId) return null;
    const override = overrides[editingExampleId];
    if (override) return override;
    if (editingExampleId === CUSTOM_SCENE_ID) return emptyScene();
    return STRUCTURED_EXAMPLES[editingExampleId]?.scene ?? null;
  }, [editingExampleId, overrides]);

  // Editor onChange handler. Writes to overrides[editingExampleId] (so
  // edits persist and re-clicks load them). If the user is editing the
  // currently active example, also update sceneRef + scene so the
  // running generation sees the edit immediately.
  //
  // If the edited content is byte-equal to the pristine constant, drop
  // the override instead of storing it — there's no semantic
  // difference, and persisting it would leave a stale "edited" badge
  // on a card whose effective prompt matches the default. For the
  // custom slot there's no pristine to revert to; drop the override
  // only when the user has cleared every field back to emptyScene().
  const handleSceneChange = useCallback(
    (next: StructuredScene) => {
      if (!editingExampleId) return;
      const pristine =
        editingExampleId === CUSTOM_SCENE_ID
          ? emptyScene()
          : STRUCTURED_EXAMPLES[editingExampleId]?.scene;
      const isDefault = pristine ? scenesEqual(next, pristine) : false;
      setOverrides((o) => {
        if (isDefault) {
          if (!(editingExampleId in o)) return o;
          const without = { ...o };
          delete without[editingExampleId];
          return without;
        }
        return { ...o, [editingExampleId]: next };
      });
      if (editingExampleId === activeExampleId) {
        sceneRef.current = next;
        setScene(next);
        if (heldSlotsRef.current.length > 0) {
          const valid = heldSlotsRef.current.filter(
            (s) => s < next.events.length,
          );
          if (valid.length !== heldSlotsRef.current.length) {
            heldSlotsRef.current = valid;
            setHeldSlots(valid);
          }
        }
        recomputePromptAndSend();
      }
    },
    [editingExampleId, activeExampleId, recomputePromptAndSend],
  );

  // "Reset to example" inside the editor — drop the override so the
  // example reverts to its pristine constant.
  const resetEditingExample = useCallback(() => {
    if (!editingExampleId) return;
    setOverrides((o) => {
      if (!(editingExampleId in o)) return o;
      const next = { ...o };
      delete next[editingExampleId];
      return next;
    });
    if (editingExampleId === activeExampleId) {
      const pristine = STRUCTURED_EXAMPLES[editingExampleId]?.scene;
      if (pristine) {
        const cloned = cloneScene(pristine);
        sceneRef.current = cloned;
        setScene(cloned);
        recomputePromptAndSend();
      }
    }
  }, [editingExampleId, activeExampleId, recomputePromptAndSend]);

  const openEditorFor = (id: string) => {
    setEditingExampleId(id);
  };

  const closeEditor = () => setEditingExampleId(null);

  // Drop a specific override (from the card-level reset button, or
  // from the Custom section's ↺). If that scene is currently running,
  // also rewind the active scene — to the pristine constant for built-
  // in examples, or unset entirely for the custom slot.
  const clearOverrideFor = useCallback(
    (id: string) => {
      if (!(id in overridesRef.current)) return;
      if (typeof window !== "undefined") {
        const label =
          id === CUSTOM_SCENE_ID
            ? "your custom scene"
            : `"${STRUCTURED_EXAMPLES[id]?.name ?? id}"`;
        const verb =
          id === CUSTOM_SCENE_ID ? "clear it" : "revert to the default prompt";
        const ok = window.confirm(
          `Discard your edits to ${label} and ${verb}?`,
        );
        if (!ok) return;
      }
      setOverrides((o) => {
        if (!(id in o)) return o;
        const next = { ...o };
        delete next[id];
        return next;
      });
      if (id === activeExampleId) {
        const pristine = STRUCTURED_EXAMPLES[id]?.scene;
        if (pristine) {
          const cloned = cloneScene(pristine);
          sceneRef.current = cloned;
          setScene(cloned);
          recomputePromptAndSend();
        } else {
          // Custom scene was running and is now gone; drop the active
          // scene state so the controller falls back to idle.
          sceneRef.current = null;
          setScene(null);
          setActiveExampleId(null);
        }
      }
    },
    [activeExampleId, recomputePromptAndSend],
  );

  // Apply the user's custom layered scene. Mirrors applyExample but
  // uses the user-picked pending image (or the already-sent custom
  // image) and the overrides[__custom__] scene.
  const applyCustomScene = useCallback(async () => {
    if (!isReady || isUploading) return;
    const customScene = overridesRef.current[CUSTOM_SCENE_ID];
    if (!customScene) {
      setErrorToast("Edit the custom prompt before applying.");
      return;
    }
    const composed = composePrompt(customScene, false, []).trim();
    if (!composed) {
      setErrorToast("Custom prompt is empty — add at least a base prose.");
      return;
    }
    if (!pendingImage && !sentImagePreview) {
      setErrorToast("Pick a starting image before applying.");
      return;
    }
    await applyScene({
      id: CUSTOM_SCENE_ID,
      scene: cloneScene(customScene),
      image: pendingImage
        ? {
            kind: "file",
            file: pendingImage.file,
            previewUrl: pendingImage.previewUrl,
          }
        : { kind: "keep" },
      errorLabel: "Failed to apply custom scene",
    });
  }, [isReady, isUploading, pendingImage, sentImagePreview, applyScene]);

  // ---- Render: sidebar (Quick Start + Custom) ----

  // When the inspector is open we swap the sidebar contents to the
  // inspector panel. That way the video on the right stays fully
  // visible — the user can see the prompt and the running video side
  // by side instead of having a modal cover everything.
  const sidebarContent =
    inspectorOpen && scene ? (
      <LivePromptInspector
        scene={scene}
        isMoving={moveL !== "idle" || moveLat !== "idle"}
        heldSlots={heldSlots}
        verticalPrompt={verticalPrompt}
        onClose={() => setInspectorOpen(false)}
      />
    ) : null;

  const sidebar = sidebarContent ?? (
    <div className="flex flex-col gap-4">
      {/* Examples — click to auto-send image + prompt + start */}
      {EXAMPLES.length > 0 && (
        <div className="flex flex-col gap-2">
          <span className="text-xs font-mono uppercase tracking-widest text-primary">
            Quick Start
          </span>
          <p className="text-[10px] text-white/40 leading-snug">
            Click an example to auto-load its image, prompt, and start
            generating. Click ✎ to preset / customize its layered prompt; your
            edits persist across re-clicks until you press ↺ to revert.
          </p>
          <div className="flex flex-col gap-2">
            {EXAMPLES.map((ex) => {
              const isActive = activeExampleId === ex.id;
              const isLoading = loadingExampleId === ex.id;
              const override = overrides[ex.id];
              const pristine = STRUCTURED_EXAMPLES[ex.id]?.scene;
              // Treat as "edited" only when the stored override actually
              // differs from the pristine constant — an override whose
              // content has reverted to default is just storage noise.
              const hasOverride = Boolean(
                override && pristine && !scenesEqual(override, pristine),
              );
              const applyDisabled =
                !isReady || isUploading || !!loadingExampleId;
              return (
                <div
                  key={ex.id}
                  className={cn(
                    "group flex items-stretch gap-1.5 rounded-lg border transition-all",
                    isActive
                      ? "border-amber-300/60 bg-amber-300/10"
                      : "border-white/10 bg-white/[0.03] hover:bg-white/[0.06] hover:border-white/20",
                  )}
                >
                  <button
                    type="button"
                    onClick={() => applyExample(ex)}
                    disabled={applyDisabled}
                    title="Apply this example (loads image and starts generation)"
                    className={cn(
                      "relative flex items-center gap-3 flex-1 min-w-0 p-2 text-left rounded-l-lg",
                      "disabled:opacity-50 disabled:cursor-not-allowed",
                    )}
                  >
                    <div className="relative shrink-0 w-24 h-14 rounded-md overflow-hidden border border-white/10">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={ex.image.src}
                        alt={ex.image.label}
                        className="w-full h-full object-cover"
                      />
                      {isLoading && (
                        <div className="absolute inset-0 bg-black/60 flex items-center justify-center">
                          <div className="w-4 h-4 border-2 border-amber-300 border-t-transparent rounded-full animate-spin" />
                        </div>
                      )}
                    </div>
                    <div className="flex flex-col gap-0.5 min-w-0 flex-1">
                      <div className="flex items-center gap-1.5 min-w-0">
                        <span className="font-mono text-sm text-white font-medium truncate">
                          {ex.name}
                        </span>
                        {hasOverride && (
                          <span
                            title="You've customized this prompt — your edits load whenever this example is applied. Press ↺ on the right to revert to the built-in default."
                            className="shrink-0 font-mono text-[9px] uppercase tracking-wider rounded bg-amber-300/15 text-amber-200 px-1.5 py-0.5 border border-amber-300/30"
                          >
                            edited
                          </span>
                        )}
                      </div>
                      {ex.description && (
                        <span className="font-mono text-[10px] text-white/50 leading-snug">
                          {ex.description}
                        </span>
                      )}
                    </div>
                    {isActive && (
                      <div className="shrink-0 w-2 h-2 rounded-full bg-amber-300" />
                    )}
                  </button>
                  {hasOverride && (
                    <button
                      type="button"
                      onClick={() => clearOverrideFor(ex.id)}
                      title={`Revert "${ex.name}" to the built-in default prompt (discards your edits)`}
                      className={cn(
                        "shrink-0 w-9 flex items-center justify-center border-l border-white/5",
                        "font-mono text-sm text-white/55 hover:text-red-300 hover:bg-white/[0.06] transition-colors",
                      )}
                      aria-label={`Reset ${ex.name} to default`}
                    >
                      ↺
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={() => openEditorFor(ex.id)}
                    title={`Edit ${ex.name} prompt (no image upload / no auto-start)`}
                    className={cn(
                      "shrink-0 w-9 flex items-center justify-center rounded-r-lg border-l border-white/5",
                      "font-mono text-sm text-white/55 hover:text-amber-200 hover:bg-white/[0.06] transition-colors",
                    )}
                  >
                    ✎
                  </button>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div className="border-t border-white/[0.06]" />

      {/* Custom layered scene — bring-your-own image + author your own
          layered prompt through the full editor. Use this when none of
          the built-in examples fit. */}
      {(() => {
        const customScene = overrides[CUSTOM_SCENE_ID];
        const customHasContent =
          !!customScene && !scenesEqual(customScene, emptyScene());
        const customComposed = customScene
          ? composePrompt(customScene, false, []).trim()
          : "";
        const customLoading = loadingExampleId === CUSTOM_SCENE_ID;
        const customIsActive = activeExampleId === CUSTOM_SCENE_ID;
        const canApplyCustom =
          isReady &&
          !isUploading &&
          !customLoading &&
          customComposed.length > 0 &&
          (!!pendingImage || !!sentImagePreview);
        const applyBlockerReason = (() => {
          if (!isReady) return "Not connected.";
          if (isUploading || customLoading) return "Uploading…";
          if (!customComposed)
            return "Edit the custom prompt first (default base prose must not be empty).";
          if (!pendingImage && !sentImagePreview)
            return "Pick a starting image first.";
          return undefined;
        })();
        return (
          <div className="flex flex-col gap-3">
            <span className="text-xs font-mono uppercase tracking-widest text-primary">
              Custom scene
            </span>
            <p className="text-[10px] text-white/40 leading-snug">
              Bring your own image, author a full layered prompt (base / camera
              / movement / events), then apply.
            </p>

            {/* Image picker */}
            <div className="flex items-center gap-2 flex-wrap">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) selectFile(f);
                }}
              />
              <Button
                size="sm"
                variant="outline"
                disabled={isUploading}
                onClick={() => fileInputRef.current?.click()}
                className="font-mono text-[10px]"
              >
                Choose image
              </Button>
              {pendingImage && (
                <div className="flex items-center gap-2">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={pendingImage.previewUrl}
                    alt={pendingImage.label}
                    className="h-8 w-14 object-cover rounded border border-amber-300/40"
                  />
                  <span className="font-mono text-[10px] text-amber-300/70 truncate max-w-[100px]">
                    {pendingImage.label}
                  </span>
                  <button
                    type="button"
                    onClick={clearPendingImage}
                    className="font-mono text-[10px] text-white/40 hover:text-white/80"
                  >
                    x
                  </button>
                </div>
              )}
              {!pendingImage && sentImagePreview && customIsActive && (
                <div className="flex items-center gap-2">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={sentImagePreview}
                    alt="current"
                    className="h-8 w-14 object-cover rounded border border-white/15"
                  />
                  <span className="font-mono text-[10px] text-white/40">
                    sent{imageInfo ? ` · ${imageInfo.w}x${imageInfo.h}` : ""}
                  </span>
                </div>
              )}
            </div>

            {/* Layered prompt editor entry */}
            <div className="flex items-center gap-2 flex-wrap">
              <Button
                size="sm"
                variant="outline"
                onClick={() => openEditorFor(CUSTOM_SCENE_ID)}
                className="font-mono text-[10px]"
              >
                ✎{" "}
                {customHasContent
                  ? "Edit custom prompt"
                  : "+ New custom prompt"}
              </Button>
              {customHasContent && (
                <>
                  <span className="font-mono text-[10px] text-white/50">
                    {customScene!.events.length} event
                    {customScene!.events.length === 1 ? "" : "s"}
                    {" · "}
                    {customComposed.length} chars
                  </span>
                  <button
                    type="button"
                    onClick={() => clearOverrideFor(CUSTOM_SCENE_ID)}
                    title="Clear your custom scene"
                    className="font-mono text-sm text-white/55 hover:text-red-300 transition-colors"
                  >
                    ↺
                  </button>
                </>
              )}
            </div>

            {/* Apply */}
            <div className="flex items-center gap-2 flex-wrap">
              <Button
                size="sm"
                onClick={applyCustomScene}
                disabled={!canApplyCustom}
                title={applyBlockerReason}
                className="font-mono text-[10px]"
              >
                {customLoading
                  ? "Applying…"
                  : customIsActive
                    ? "Re-apply custom scene"
                    : "Apply custom scene"}
              </Button>
              {customIsActive && (
                <span className="font-mono text-[10px] text-amber-300/70">
                  · running
                </span>
              )}
            </div>
          </div>
        );
      })()}

      <div className="border-t border-white/[0.06]" />

      {/* Generation state pills — visible regardless of which scene path
          (example or custom) is active. */}
      <div className="flex flex-col gap-3">
        <div className="flex items-center gap-2 flex-wrap">
          <div className="flex items-center gap-1.5">
            <span
              className={cn(
                "w-1.5 h-1.5 rounded-full",
                hasPrompt ? "bg-green-400" : "bg-white/20",
              )}
            />
            <span className="font-mono text-[10px] text-white/50">Prompt</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span
              className={cn(
                "w-1.5 h-1.5 rounded-full",
                hasImage ? "bg-green-400" : "bg-white/20",
              )}
            />
            <span className="font-mono text-[10px] text-white/50">Image</span>
          </div>
          <div className="flex-1" />
          <Button
            size="sm"
            disabled={!canStart}
            onClick={() => sendLifecycle("start")}
            title={startBlockerReason ?? undefined}
          >
            Start
          </Button>
        </div>
      </div>

      {errorToast && (
        <div className="rounded border border-red-500/30 bg-red-500/10 px-3 py-2 font-mono text-xs text-red-300">
          {errorToast}
        </div>
      )}

      {/* Advanced */}
      <div className="border-t border-white/[0.06] pt-3 flex flex-col gap-2">
        <div className="flex items-center gap-3 flex-wrap">
          <button
            type="button"
            onClick={() => setAdvancedOpen((v) => !v)}
            className="font-mono text-[10px] uppercase tracking-wider text-white/40 hover:text-white/70 transition-colors"
          >
            {advancedOpen ? "▾" : "▸"} Advanced
          </button>
        </div>

        {advancedOpen && (
          <div className="rounded border border-white/10 bg-white/[0.02] p-3 flex flex-col gap-3">
            <div className="flex items-center gap-3">
              <label className="font-mono text-[10px] uppercase tracking-wider text-white/50 w-28 shrink-0">
                Show prompt
              </label>
              <button
                type="button"
                role="switch"
                aria-checked={inspectorOpen}
                onClick={() => setInspectorOpen((v) => !v)}
                title="Show the composed prompt and per-layer breakdown for the active example"
                className={cn(
                  "relative h-5 w-9 shrink-0 rounded-full transition-colors",
                  inspectorOpen ? "bg-amber-300" : "bg-white/15",
                )}
              >
                <span
                  className={cn(
                    "absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-all",
                    inspectorOpen ? "left-[18px]" : "left-0.5",
                  )}
                />
              </button>
              <span className="font-mono text-[10px] text-white/40">
                {inspectorOpen ? "on — sidebar shows current prompt" : "off"}
              </span>
            </div>
            <div className="flex items-center gap-3">
              <label className="font-mono text-[10px] uppercase tracking-wider text-white/50 w-28 shrink-0">
                Rotation speed
              </label>
              <input
                type="range"
                min={0}
                max={30}
                step={0.5}
                value={rotationSpeed}
                onChange={(e) => pushRotationSpeed(Number(e.target.value))}
                className="flex-1 accent-amber-300"
              />
              <span className="font-mono text-xs text-white/70 w-20 text-right tabular-nums">
                {rotationSpeed.toFixed(1)}
                <span className="text-white/40"> °/step</span>
              </span>
            </div>
            <div className="flex items-center gap-3">
              <label className="font-mono text-[10px] uppercase tracking-wider text-white/50 w-28 shrink-0">
                Mouse sens
              </label>
              <input
                type="range"
                min={MOUSE_SENS_MIN}
                max={MOUSE_SENS_MAX}
                step={0.00005}
                value={mouseSens}
                onChange={(e) => setMouseSens(Number(e.target.value))}
                className="flex-1 accent-amber-300"
              />
              <span className="font-mono text-xs text-white/70 w-20 text-right tabular-nums">
                {((mouseSens * 180) / Math.PI).toFixed(3)}
                <span className="text-white/40"> °/px</span>
              </span>
            </div>
            <div className="flex items-center gap-3">
              <label className="font-mono text-[10px] uppercase tracking-wider text-white/50 w-28 shrink-0">
                Seed
              </label>
              <Input
                type="number"
                value={seed}
                onChange={(e) => pushSeed(Number(e.target.value))}
                className="w-24 font-mono text-xs"
              />
              <Button
                size="sm"
                variant="ghost"
                onClick={() => pushSeed(Math.floor(Math.random() * 1_000_000))}
              >
                Random
              </Button>
            </div>
            <div className="flex items-center gap-3">
              <label className="font-mono text-[10px] uppercase tracking-wider text-white/50 w-28 shrink-0">
                Camera pose
              </label>
              <span
                className={cn(
                  "font-mono text-[10px]",
                  cameraPoseActive ? "text-amber-300" : "text-white/30",
                )}
              >
                {cameraPoseActive
                  ? "active (pose layer driving rotation)"
                  : "inactive — keyboard only"}
              </span>
            </div>
            {/* DiT self-attention window override (backend set_attn_window). */}
            <div className="flex items-center gap-3">
              <label className="font-mono text-[10px] uppercase tracking-wider text-white/50 w-28 shrink-0">
                Attn window
              </label>
              <div className="flex gap-1">
                {(
                  [
                    [
                      "auto",
                      "Auto — motion-based: small window when still, full window when moving (default)",
                    ],
                    [
                      "small",
                      "Small — force the still (small) attention window always",
                    ],
                    [
                      "large",
                      "Large — force the moving (full) attention window always",
                    ],
                  ] as const
                ).map(([w, title]) => (
                  <button
                    key={w}
                    type="button"
                    disabled={!isReady}
                    onClick={() => pushAttnWindow(w)}
                    title={title}
                    className={cn(
                      "h-7 rounded border px-3 font-mono text-[11px] capitalize transition-colors disabled:opacity-30",
                      attnWindow === w
                        ? "border-amber-300/60 bg-amber-300/20 text-amber-200"
                        : "border-white/15 bg-white/5 text-white/60 hover:bg-white/10",
                    )}
                  >
                    {w}
                  </button>
                ))}
              </div>
            </div>
            {/* KV-cache/RoPE reset mode (off/auto/manual) + one-shot manual
                trigger (backend set_kv_cache_reset / trigger_kv_cache_reset). */}
            <div className="flex items-start gap-3">
              <label className="font-mono text-[10px] uppercase tracking-wider text-white/50 w-28 shrink-0 pt-1.5">
                KV reset
              </label>
              <div className="flex flex-wrap items-center gap-2">
                <div className="flex gap-1">
                  {(
                    [
                      [
                        "off",
                        "Off — no KV-cache reset; RoPE positions grow unbounded on long runs",
                      ],
                      [
                        "auto",
                        "Auto — periodic window reset (~every 27 chunks) + manual trigger (default)",
                      ],
                      [
                        "manual",
                        "Manual — no periodic reset; only the Reset-now button fires one",
                      ],
                    ] as const
                  ).map(([m, title]) => (
                    <button
                      key={m}
                      type="button"
                      disabled={!isReady}
                      onClick={() => pushKvCacheResetMode(m)}
                      title={title}
                      className={cn(
                        "h-7 rounded border px-3 font-mono text-[11px] capitalize transition-colors disabled:opacity-30",
                        kvCacheResetMode === m
                          ? "border-amber-300/60 bg-amber-300/20 text-amber-200"
                          : "border-white/15 bg-white/5 text-white/60 hover:bg-white/10",
                      )}
                    >
                      {m}
                    </button>
                  ))}
                </div>
                <Button
                  size="sm"
                  variant="ghost"
                  disabled={!isReady || kvCacheResetMode === "off"}
                  onClick={triggerKvCacheReset}
                  title="Force a one-shot KV-cache reset on the next chunk (e.g. at a scene cut). Available in auto and manual modes."
                >
                  Reset now
                </Button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Layered-scene editor — full-screen modal overlaying the page.
          Lives here in the sidebar tree because it's position: fixed;
          visually it covers the whole viewport. Reads / writes the
          per-example override store so edits persist across re-clicks. */}
      {editingExampleId &&
        editingScene &&
        (() => {
          const isCustom = editingExampleId === CUSTOM_SCENE_ID;
          // Only offer "Reset to example" when the current scene actually
          // differs from the pristine constant; otherwise there's nothing
          // to reset.
          const pristine = isCustom
            ? undefined
            : STRUCTURED_EXAMPLES[editingExampleId]?.scene;
          const canReset =
            pristine != null && !scenesEqual(editingScene, pristine);
          const title = isCustom
            ? "Edit · Custom scene"
            : `Edit · ${STRUCTURED_EXAMPLES[editingExampleId]?.name ?? editingExampleId}`;
          const subtitle = isCustom
            ? "Author a fully custom layered prompt. Apply it from the Custom card on the right."
            : editingExampleId === activeExampleId
              ? "Editing the currently-running scene — changes apply live."
              : "Pre-editing this scene. Click the example card to apply your edits.";
          return (
            <LayeredSceneEditor
              title={title}
              subtitle={subtitle}
              scene={editingScene}
              pristine={pristine}
              onChange={handleSceneChange}
              onReset={canReset ? resetEditingExample : undefined}
              resetLabel="Reset to example"
              onClose={closeEditor}
            />
          );
        })()}
    </div>
  );

  // ---- Render: bottom controls (hold chips, movement, look) ----

  // Mouse-signal HUD geometry: arrow from circle center in the direction of
  // recent mouse motion, length ∝ strength (clamped to the circle radius).
  const HUD = 72,
    HUD_C = HUD / 2,
    HUD_R = 28;
  const _vx = mouseViz.x * 0.5,
    _vy = mouseViz.y * 0.5;
  const _mag = Math.hypot(_vx, _vy);
  const _s = _mag > HUD_R ? HUD_R / _mag : 1;
  // Arrow-key look direction (svg coords: up = -y). Mirrored into the Mouse HUD
  // when mouse-look is off, so arrows ↔ Mouse read as one "look" control —
  // exactly like WASD ↔ joystick below.
  const _arrowX = (lookH === "right" ? 1 : 0) - (lookH === "left" ? 1 : 0);
  const _arrowY = (lookV === "down" ? 1 : 0) - (lookV === "up" ? 1 : 0);
  const _arrowActive = _arrowX !== 0 || _arrowY !== 0;
  let hudEx = HUD_C,
    hudEy = HUD_C,
    hudActive = false;
  if (mouseLook) {
    hudEx = HUD_C + _vx * _s;
    hudEy = HUD_C + _vy * _s;
    hudActive = _mag > 0.5;
  } else if (_arrowActive) {
    const _am = (HUD_R * 0.9) / Math.hypot(_arrowX, _arrowY);
    hudEx = HUD_C + _arrowX * _am;
    hudEy = HUD_C + _arrowY * _am;
    hudActive = true;
  }

  // Joystick knob: show the drag vector if dragging, else mirror the WASD
  // state (so the joystick and WASD read as the same "movement" control).
  const _wasdX =
    (moveLat === "strafe_right" ? 1 : 0) - (moveLat === "strafe_left" ? 1 : 0);
  const _wasdY = (moveL === "back" ? 1 : 0) - (moveL === "forward" ? 1 : 0);
  const joyDragging = joy.x !== 0 || joy.y !== 0;
  const joyDispX = joyDragging ? joy.x : _wasdX;
  const joyDispY = joyDragging ? joy.y : _wasdY;
  const joyLit = joyDragging || _wasdX !== 0 || _wasdY !== 0;

  const controls = (
    <div className="flex flex-col gap-3">
      {/* Status telemetry */}
      {(tspSize !== null || (isGenerating && chunkNum > 0) || isReady) && (
        <div className="flex items-center gap-3 flex-wrap">
          {tspSize !== null && (
            <span className="font-mono text-[10px] text-white/40">
              workers: {tspSize}
            </span>
          )}
          {isGenerating && chunkNum > 0 && (
            <span className="font-mono text-[10px] text-white/40">
              chunk {chunkIndex + 1}/{chunkNum}
            </span>
          )}
          {isReady && (
            <span className="font-mono text-[10px] text-amber-300/70">
              action: {activeAction}
            </span>
          )}
          <div className="flex-1" />
          {(isGenerating || isPaused) && (
            <Button
              size="sm"
              variant="secondary"
              onClick={() => sendLifecycle(isPaused ? "resume" : "pause")}
              disabled={!canPauseResume}
              className="font-mono text-[10px]"
            >
              {isPaused ? "Resume" : "Pause"}
            </Button>
          )}
        </div>
      )}

      {/* Hold-key event chips, derived from the active scene's events */}
      {scene && scene.events.length > 0 && (
        <div className="flex flex-col gap-1">
          <span className="font-mono text-[9px] text-white/40 uppercase tracking-wider">
            Hold (keys 1-{scene.events.length} — reverts on release)
          </span>
          <div className="flex flex-wrap gap-1">
            {scene.events.slice(0, MAX_EVENTS).map((event, slot) => {
              const detailEmpty =
                typeof event.detail === "string"
                  ? !event.detail.trim()
                  : !event.detail.static.trim() && !event.detail.dynamic.trim();
              return (
                <HoldChip
                  key={slot}
                  slot={slot}
                  name={event.name}
                  empty={detailEmpty && !event.name.trim()}
                  pressed={heldSlots.includes(slot)}
                  disabled={!isReady}
                  onPress={() => holdPress(slot)}
                  onRelease={() => holdRelease(slot)}
                />
              );
            })}
          </div>
        </div>
      )}

      {/* Move (WASD) + Joystick + Look + Mouse-signal HUD */}
      <div className="grid grid-cols-4 gap-3">
        {/* Move (WASD) + the camera-pose action buttons right beside it */}
        <div className="flex flex-col items-center gap-1.5">
          <span className="font-mono text-[9px] uppercase tracking-wider text-white/40">
            Move (WASD)
          </span>
          {/* WASD pad (with Q/E roll above, left & right) + Space/C stacked to the right.
              items-end so the Space/C stack aligns to the WASD pad (W + ASD), not Q/E. */}
          <div className="flex items-end gap-2">
            <div className="flex flex-col items-center gap-0.5">
              {/* Top row: Q | W | E — Q/E fill the gaps either side of W */}
              <div className="flex gap-0.5">
                <HoldBtn
                  label="↺ Q"
                  lit={rollDir === -1}
                  disabled={!isReady}
                  className="h-10 w-10"
                  title="Roll left (Q)"
                  onDown={() => setRoll(-1)}
                  onUp={() => setRoll(0)}
                />
                <PadButton
                  label="W"
                  pressed={moveL === "forward"}
                  disabled={!isReady}
                  onPress={() => onMoveLPress("forward")}
                  onRelease={() => onMoveLRelease("forward")}
                />
                <HoldBtn
                  label="↻ E"
                  lit={rollDir === 1}
                  disabled={!isReady}
                  className="h-10 w-10"
                  title="Roll right (E)"
                  onDown={() => setRoll(1)}
                  onUp={() => setRoll(0)}
                />
              </div>
              {/* Bottom row: A | S | D */}
              <div className="flex gap-0.5">
                <PadButton
                  label="A"
                  pressed={moveLat === "strafe_left"}
                  disabled={!isReady}
                  onPress={() => onMoveLatPress("strafe_left")}
                  onRelease={() => onMoveLatRelease("strafe_left")}
                />
                <PadButton
                  label="S"
                  pressed={moveL === "back"}
                  disabled={!isReady}
                  onPress={() => onMoveLPress("back")}
                  onRelease={() => onMoveLRelease("back")}
                />
                <PadButton
                  label="D"
                  pressed={moveLat === "strafe_right"}
                  disabled={!isReady}
                  onPress={() => onMoveLatPress("strafe_right")}
                  onRelease={() => onMoveLatRelease("strafe_right")}
                />
              </div>
            </div>
            {/* Space jump (up) / C crouch (down) — bigger, stacked to the right of WASD.
                h-10 + gap-0.5 = the WASD pad's height, so the stack's top & bottom
                edges line up with the W and A/S/D rows. */}
            <div className="flex flex-col gap-0.5">
              <HoldBtn
                label="⤒ Space"
                lit={jumpLit}
                disabled={!isReady}
                className="h-10 w-16 text-[11px]"
                title="Jump (Space) — see the Jump mode switch"
                onDown={onJumpDown}
                onUp={onJumpUp}
              />
              <HoldBtn
                label="⤓ C"
                lit={vertDir < 0}
                disabled={!isReady}
                className="h-10 w-16 text-[11px]"
                title="Crouch (C) — see the Crouch mode switch below"
                onDown={() => setVert(-1)}
                onUp={() => setVert(0)}
              />
            </div>
            {/* Jump/Crouch mode switches + charge-level editor live in their own
                full-width row below the grid (see "Vertical" section). */}
          </div>
        </div>

        {/* Charge-level grid editor popup: click cells to cycle each latent's
            up (↑) / down (↓) / still (·) state. Persists automatically. */}
        {editingLevel !== null && (
          <div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
            onClick={() => setEditingLevel(null)}
          >
            <div
              className="w-full max-w-md rounded-xl border border-white/15 bg-neutral-950 p-5 shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between">
                <h3 className="font-mono text-sm text-white">
                  Jump level {editingLevel} · {editingLevel} chunk
                  {editingLevel > 1 ? "s" : ""} ({editingLevel * CHUNK_LATENTS}{" "}
                  latents)
                </h3>
                <button
                  type="button"
                  onClick={() => setEditingLevel(null)}
                  className="h-7 w-7 rounded font-mono text-xs text-white/60 hover:bg-white/10"
                >
                  ✕
                </button>
              </div>
              <p className="mt-1 mb-4 font-mono text-[11px] leading-relaxed text-white/50">
                One cell per latent (3 per chunk), played left→right,
                top→bottom. Click a cell to cycle{" "}
                <span className="text-amber-200">↑ up</span> →{" "}
                <span className="text-sky-200">↓ down</span> →{" "}
                <span className="text-white/40">· still</span>. Stills are the
                pause / hang; put as many as you want. Saved automatically.
              </p>
              <div className="flex flex-col gap-2">
                {Array.from({ length: editingLevel }, (_, c) => (
                  <div key={c} className="flex items-center gap-3">
                    <span className="w-14 font-mono text-[9px] uppercase tracking-wider text-white/30">
                      chunk {c + 1}
                    </span>
                    <div className="flex gap-2">
                      {Array.from({ length: CHUNK_LATENTS }, (_, j) => {
                        const idx = c * CHUNK_LATENTS + j;
                        const v = chargePatterns[editingLevel - 1][idx];
                        return (
                          <button
                            key={j}
                            type="button"
                            onClick={() => cycleChargeCell(editingLevel, idx)}
                            title={v === 1 ? "up" : v === -1 ? "down" : "still"}
                            className={cn(
                              "flex h-12 w-12 items-center justify-center rounded-md border font-mono text-xl transition-colors",
                              v === 1
                                ? "bg-amber-300/20 border-amber-300/60 text-amber-200"
                                : v === -1
                                  ? "bg-sky-400/20 border-sky-400/60 text-sky-200"
                                  : "bg-white/5 border-white/15 text-white/40",
                            )}
                          >
                            {v === 1 ? "↑" : v === -1 ? "↓" : "·"}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>
              <div className="mt-5 flex items-center justify-between">
                <button
                  type="button"
                  onClick={() => resetChargeLevel(editingLevel)}
                  className="rounded border border-white/15 px-3 py-1.5 font-mono text-[11px] text-white/60 hover:bg-white/10"
                >
                  Reset to default
                </button>
                <button
                  type="button"
                  onClick={() => setEditingLevel(null)}
                  className="rounded border border-amber-300/40 bg-amber-300/15 px-3 py-1.5 font-mono text-[11px] text-amber-200 hover:bg-amber-300/25"
                >
                  Done
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Crouch dip editor popup: the single dip chunk's per-latent pattern. */}
        {editingCrouch && (
          <div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
            onClick={() => setEditingCrouch(false)}
          >
            <div
              className="w-full max-w-md rounded-xl border border-white/15 bg-neutral-950 p-5 shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between">
                <h3 className="font-mono text-sm text-white">
                  Crouch — press &amp; release chunks
                </h3>
                <button
                  type="button"
                  onClick={() => setEditingCrouch(false)}
                  className="h-7 w-7 rounded font-mono text-xs text-white/60 hover:bg-white/10"
                >
                  ✕
                </button>
              </div>
              <p className="mt-1 mb-4 font-mono text-[11px] leading-relaxed text-white/50">
                Two one-chunk dips: <strong>press</strong> fires on C-down,{" "}
                <strong>release</strong> fires on C-up (standing back up). One
                cell per latent — click to cycle{" "}
                <span className="text-amber-200">↑ up</span> →{" "}
                <span className="text-sky-200">↓ down</span> →{" "}
                <span className="text-white/40">· still</span>. Added on top of
                forward. Saved automatically.
              </p>
              <div className="flex flex-col gap-3">
                {(["press", "release"] as const).map((phase) => (
                  <div key={phase} className="flex items-center gap-3">
                    <span className="w-16 font-mono text-[9px] uppercase tracking-wider text-white/30">
                      {phase} {phase === "press" ? "(↓)" : "(↑)"}
                    </span>
                    <div className="flex gap-2">
                      {Array.from({ length: CHUNK_LATENTS }, (_, j) => {
                        const v = crouchPatterns[phase][j];
                        return (
                          <button
                            key={j}
                            type="button"
                            onClick={() => cycleCrouchCell(phase, j)}
                            title={v === 1 ? "up" : v === -1 ? "down" : "still"}
                            className={cn(
                              "flex h-12 w-12 items-center justify-center rounded-md border font-mono text-xl transition-colors",
                              v === 1
                                ? "bg-amber-300/20 border-amber-300/60 text-amber-200"
                                : v === -1
                                  ? "bg-sky-400/20 border-sky-400/60 text-sky-200"
                                  : "bg-white/5 border-white/15 text-white/40",
                            )}
                          >
                            {v === 1 ? "↑" : v === -1 ? "↓" : "·"}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>
              <div className="mt-5 flex items-center justify-between">
                <button
                  type="button"
                  onClick={resetCrouchPatterns}
                  className="rounded border border-white/15 px-3 py-1.5 font-mono text-[11px] text-white/60 hover:bg-white/10"
                >
                  Reset to default
                </button>
                <button
                  type="button"
                  onClick={() => setEditingCrouch(false)}
                  className="rounded border border-amber-300/40 bg-amber-300/15 px-3 py-1.5 font-mono text-[11px] text-amber-200 hover:bg-amber-300/25"
                >
                  Done
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Drag joystick: continuous translation; mirrors WASD when not dragging */}
        <div className="flex flex-col items-center gap-1.5">
          <span className="font-mono text-[9px] uppercase tracking-wider text-white/40">
            Joystick
          </span>
          <div
            ref={joyAreaRef}
            onPointerDown={(e) => {
              if (!isReady) return;
              e.currentTarget.setPointerCapture(e.pointerId);
              onJoyPointer(e, "down");
            }}
            onPointerMove={(e) => {
              if (e.buttons !== 0) onJoyPointer(e, "move");
            }}
            onPointerUp={(e) => onJoyPointer(e, "up")}
            onPointerCancel={(e) => onJoyPointer(e, "up")}
            title="Drag for continuous movement (up = forward, down = back, left/right = strafe). Adds to WASD; mirrors WASD when idle."
            className={cn(
              "relative rounded-full border touch-none select-none",
              isReady
                ? "cursor-grab active:cursor-grabbing border-white/15 bg-white/[0.03]"
                : "opacity-30 border-white/10",
            )}
            style={{ width: HUD, height: HUD }}
          >
            <div className="absolute left-1/2 top-0 h-full w-px -translate-x-1/2 bg-white/8" />
            <div className="absolute top-1/2 left-0 w-full h-px -translate-y-1/2 bg-white/8" />
            <div
              className={cn(
                "absolute h-4 w-4 rounded-full -translate-x-1/2 -translate-y-1/2 transition-all",
                joyLit ? "bg-amber-300" : "bg-white/40",
              )}
              style={{
                left: `calc(50% + ${joyDispX * HUD_R}px)`,
                top: `calc(50% + ${joyDispY * HUD_R}px)`,
              }}
            />
          </div>
        </div>

        <div className="flex flex-col items-center gap-1.5">
          <span className="font-mono text-[9px] uppercase tracking-wider text-white/40">
            Look (Arrows)
          </span>
          <div className="flex flex-col items-center gap-0.5">
            <PadButton
              label="↑"
              pressed={lookV === "up"}
              disabled={!isReady}
              onPress={() => pushLookV("up")}
              onRelease={() => pushLookV("idle")}
            />
            <div className="flex gap-0.5">
              <PadButton
                label="←"
                pressed={lookH === "left"}
                disabled={!isReady}
                onPress={() => pushLookH("left")}
                onRelease={() => pushLookH("idle")}
              />
              <PadButton
                label="↓"
                pressed={lookV === "down"}
                disabled={!isReady}
                onPress={() => pushLookV("down")}
                onRelease={() => pushLookV("idle")}
              />
              <PadButton
                label="→"
                pressed={lookH === "right"}
                disabled={!isReady}
                onPress={() => pushLookH("right")}
                onRelease={() => pushLookH("idle")}
              />
            </div>
          </div>
        </div>

        {/* Mouse-look: click the circle to engage (pointer lock); Esc/M to release.
            Arrow shows live mouse signal direction + strength. */}
        <div className="flex flex-col items-center gap-1.5">
          <span className="font-mono text-[9px] uppercase tracking-wider text-white/40">
            Mouse {mouseLook ? "(Esc/M)" : "(click)"}
          </span>
          <button
            type="button"
            disabled={!isReady}
            onClick={toggleMouseLook}
            title={
              mouseLook
                ? "Mouse-look ON — move the mouse to rotate (yaw+pitch). Esc, M, or an arrow key to release."
                : "Click to engage mouse-look (pointer lock). Move the mouse to rotate; Esc/M to release."
            }
            className={cn(
              "rounded-full transition-transform disabled:opacity-30",
              isReady && "cursor-pointer hover:scale-105",
              mouseLook && "ring-2 ring-amber-300/60",
            )}
          >
            <svg width={HUD} height={HUD} className="shrink-0 block">
              <circle
                cx={HUD_C}
                cy={HUD_C}
                r={HUD_R}
                fill={
                  mouseLook ? "rgba(252,211,77,0.06)" : "rgba(255,255,255,0.02)"
                }
                stroke={
                  mouseLook ? "rgba(252,211,77,0.45)" : "rgba(255,255,255,0.15)"
                }
                strokeWidth="1"
              />
              <line
                x1={HUD_C}
                y1={HUD_C - HUD_R}
                x2={HUD_C}
                y2={HUD_C + HUD_R}
                stroke="rgba(255,255,255,0.08)"
                strokeWidth="1"
              />
              <line
                x1={HUD_C - HUD_R}
                y1={HUD_C}
                x2={HUD_C + HUD_R}
                y2={HUD_C}
                stroke="rgba(255,255,255,0.08)"
                strokeWidth="1"
              />
              {hudActive && (
                <>
                  <line
                    x1={HUD_C}
                    y1={HUD_C}
                    x2={hudEx}
                    y2={hudEy}
                    stroke="rgb(252,211,77)"
                    strokeWidth="2"
                    strokeLinecap="round"
                  />
                  <circle cx={hudEx} cy={hudEy} r="3" fill="rgb(252,211,77)" />
                </>
              )}
              <circle
                cx={HUD_C}
                cy={HUD_C}
                r="2"
                fill={
                  mouseLook ? "rgba(252,211,77,0.8)" : "rgba(255,255,255,0.3)"
                }
              />
            </svg>
          </button>
        </div>
      </div>

      {/* Vertical controls — jump + crouch mode switches and the charge-level
          editor. Full-width row so the buttons are comfortably clickable. */}
      <div className="flex flex-wrap items-start gap-x-8 gap-y-3 border-t border-white/[0.06] pt-3">
        <div className="flex flex-col gap-1.5">
          <span className="font-mono text-[9px] uppercase tracking-wider text-white/40">
            Jump (Space)
          </span>
          <div className="flex gap-1">
            {(
              [
                ["hold", "Hold — translate up while held (no descent)"],
                ["prompt", "Prompt — only append the scene's jump prompt"],
                [
                  "charge",
                  "Charge — hold to charge a level, release to fire that level's arc",
                ],
              ] as const
            ).map(([m, title]) => (
              <button
                key={m}
                type="button"
                disabled={!isReady}
                onClick={() => changeJumpMode(m)}
                title={title}
                className={cn(
                  "h-7 rounded border px-3 font-mono text-[11px] capitalize transition-colors disabled:opacity-30",
                  jumpMode === m
                    ? "border-amber-300/60 bg-amber-300/20 text-amber-200"
                    : "border-white/15 bg-white/5 text-white/60 hover:bg-white/10",
                )}
              >
                {m}
              </button>
            ))}
          </div>
        </div>

        {jumpMode === "charge" && (
          <div className="flex flex-col gap-1.5">
            <span className="font-mono text-[9px] uppercase tracking-wider text-white/40">
              Charge levels — click to edit
            </span>
            <div className="flex gap-1.5">
              {Array.from({ length: NUM_CHARGE_LEVELS }, (_, i) => {
                const level = i + 1;
                return (
                  <button
                    key={i}
                    type="button"
                    onClick={() => setEditingLevel(level)}
                    title={`Level ${level} — ${level} chunk${level > 1 ? "s" : ""}; click to edit its per-latent up/down/still pattern`}
                    className={cn(
                      "flex h-9 w-14 flex-col items-center justify-center rounded border font-mono transition-colors",
                      i < chargeLevel
                        ? "border-amber-300/60 bg-amber-300/25 text-amber-100"
                        : "border-white/15 bg-white/5 text-white/70 hover:bg-white/10",
                    )}
                  >
                    <span className="text-[13px] leading-none">{level}</span>
                    <span className="text-[8px] leading-none text-white/45">
                      ✎ edit
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        <div className="flex flex-col gap-1.5">
          <span className="font-mono text-[9px] uppercase tracking-wider text-white/40">
            Crouch (C)
          </span>
          <div className="flex items-center gap-1">
            {(
              [
                [
                  "hold",
                  "Hold — translate straight down for as long as held (no return)",
                ],
                [
                  "prompt",
                  "Prompt — inject the scene's crouch line, no camera motion",
                ],
                [
                  "camera",
                  "Camera — a one-shot downward dip (+ the crouch line) while held",
                ],
              ] as const
            ).map(([m, title]) => (
              <button
                key={m}
                type="button"
                disabled={!isReady}
                onClick={() => changeCrouchMode(m)}
                title={title}
                className={cn(
                  "h-7 rounded border px-3 font-mono text-[11px] capitalize transition-colors disabled:opacity-30",
                  crouchMode === m
                    ? "border-amber-300/60 bg-amber-300/20 text-amber-200"
                    : "border-white/15 bg-white/5 text-white/60 hover:bg-white/10",
                )}
              >
                {m}
              </button>
            ))}
            {crouchMode === "camera" && (
              <button
                type="button"
                onClick={() => setEditingCrouch(true)}
                title="Edit the crouch dip — which of this chunk's latents are down / still"
                className="ml-1 flex h-7 items-center rounded border border-white/15 bg-white/5 px-2 font-mono text-[10px] text-white/70 hover:bg-white/10"
              >
                ✎ dip
              </button>
            )}
          </div>
        </div>

        <div className="flex flex-col gap-1.5">
          <span className="font-mono text-[9px] uppercase tracking-wider text-white/40">
            Orbit radius (O)
          </span>
          <div className="flex items-center gap-1.5">
            <Input
              type="number"
              inputMode="decimal"
              step={ORBIT_RADIUS_STEP}
              min={0}
              value={orbitRadius}
              disabled={!isReady}
              onChange={(e) => {
                const v = Number(e.target.value);
                setOrbitRadius(Number.isFinite(v) ? Math.max(0, v) : 0);
              }}
              className={cn(
                "h-7 w-24 font-mono text-xs tabular-nums",
                "[appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none",
                orbitRadius > 0 ? "text-amber-200" : "text-white/60",
              )}
              title="Orbit radius — 0 = rotate in place (normal). Type any value; bigger = wider/farther. Press O to mute / un-mute."
            />
            <div className="flex flex-col">
              <button
                type="button"
                disabled={!isReady}
                onClick={() =>
                  setOrbitRadius((r) => Math.max(0, r + ORBIT_RADIUS_STEP))
                }
                className="flex h-3.5 w-5 items-center justify-center rounded-t border border-b-0 border-white/15 bg-white/5 text-[8px] leading-none text-white/60 hover:bg-white/10 disabled:opacity-30"
                title="Increase orbit radius"
              >
                &#9650;
              </button>
              <button
                type="button"
                disabled={!isReady}
                onClick={() =>
                  setOrbitRadius((r) => Math.max(0, r - ORBIT_RADIUS_STEP))
                }
                className="flex h-3.5 w-5 items-center justify-center rounded-b border border-white/15 bg-white/5 text-[8px] leading-none text-white/60 hover:bg-white/10 disabled:opacity-30"
                title="Decrease orbit radius"
              >
                &#9660;
              </button>
            </div>
          </div>
        </div>
      </div>

      <p className="font-mono text-[9px] text-white/35 leading-snug">
        WASD / joystick = move · arrows = look · Q/E = roll · Space = jump / C =
        crouch · click the Mouse circle for free-look (Esc/M to release).
        <span className="text-white/50">Orbit radius</span> = while looking
        (mouse or arrows), circle a point R ahead instead of turning in place; R
        = 0 is normal rotation, press <span className="text-white/50">O</span>{" "}
        to mute / un-mute. Jump mode:{" "}
        <span className="text-white/50">Hold</span> = up while held ·
        <span className="text-white/50"> Prompt</span> = prompt only ·
        <span className="text-white/50"> Charge</span> = hold to charge a level,
        release for that level&apos;s up→down arc (click a level to edit its
        per-latent pattern). Crouch (all modes inject the scene&apos;s crouch
        line):
        <span className="text-white/50"> Hold</span> = straight down while held
        ·<span className="text-white/50"> Prompt</span> = line only ·
        <span className="text-white/50"> Camera</span> = a one-shot editable dip
        (✎) plus the line.
      </p>
    </div>
  );

  return { sidebar, controls };
}
