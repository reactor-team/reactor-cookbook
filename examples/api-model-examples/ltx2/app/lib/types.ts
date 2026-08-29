// Pre-connect fallbacks for the pace bounds. Once a snapshot arrives the
// deployment's own `wpm_min` / `wpm_max` win — they are configurable per
// deployment, so treat these as placeholders, not as the model's limits.
export const DEFAULT_MIN_WPM = 80;
export const DEFAULT_MAX_WPM = 220;

// The UI's projection of the model's session, reduced from the `state_update`
// snapshot the model broadcasts on connect and after every observable change.
// The model is the source of truth: this app never infers session state from
// its own button clicks.
export interface Ltx2UiState {
  /** The words the avatar speaks, or null when none is set. */
  script: string | null;
  /** Scene description conditioning how the avatar looks and moves. */
  prompt: string;
  /** An avatar image is set, so `start` has a face to anchor to. */
  hasAvatarImage: boolean;
  /** Speaking pace in words per minute. */
  wpm: number;
  /** Pinned run length, or 0 when the length derives from the script. */
  durationSeconds: number;
  /** Length the next run will actually have. Zero until a script is set. */
  effectiveSeconds: number;
  /** Seed for the next take. Same inputs + same seed reproduce a take. */
  seed: number;
  /** An avatar image and a script are set, so `start` is valid. */
  ready: boolean;
  /**
   * A run is in flight. Conditions stay editable — the model queues each
   * change for the next take and reports it in {@link queuedChanges}.
   */
  generating: boolean;
  /**
   * Commands the session would accept right now, straight from the model.
   * Authoritative: anything absent would come back as `command_error`. Ask
   * `validCommands()` in lib/machine.ts rather than reading this directly.
   */
  validCommands: string[];
  /**
   * Condition fields changed during the run in flight, in the order first
   * changed. Their values are already the ones in this snapshot; they take
   * effect on the next take. Empty when nothing is queued or nothing is
   * running.
   */
  queuedChanges: string[];
  /** Lowest pace this deployment accepts via `set_wpm`. */
  wpmMin: number;
  /** Highest pace this deployment accepts via `set_wpm`. */
  wpmMax: number;
  /** The output stream is frozen mid-run; `resume` continues instantly. */
  paused: boolean;
  /** The last run ended and the model is idle. */
  finished: boolean;
  /** Index of the most recent window streamed, or -1 before the first. */
  windowIndex: number;
  /** Windows the current run will stream; zero before one starts. */
  totalWindows: number;
  /** Seconds of A/V sent so far this run. */
  secondsSent: number;
}

export const DEFAULT_UI_STATE: Ltx2UiState = {
  script: null,
  prompt: "A single person speaks directly to the camera in a natural tone.",
  hasAvatarImage: false,
  wpm: 140,
  durationSeconds: 0,
  effectiveSeconds: 0,
  seed: 1717,
  ready: false,
  generating: false,
  validCommands: [],
  queuedChanges: [],
  wpmMin: DEFAULT_MIN_WPM,
  wpmMax: DEFAULT_MAX_WPM,
  paused: false,
  finished: false,
  windowIndex: -1,
  totalWindows: 0,
  secondsSent: 0,
};

/**
 * The conditions the take panel edits. Each maps to one `set_*` command, and
 * all of them stay editable during a run: the model accepts the change and
 * queues it for the next take, reporting it in `state_update.queued_changes`.
 * The panel therefore sends every edit straight to the wire and reads the
 * queue back off the snapshot — it holds no pending-edit state of its own.
 */
export interface TakeFields {
  script: string;
  prompt: string;
  wpm: number;
  duration_seconds: number;
  seed: number;
}

/**
 * One edit to commit, as a discriminated union rather than a
 * `(field, value)` pair.
 *
 * The union is what lets the app shell dispatch to the right typed method
 * with the value already narrowed — `setWpm({ wpm })` will not compile
 * against a string. A generic `<K extends keyof TakeFields>(field: K, value:
 * TakeFields[K])` reads tidier at the call site but does not narrow inside a
 * switch, which is exactly where the type safety has to hold.
 */
export type TakeEdit = {
  [K in keyof TakeFields]: { field: K; value: TakeFields[K] };
}[keyof TakeFields];

/**
 * The model's own name for each take field, as it appears in
 * `state_update.queued_changes`. Used only to match a field against the
 * queue — commands are sent through the typed methods on `useLtx2()`,
 * never by name.
 */
export const FIELD_WIRE_NAME: Record<keyof TakeFields, string> = {
  script: "script",
  prompt: "prompt",
  wpm: "wpm",
  duration_seconds: "duration_seconds",
  seed: "seed",
};

// Text caps, enforced by the schema's `maxLength` on each command.
export const MAX_SCRIPT_CHARS = 4000;
export const MAX_PROMPT_CHARS = 800;

/** The generation canvas. Uploads are cropped to this before they go up. */
export const FRAME_WIDTH = 640;
export const FRAME_HEIGHT = 352;
