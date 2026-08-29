// The structured prompt model behind the example scenes, plus the pure
// composePrompt() function that flattens a scene to the prose string sent
// via set_prompt. Edits made through the in-page editor are persisted to
// localStorage and survive reloads; ↺ on a card wipes its override.
//
// The default examples (see lingbot-cases-examples.ts) are converted from
// the "lingbot-cases" corpus, one StructuredExample per case, each with
// hold-key events (keys 1..9) plus optional jump/crouch/stand prompts.

import { LINGBOT_CASES_EXAMPLE_LIST } from "@/lib/lingbot-cases-examples";

// A starting-image thumbnail for an example card. URLs are served out of
// this app's public/ directory.
export interface ImagePreset {
  label: string;
  src: string;
}

// Structured / orthogonal authoring form.
//
// Each scene has three layers and a list of events:
//   - base:     world identity (subject, environment, style)
//   - camera:   { static, dynamic } framing — selected by WASD state
//   - movement: { static, dynamic } subject motion — selected by WASD
//   - events:   detail clauses bound to hold-keys 1..9
//
// Each layer is a registry keyed by version id, with a required
// "default" key. Alternate versions are just additional keys in the
// same registry — there is no separate "default field + side
// registry" split. Example shape:
//
//   base: {
//     default: "...subject + environment + style...",
//     portal_world: "...alternate world this scene can transition to...",
//     empty: "",
//   }
//
// Events independently choose a version per layer via `baseVersion` /
// `cameraVersion` / `movementVersion`; omitted means "default".
//
// Compatibility / stacking rule. Events stack only if their
// `baseVersion` matches the active base version (events written
// against different worlds can't share prose coherently). The active
// base version is the most-recently-pressed held event's
// `baseVersion`; events from other versions are suppressed while held.
// Within the compatible set, the active camera / movement versions are
// taken from the most-recently-pressed compatible event's choices on
// those layers — letting a single big event swap to a more compact
// camera/movement to free token budget for its detail without forcing
// peers off the stack.
//
// Mutex / scene-replace pattern. To author an event whose detail
// already carries its own subject + environment + camera (the old
// mutex_replace behaviour), register an "empty" version on each
// layer (`base.empty = ""`, `camera.empty = { static: "", dynamic:
// "" }`, `movement.empty = { static: "", dynamic: "" }`) and tag the
// event with all three. The event's standalone prose becomes the
// only content sent.
//
// camera.<v>.dynamic = strict rear-view tracking (look-input turns
// the subject's heading); camera.<v>.static = subject locked in place
// and the camera orbits around it (look-input swings the camera).
//
// composePrompt() concatenates the active selection into a single
// natural-language string. The encoder only ever sees prose; the structure
// is authoring-time only.

export type ShotVariant = { static: string; dynamic: string };
export type EventVariant = string | ShotVariant;

// Each layer registry must include a "default" key plus any number of
// alternate versions. `LayerRegistry<T>` enforces the default; other
// keys are arbitrary identifiers chosen by the author.
export type LayerRegistry<T> = { default: T } & Record<string, T>;

// The required default key for every layer registry; events that omit
// a *Version field implicitly select this version.
export const DEFAULT_LAYER_VERSION = "default";

export interface NamedEvent {
  name: string;
  // Which layer versions this event is authored against. Omit (or set
  // to "default") to compose against the layer's default version;
  // otherwise must be a key in the corresponding layer registry.
  baseVersion?: string;
  cameraVersion?: string;
  movementVersion?: string;
  detail: EventVariant;
}

export interface StructuredScene {
  base: LayerRegistry<string>;
  camera: LayerRegistry<ShotVariant>;
  movement: LayerRegistry<ShotVariant>;
  events: NamedEvent[];
  // Sentences appended to the prompt for the vertical controls: jumpPrompt while
  // jumping (Space), crouchPrompt while crouching (C held), standPrompt on the
  // crouch RELEASE (the "stands back up" line). Per-scene so they read in-context
  // and are edited in the scene editor (persisted via the override store).
  jumpPrompt?: string;
  crouchPrompt?: string;
  standPrompt?: string;
}

export interface StructuredExample {
  id: string;
  name: string;
  description?: string;
  image: ImagePreset;
  scene: StructuredScene;
}

function resolveDetail(e: NamedEvent, isMoving: boolean): string {
  return typeof e.detail === "string"
    ? e.detail
    : e.detail[isMoving ? "dynamic" : "static"];
}

function baseVersionOf(e: NamedEvent): string {
  return e.baseVersion ?? DEFAULT_LAYER_VERSION;
}
function cameraVersionOf(e: NamedEvent): string {
  return e.cameraVersion ?? DEFAULT_LAYER_VERSION;
}
function movementVersionOf(e: NamedEvent): string {
  return e.movementVersion ?? DEFAULT_LAYER_VERSION;
}

function resolveBase(scene: StructuredScene, version: string): string {
  return scene.base[version] ?? scene.base.default;
}
function resolveCamera(
  scene: StructuredScene,
  version: string,
  isMoving: boolean,
): string {
  const variant = scene.camera[version] ?? scene.camera.default;
  return isMoving ? variant.dynamic : variant.static;
}
function resolveMovement(
  scene: StructuredScene,
  version: string,
  isMoving: boolean,
): string {
  const variant = scene.movement[version] ?? scene.movement.default;
  return isMoving ? variant.dynamic : variant.static;
}

export function composePrompt(
  scene: StructuredScene,
  isMoving: boolean,
  heldSlots: number[],
  // Sentence appended while a vertical control is engaged (jump/crouch). Passed
  // in (rather than read from input state here) so this stays a pure function;
  // the caller decides when jump/crouch is active.
  verticalPrompt = "",
): string {
  const heldEvents = heldSlots
    .map((slot) => scene.events[slot])
    .filter((e): e is NamedEvent => Boolean(e));

  // Active base version = the most-recently-pressed held event's choice
  // (default when nothing is held). Events tagged for other base
  // versions are suppressed because their detail was authored against a
  // different world.
  const activeBase =
    heldEvents.length > 0
      ? baseVersionOf(heldEvents[heldEvents.length - 1])
      : DEFAULT_LAYER_VERSION;
  const compatible = heldEvents.filter((e) => baseVersionOf(e) === activeBase);

  // Within the compatible set, camera and movement versions are taken
  // from the most-recently-pressed compatible event's choices. This
  // lets one big event swap to a compact camera/movement without
  // forcing other compatible events off the stack.
  const mostRecentCompatible = compatible[compatible.length - 1];
  const activeCamera = mostRecentCompatible
    ? cameraVersionOf(mostRecentCompatible)
    : DEFAULT_LAYER_VERSION;
  const activeMovement = mostRecentCompatible
    ? movementVersionOf(mostRecentCompatible)
    : DEFAULT_LAYER_VERSION;

  const base = resolveBase(scene, activeBase);
  const camera = resolveCamera(scene, activeCamera, isMoving);
  const movement = resolveMovement(scene, activeMovement, isMoving);
  const details = compatible.map((e) => resolveDetail(e, isMoving));
  return [base, camera, movement, ...details, verticalPrompt]
    .map((s) => s.trim())
    .filter(Boolean)
    .join(" ");
}

// Deep-clone a StructuredScene so the editor can mutate a scene without
// touching the original constants. Layer registries and events are
// recreated from scratch; ShotVariant objects are spread.
export function cloneScene(scene: StructuredScene): StructuredScene {
  const cloneShotRegistry = (
    reg: LayerRegistry<ShotVariant>,
  ): LayerRegistry<ShotVariant> => {
    const out: Record<string, ShotVariant> = {};
    for (const [k, v] of Object.entries(reg)) out[k] = { ...v };
    return out as LayerRegistry<ShotVariant>;
  };
  return {
    base: { ...scene.base },
    camera: cloneShotRegistry(scene.camera),
    movement: cloneShotRegistry(scene.movement),
    events: scene.events.map((e) => ({
      ...e,
      detail: typeof e.detail === "string" ? e.detail : { ...e.detail },
    })),
    jumpPrompt: scene.jumpPrompt,
    crouchPrompt: scene.crouchPrompt,
    standPrompt: scene.standPrompt,
  };
}

// Structural equality on two scenes. Scenes are small and fully
// JSON-serialisable (strings, plain objects, arrays), so byte-equal
// JSON is a sound proxy for "no semantic difference". Used to decide
// whether a user override is meaningful or has reverted to the
// pristine constant.
export function scenesEqual(a: StructuredScene, b: StructuredScene): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

// Blank scene used when the user creates a new layered prompt from
// scratch. The required "default" key is present on every layer.
export function emptyScene(): StructuredScene {
  return {
    base: { default: "" },
    camera: { default: { static: "", dynamic: "" } },
    movement: { default: { static: "", dynamic: "" } },
    events: [],
    jumpPrompt: "",
    crouchPrompt: "",
    standPrompt: "",
  };
}

// The example scenarios are sourced from the "lingbot-cases" corpus rather
// than hand-authored here; see lingbot-cases-examples.ts for the conversion
// (base_prompt -> base.default, per-slot actions -> events, "space" slot ->
// jumpPrompt) and provenance notes.
export const EXAMPLES: StructuredExample[] = LINGBOT_CASES_EXAMPLE_LIST;

// The same examples keyed by id. The controller looks up scenes by example
// id and recomposes set_prompt strings as movement / hold-key state changes.
export const STRUCTURED_EXAMPLES: Record<string, StructuredExample> =
  Object.fromEntries(EXAMPLES.map((ex) => [ex.id, ex]));
