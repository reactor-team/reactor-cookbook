import {
  DEFAULT_LAYER_VERSION,
  type NamedEvent,
  type StructuredScene,
} from "@/lib/lingbot-world-prompts";

export type SegmentKind = "base" | "camera" | "movement" | "event" | "vertical";

export const SEGMENT_PALETTE: Record<
  SegmentKind,
  {
    bg: string;
    bgStrong: string;
    text: string;
    border: string;
    dot: string;
    label: string;
  }
> = {
  base: {
    bg: "bg-sky-400/10",
    bgStrong: "bg-sky-400/20",
    text: "text-sky-100",
    border: "border-sky-400/40",
    dot: "bg-sky-400",
    label: "base",
  },
  camera: {
    bg: "bg-emerald-400/10",
    bgStrong: "bg-emerald-400/20",
    text: "text-emerald-100",
    border: "border-emerald-400/40",
    dot: "bg-emerald-400",
    label: "camera",
  },
  movement: {
    bg: "bg-violet-400/10",
    bgStrong: "bg-violet-400/20",
    text: "text-violet-100",
    border: "border-violet-400/40",
    dot: "bg-violet-400",
    label: "movement",
  },
  event: {
    bg: "bg-rose-400/10",
    bgStrong: "bg-rose-400/20",
    text: "text-rose-100",
    border: "border-rose-400/40",
    dot: "bg-rose-400",
    label: "event",
  },
  vertical: {
    bg: "bg-amber-300/10",
    bgStrong: "bg-amber-300/20",
    text: "text-amber-100",
    border: "border-amber-300/40",
    dot: "bg-amber-300",
    label: "jump / crouch",
  },
};

export type Segment = {
  kind: SegmentKind;
  text: string;
  // Layer-specific tag shown after the colored swatch (e.g. "base ·
  // calm_clear" or "event 2 · Storm break") to disambiguate when
  // several events share the same hue.
  tag: string;
};

// Mirror of composePrompt's selection logic, surfacing the intermediate
// versions / sets so callers can show *why* the running prompt looks
// the way it does, not just what the final text is.
export function resolveSelection(
  scene: StructuredScene,
  isMoving: boolean,
  heldSlots: number[],
) {
  const heldEvents = heldSlots
    .map((slot) => ({ slot, event: scene.events[slot] }))
    .filter((x): x is { slot: number; event: NamedEvent } => Boolean(x.event));

  const activeBase =
    heldEvents.length > 0
      ? (heldEvents[heldEvents.length - 1].event.baseVersion ??
        DEFAULT_LAYER_VERSION)
      : DEFAULT_LAYER_VERSION;

  const compatible = heldEvents.filter(
    (h) => (h.event.baseVersion ?? DEFAULT_LAYER_VERSION) === activeBase,
  );
  const suppressed = heldEvents.filter(
    (h) => (h.event.baseVersion ?? DEFAULT_LAYER_VERSION) !== activeBase,
  );

  const mostRecentCompatible = compatible[compatible.length - 1]?.event;
  const activeCamera = mostRecentCompatible
    ? (mostRecentCompatible.cameraVersion ?? DEFAULT_LAYER_VERSION)
    : DEFAULT_LAYER_VERSION;
  const activeMovement = mostRecentCompatible
    ? (mostRecentCompatible.movementVersion ?? DEFAULT_LAYER_VERSION)
    : DEFAULT_LAYER_VERSION;

  const baseProse = scene.base[activeBase] ?? scene.base[DEFAULT_LAYER_VERSION];
  const cameraVariant =
    scene.camera[activeCamera] ?? scene.camera[DEFAULT_LAYER_VERSION];
  const movementVariant =
    scene.movement[activeMovement] ?? scene.movement[DEFAULT_LAYER_VERSION];

  return {
    activeBase,
    activeCamera,
    activeMovement,
    heldEvents,
    compatible,
    suppressed,
    baseProse,
    cameraBranch: isMoving ? cameraVariant.dynamic : cameraVariant.static,
    movementBranch: isMoving ? movementVariant.dynamic : movementVariant.static,
  };
}

export function detailOf(event: NamedEvent, isMoving: boolean): string {
  return typeof event.detail === "string"
    ? event.detail
    : event.detail[isMoving ? "dynamic" : "static"];
}

// Build the segmented view of composePrompt's output: an ordered list
// of contributing pieces so renderers can color each one and still
// present them as a single flowing prompt.
export function composePromptSegments(
  sel: ReturnType<typeof resolveSelection>,
  isMoving: boolean,
  verticalPrompt = "",
): Segment[] {
  const out: Segment[] = [];
  const baseText = sel.baseProse?.trim();
  if (baseText) {
    out.push({ kind: "base", text: baseText, tag: `base · ${sel.activeBase}` });
  }
  const cameraText = sel.cameraBranch?.trim();
  if (cameraText) {
    out.push({
      kind: "camera",
      text: cameraText,
      tag: `camera · ${sel.activeCamera} · ${isMoving ? "dynamic" : "static"}`,
    });
  }
  const movementText = sel.movementBranch?.trim();
  if (movementText) {
    out.push({
      kind: "movement",
      text: movementText,
      tag: `movement · ${sel.activeMovement} · ${isMoving ? "dynamic" : "static"}`,
    });
  }
  for (const h of sel.compatible) {
    const text = detailOf(h.event, isMoving).trim();
    if (!text) continue;
    out.push({
      kind: "event",
      text,
      tag: `event ${h.slot + 1} · ${h.event.name.trim() || "unnamed"}`,
    });
  }
  const verticalText = verticalPrompt.trim();
  if (verticalText) {
    out.push({ kind: "vertical", text: verticalText, tag: "jump / crouch" });
  }
  return out;
}
