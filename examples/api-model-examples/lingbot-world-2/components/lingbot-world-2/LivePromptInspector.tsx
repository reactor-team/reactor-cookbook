"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  DEFAULT_LAYER_VERSION,
  composePrompt,
  type NamedEvent,
  type StructuredScene,
} from "@/lib/lingbot-world-prompts";
import {
  SEGMENT_PALETTE,
  composePromptSegments,
  detailOf,
  resolveSelection,
  type SegmentKind,
} from "@/components/lingbot-world-2/prompt-segments";

const SectionLabel = ({ children }: { children: React.ReactNode }) => (
  <span className="font-mono text-[10px] uppercase tracking-wider text-white/55">
    {children}
  </span>
);

const VersionPill = ({
  version,
  branch,
  highlight,
}: {
  version: string;
  branch?: "static" | "dynamic";
  highlight?: boolean;
}) => (
  <span
    className={cn(
      "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 font-mono text-[10px]",
      highlight
        ? "border-amber-300/60 bg-amber-300/15 text-amber-200"
        : "border-white/15 bg-white/5 text-white/75",
    )}
  >
    <span className="font-semibold">{version}</span>
    {branch && (
      <>
        <span className="text-white/35">·</span>
        <span className="text-white/70">{branch}</span>
      </>
    )}
  </span>
);

function ProseBlock({ text }: { text: string }) {
  return (
    <div className="rounded-md border border-white/10 bg-black/30 p-3">
      {text.trim() ? (
        <p className="font-mono text-[12px] leading-relaxed text-white/85 whitespace-pre-wrap break-words">
          {text}
        </p>
      ) : (
        <p className="font-mono text-[11px] italic text-white/30">(empty)</p>
      )}
    </div>
  );
}

function LayerCard({
  layer,
  activeVersion,
  branch,
  prose,
  kind,
}: {
  layer: string;
  activeVersion: string;
  branch?: "static" | "dynamic";
  prose: string;
  kind: SegmentKind;
}) {
  const c = SEGMENT_PALETTE[kind];
  return (
    <div
      className={cn(
        "rounded-lg border-l-4 border border-white/10 bg-white/[0.025] p-4 flex flex-col gap-2",
        c.border,
      )}
    >
      <div className="flex items-center gap-2">
        <span className={cn("w-2 h-2 rounded-full shrink-0", c.dot)} />
        <SectionLabel>{layer}</SectionLabel>
        <VersionPill version={activeVersion} branch={branch} highlight />
      </div>
      <ProseBlock text={prose} />
    </div>
  );
}

function EventRow({
  index,
  event,
  state,
  isMoving,
}: {
  index: number;
  event: NamedEvent;
  state: "compatible" | "suppressed" | "idle";
  isMoving: boolean;
}) {
  const branched = typeof event.detail !== "string";
  const eventPalette = SEGMENT_PALETTE.event;
  return (
    <div
      className={cn(
        "rounded-md border-l-4 border p-3 flex flex-col gap-2",
        state === "compatible" && cn(eventPalette.border, eventPalette.bg),
        state === "suppressed" && "border-red-400/30 bg-red-400/5",
        state === "idle" && "border-white/10 bg-white/[0.02]",
      )}
    >
      <div className="flex items-center gap-2 flex-wrap">
        {state === "compatible" && (
          <span
            className={cn("w-2 h-2 rounded-full shrink-0", eventPalette.dot)}
          />
        )}
        <span className="inline-flex h-5 w-5 items-center justify-center rounded border border-white/20 bg-white/10 font-mono text-[10px] font-bold text-white/85">
          {index + 1}
        </span>
        <span className="font-mono text-[12px] text-white">
          {event.name.trim() || "(unnamed)"}
        </span>
        {state === "compatible" && (
          <span
            className={cn(
              "font-mono text-[9px] uppercase tracking-wider rounded px-1.5 py-0.5",
              eventPalette.bgStrong,
              eventPalette.text,
            )}
          >
            held · contributing
          </span>
        )}
        {state === "suppressed" && (
          <span
            className="font-mono text-[9px] uppercase tracking-wider rounded bg-red-400/20 text-red-300 px-1.5 py-0.5"
            title="Held, but its baseVersion doesn't match the active base — composePrompt skips this event"
          >
            held · suppressed
          </span>
        )}
        <div className="ml-auto flex items-center gap-1">
          <VersionPill version={event.baseVersion ?? DEFAULT_LAYER_VERSION} />
          <VersionPill version={event.cameraVersion ?? DEFAULT_LAYER_VERSION} />
          <VersionPill
            version={event.movementVersion ?? DEFAULT_LAYER_VERSION}
          />
        </div>
      </div>
      <ProseBlock
        text={
          branched
            ? `[${isMoving ? "dynamic" : "static"} branch] ${detailOf(event, isMoving)}`
            : detailOf(event, isMoving)
        }
      />
    </div>
  );
}

export function LivePromptInspector({
  scene,
  isMoving,
  heldSlots,
  verticalPrompt = "",
  onClose,
}: {
  scene: StructuredScene;
  isMoving: boolean;
  heldSlots: number[];
  verticalPrompt?: string;
  onClose: () => void;
}) {
  const [copied, setCopied] = useState(false);

  // Side panel: don't lock body scroll (user must still be able to
  // interact with the video / controls visible on the right).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      const t = e.target as HTMLElement | null;
      if (
        t &&
        (t.tagName === "INPUT" ||
          t.tagName === "TEXTAREA" ||
          t.isContentEditable)
      )
        return;
      onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const sel = resolveSelection(scene, isMoving, heldSlots);
  const composed = composePrompt(scene, isMoving, heldSlots, verticalPrompt);
  const segments = composePromptSegments(sel, isMoving, verticalPrompt);

  const heldSet = new Set(heldSlots);
  const suppressedSet = new Set(sel.suppressed.map((h) => h.slot));

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(composed);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {}
  };

  return (
    <div
      role="region"
      aria-label="Current prompt inspector"
      className="flex flex-col"
    >
      <div className="flex w-full flex-col">
        {/* Header */}
        <div className="flex items-start gap-2 pb-3 border-b border-white/10">
          <div className="flex flex-col gap-0.5 min-w-0 flex-1">
            <h2 className="font-mono text-xs uppercase tracking-widest text-white">
              Current prompt
            </h2>
            <span className="font-mono text-[10px] text-white/45 leading-snug">
              Live view — updates as movement / held events change.
            </span>
          </div>
          <span
            className={cn(
              "inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 font-mono text-[10px] shrink-0",
              isMoving
                ? "border-amber-300/60 bg-amber-300/15 text-amber-200"
                : "border-white/15 bg-white/5 text-white/55",
            )}
            title="True while any WASD key is held"
          >
            <span
              className={cn(
                "w-1.5 h-1.5 rounded-full",
                isMoving ? "bg-amber-300" : "bg-white/30",
              )}
            />
            isMoving·{String(isMoving)}
          </span>
          <Button
            size="xs"
            variant="ghost"
            onClick={onClose}
            className="shrink-0 h-7 px-2 font-mono text-[10px]"
          >
            ✕
          </Button>
        </div>

        {/* Body — outer parent (sidebar column) handles scrolling */}
        <div className="pt-3">
          <div className="flex flex-col gap-4">
            {/* Composed prompt — the actual string sent to set_prompt.
                Each segment is rendered in its source layer's color so
                the user can visually trace where every piece of prose
                came from. */}
            <div className="flex flex-col gap-2">
              <div className="flex items-center gap-3 flex-wrap">
                <SectionLabel>composed prompt sent to backend</SectionLabel>
                <span className="font-mono text-[10px] text-white/35">
                  {composed.length} chars
                </span>
                <div className="ml-auto">
                  <Button size="sm" variant="ghost" onClick={copy}>
                    {copied ? "Copied" : "Copy"}
                  </Button>
                </div>
              </div>

              {/* Color legend — maps each hue to its layer so the user
                  doesn't have to scroll down to figure out what each
                  color means. */}
              <div className="flex items-center gap-2 flex-wrap font-mono text-[10px] text-white/55">
                {(
                  [
                    "base",
                    "camera",
                    "movement",
                    "event",
                    "vertical",
                  ] as SegmentKind[]
                ).map((k) => {
                  const c = SEGMENT_PALETTE[k];
                  return (
                    <span key={k} className="inline-flex items-center gap-1.5">
                      <span className={cn("w-2 h-2 rounded-full", c.dot)} />
                      {c.label}
                    </span>
                  );
                })}
              </div>

              <div className="rounded-lg border border-white/10 bg-black/30 p-3">
                {segments.length === 0 ? (
                  <p className="font-mono text-[11px] italic text-white/30">
                    (empty)
                  </p>
                ) : (
                  <p className="font-mono text-[12px] leading-relaxed break-words">
                    {segments.map((s, i) => {
                      const c = SEGMENT_PALETTE[s.kind];
                      return (
                        <span key={i}>
                          <span
                            title={s.tag}
                            className={cn(
                              "rounded px-1.5 py-0.5 box-decoration-clone",
                              c.bg,
                              c.text,
                            )}
                          >
                            {s.text}
                          </span>
                          {i < segments.length - 1 && " "}
                        </span>
                      );
                    })}
                  </p>
                )}
              </div>
            </div>

            {/* Layer breakdown */}
            <div className="flex flex-col gap-3">
              <SectionLabel>layer breakdown</SectionLabel>
              <LayerCard
                kind="base"
                layer="base"
                activeVersion={sel.activeBase}
                prose={sel.baseProse ?? ""}
              />
              <LayerCard
                kind="camera"
                layer="camera"
                activeVersion={sel.activeCamera}
                branch={isMoving ? "dynamic" : "static"}
                prose={sel.cameraBranch ?? ""}
              />
              <LayerCard
                kind="movement"
                layer="movement"
                activeVersion={sel.activeMovement}
                branch={isMoving ? "dynamic" : "static"}
                prose={sel.movementBranch ?? ""}
              />
            </div>

            {/* Events */}
            <div className="flex flex-col gap-3">
              <div className="flex items-center gap-3 flex-wrap">
                <SectionLabel>events</SectionLabel>
                <span className="font-mono text-[10px] text-white/35">
                  {heldSlots.length} held · {sel.compatible.length} contributing
                  · {sel.suppressed.length} suppressed
                </span>
              </div>
              {scene.events.length === 0 && (
                <p className="font-mono text-[11px] italic text-white/30">
                  No events defined.
                </p>
              )}
              {scene.events.map((event, i) => {
                const state = suppressedSet.has(i)
                  ? "suppressed"
                  : heldSet.has(i)
                    ? "compatible"
                    : "idle";
                return (
                  <EventRow
                    key={i}
                    index={i}
                    event={event}
                    state={state as "compatible" | "suppressed" | "idle"}
                    isMoving={isMoving}
                  />
                );
              })}
            </div>

            {/* How it composes */}
            <div className="rounded-lg border border-white/10 bg-white/[0.02] p-4 flex flex-col gap-1.5">
              <SectionLabel>how this was composed</SectionLabel>
              <p className="font-mono text-[11px] leading-relaxed text-white/55">
                <code className="text-amber-300/80">composePrompt</code> joins{" "}
                <code>base[{sel.activeBase}]</code>,{" "}
                <code>
                  camera[{sel.activeCamera}].{isMoving ? "dynamic" : "static"}
                </code>
                ,{" "}
                <code>
                  movement[{sel.activeMovement}].
                  {isMoving ? "dynamic" : "static"}
                </code>
                {sel.compatible.length > 0 && (
                  <>
                    {" "}
                    and the details of {sel.compatible.length} compatible event
                    {sel.compatible.length === 1 ? "" : "s"}
                  </>
                )}
                .
                {sel.suppressed.length > 0 && (
                  <>
                    {" "}
                    {sel.suppressed.length} event
                    {sel.suppressed.length === 1 ? "" : "s"} held but suppressed
                    because the <code>baseVersion</code> doesn&apos;t match the
                    active base <code>{sel.activeBase}</code>.
                  </>
                )}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
