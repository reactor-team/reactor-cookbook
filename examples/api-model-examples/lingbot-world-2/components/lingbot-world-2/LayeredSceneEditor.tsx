"use client";

import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import {
  DEFAULT_LAYER_VERSION,
  composePrompt,
  type LayerRegistry,
  type NamedEvent,
  type ShotVariant,
  type StructuredScene,
} from "@/lib/lingbot-world-prompts";
import {
  SEGMENT_PALETTE,
  composePromptSegments,
  resolveSelection,
  type SegmentKind,
} from "@/components/lingbot-world-2/prompt-segments";

// ---- Registry mutation helpers (order-preserving) ----

function renameKey<T>(
  reg: Record<string, T>,
  oldKey: string,
  newKey: string,
): Record<string, T> {
  if (oldKey === newKey) return reg;
  const out: Record<string, T> = {};
  for (const [k, v] of Object.entries(reg)) {
    out[k === oldKey ? newKey : k] = v;
  }
  return out;
}

function setKey<T>(
  reg: Record<string, T>,
  key: string,
  value: T,
): Record<string, T> {
  return { ...reg, [key]: value };
}

function deleteKey<T>(reg: Record<string, T>, key: string): Record<string, T> {
  const out: Record<string, T> = {};
  for (const [k, v] of Object.entries(reg)) {
    if (k !== key) out[k] = v;
  }
  return out;
}

function uniqueKey(reg: Record<string, unknown>, base: string): string {
  if (!(base in reg)) return base;
  for (let i = 2; i < 1000; i++) {
    const candidate = `${base}_${i}`;
    if (!(candidate in reg)) return candidate;
  }
  return `${base}_${Date.now()}`;
}

// ---- Diff against pristine ----
//
// When the editor is opened against an example, we compare the current
// scene against the pristine constant so each tab / card / sub-field
// can show *where* the user's edits live. For the custom slot there is
// no pristine to compare against — pristine is undefined and no diff
// markers are rendered.

type RegistryDiff = {
  added: Set<string>;
  removed: Set<string>;
  edited: Set<string>;
};

type ShotRegistryDiff = {
  added: Set<string>;
  removed: Set<string>;
  staticChanged: Set<string>;
  dynamicChanged: Set<string>;
};

type EventDiff = {
  state: "added" | "edited" | "same";
  nameChanged: boolean;
  baseVersionChanged: boolean;
  cameraVersionChanged: boolean;
  movementVersionChanged: boolean;
  detailChanged: boolean;
  // For branched details we surface which branch differs so we can
  // mark only the right sub-textarea (not both).
  detailStaticChanged: boolean;
  detailDynamicChanged: boolean;
  // True when the detail switched between flat-string and branched
  // ShotVariant shapes (covers all sub-textareas).
  detailTypeChanged: boolean;
};

type SceneDiff = {
  base: RegistryDiff;
  camera: ShotRegistryDiff;
  movement: ShotRegistryDiff;
  events: EventDiff[];
  // pristine had events past the user's list (e.g. user deleted some).
  // We surface as a count next to the Events tab so the user knows
  // something was removed even though there's no card to point at.
  eventsRemovedCount: number;
};

function diffStringRegistry(
  scene: LayerRegistry<string>,
  pristine: LayerRegistry<string>,
): RegistryDiff {
  const added = new Set<string>();
  const removed = new Set<string>();
  const edited = new Set<string>();
  for (const [k, v] of Object.entries(scene)) {
    if (!(k in pristine)) added.add(k);
    else if (v !== pristine[k]) edited.add(k);
  }
  for (const k of Object.keys(pristine)) {
    if (!(k in scene)) removed.add(k);
  }
  return { added, removed, edited };
}

function diffShotRegistry(
  scene: LayerRegistry<ShotVariant>,
  pristine: LayerRegistry<ShotVariant>,
): ShotRegistryDiff {
  const added = new Set<string>();
  const removed = new Set<string>();
  const staticChanged = new Set<string>();
  const dynamicChanged = new Set<string>();
  for (const [k, v] of Object.entries(scene)) {
    if (!(k in pristine)) {
      added.add(k);
      continue;
    }
    if (v.static !== pristine[k].static) staticChanged.add(k);
    if (v.dynamic !== pristine[k].dynamic) dynamicChanged.add(k);
  }
  for (const k of Object.keys(pristine)) {
    if (!(k in scene)) removed.add(k);
  }
  return { added, removed, staticChanged, dynamicChanged };
}

function diffEvent(scene: NamedEvent, pristine: NamedEvent): EventDiff {
  const sBranched = typeof scene.detail !== "string";
  const pBranched = typeof pristine.detail !== "string";
  const detailTypeChanged = sBranched !== pBranched;

  let detailStaticChanged = false;
  let detailDynamicChanged = false;
  if (!detailTypeChanged) {
    if (!sBranched) {
      // Both flat strings — mark both branches as "changed" if the prose
      // differs, since the same text would render in either branch.
      const changed = scene.detail !== pristine.detail;
      detailStaticChanged = changed;
      detailDynamicChanged = changed;
    } else {
      const s = scene.detail as ShotVariant;
      const p = pristine.detail as ShotVariant;
      detailStaticChanged = s.static !== p.static;
      detailDynamicChanged = s.dynamic !== p.dynamic;
    }
  } else {
    // Type changed — both sub-textareas (or the flat one) are
    // effectively a new authoring shape, so flag both.
    detailStaticChanged = true;
    detailDynamicChanged = true;
  }
  const detailChanged =
    detailTypeChanged || detailStaticChanged || detailDynamicChanged;

  const nameChanged = scene.name !== pristine.name;
  const baseVersionChanged =
    (scene.baseVersion ?? DEFAULT_LAYER_VERSION) !==
    (pristine.baseVersion ?? DEFAULT_LAYER_VERSION);
  const cameraVersionChanged =
    (scene.cameraVersion ?? DEFAULT_LAYER_VERSION) !==
    (pristine.cameraVersion ?? DEFAULT_LAYER_VERSION);
  const movementVersionChanged =
    (scene.movementVersion ?? DEFAULT_LAYER_VERSION) !==
    (pristine.movementVersion ?? DEFAULT_LAYER_VERSION);

  const any =
    nameChanged ||
    baseVersionChanged ||
    cameraVersionChanged ||
    movementVersionChanged ||
    detailChanged;

  return {
    state: any ? "edited" : "same",
    nameChanged,
    baseVersionChanged,
    cameraVersionChanged,
    movementVersionChanged,
    detailChanged,
    detailStaticChanged,
    detailDynamicChanged,
    detailTypeChanged,
  };
}

function computeSceneDiff(
  scene: StructuredScene,
  pristine: StructuredScene,
): SceneDiff {
  const events: EventDiff[] = scene.events.map((e, i) => {
    const p = pristine.events[i];
    if (!p) {
      return {
        state: "added",
        nameChanged: true,
        baseVersionChanged: true,
        cameraVersionChanged: true,
        movementVersionChanged: true,
        detailChanged: true,
        detailStaticChanged: true,
        detailDynamicChanged: true,
        detailTypeChanged: true,
      };
    }
    return diffEvent(e, p);
  });
  return {
    base: diffStringRegistry(scene.base, pristine.base),
    camera: diffShotRegistry(scene.camera, pristine.camera),
    movement: diffShotRegistry(scene.movement, pristine.movement),
    events,
    eventsRemovedCount: Math.max(
      0,
      pristine.events.length - scene.events.length,
    ),
  };
}

// ---- Layout primitives ----

type LayerName = "base" | "camera" | "movement";
type VersionField = "baseVersion" | "cameraVersion" | "movementVersion";
const LAYER_TO_FIELD: Record<LayerName, VersionField> = {
  base: "baseVersion",
  camera: "cameraVersion",
  movement: "movementVersion",
};

type Tab = "base" | "camera" | "movement" | "vertical" | "events" | "preview";
const TABS: { id: Tab; label: string }[] = [
  { id: "base", label: "Base" },
  { id: "camera", label: "Camera" },
  { id: "movement", label: "Movement" },
  { id: "vertical", label: "Jump / Crouch" },
  { id: "events", label: "Events" },
  { id: "preview", label: "Preview" },
];

const Hint = ({ children }: { children: React.ReactNode }) => (
  <p className="font-mono text-[11px] text-white/45 leading-relaxed">
    {children}
  </p>
);

const FieldLabel = ({ children }: { children: React.ReactNode }) => (
  <span className="font-mono text-[10px] uppercase tracking-wider text-white/55">
    {children}
  </span>
);

// Small inline indicator used wherever a field / card / sub-textarea
// differs from the pristine example. Variant `new` is reserved for
// keys / events the user added on top of the pristine constant.
function DiffMark({
  variant,
  label,
  title,
}: {
  variant: "edited" | "new" | "removed";
  label?: string;
  title?: string;
}) {
  const palette =
    variant === "edited"
      ? "border-amber-300/50 bg-amber-300/15 text-amber-200"
      : variant === "new"
        ? "border-emerald-300/50 bg-emerald-300/15 text-emerald-200"
        : "border-red-400/40 bg-red-400/10 text-red-300";
  return (
    <span
      title={title}
      className={cn(
        "inline-flex items-center gap-1 shrink-0 rounded border px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wider",
        palette,
      )}
    >
      <span className="w-1 h-1 rounded-full bg-current opacity-80" />
      {label ?? variant}
    </span>
  );
}

// Wide textarea that fills the available column horizontally and (when
// dropped inside a flex-1 wrapper) vertically too. Lines wrap inside
// the textarea, so the user can author long prose without horizontal
// overflow.
function ProseTextarea({
  value,
  onChange,
  placeholder,
  className,
}: {
  value: string;
  onChange: (next: string) => void;
  placeholder?: string;
  className?: string;
}) {
  return (
    <textarea
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      className={cn(
        "block w-full flex-1 min-h-[160px] rounded-md border border-white/10 bg-white/[0.04] px-3 py-2 font-mono text-[12px] leading-relaxed text-white placeholder-white/30 focus:outline-none focus:border-white/30 resize-none",
        className,
      )}
    />
  );
}

// ---- Version key input (with rename validation) ----

function VersionKeyInput({
  value,
  isDefault,
  onCommit,
  takenKeys,
}: {
  value: string;
  isDefault: boolean;
  onCommit: (next: string) => void;
  takenKeys: string[];
}) {
  const [draft, setDraft] = useState(value);
  const [error, setError] = useState<string | null>(null);
  // Sync external rename / re-mount
  useEffect(() => {
    setDraft(value);
  }, [value]);

  const commit = () => {
    const next = draft.trim();
    if (!next || next === value) {
      setDraft(value);
      setError(null);
      return;
    }
    if (takenKeys.includes(next)) {
      setError("key already used");
      setDraft(value);
      setTimeout(() => setError(null), 1500);
      return;
    }
    setError(null);
    onCommit(next);
  };

  return (
    <div className="flex flex-col gap-0.5">
      <Input
        value={isDefault ? DEFAULT_LAYER_VERSION : draft}
        disabled={isDefault}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") (e.target as HTMLInputElement).blur();
        }}
        placeholder="version key"
        className={cn(
          "h-8 w-44 font-mono text-[12px]",
          error && "border-red-500/60",
        )}
      />
      {error && (
        <span className="font-mono text-[10px] text-red-400/80">{error}</span>
      )}
    </div>
  );
}

// ---- Layer panels ----

function VersionCard({
  versionKey,
  isDefault,
  onRename,
  takenKeys,
  onRemove,
  status,
  children,
}: {
  versionKey: string;
  isDefault: boolean;
  onRename: (next: string) => void;
  takenKeys: string[];
  onRemove: () => void;
  status?: "added" | "edited" | null;
  children: React.ReactNode;
}) {
  return (
    <div
      className={cn(
        "rounded-md border bg-white/[0.025] p-4 flex flex-col gap-3 flex-1 min-h-[260px]",
        status === "added"
          ? "border-emerald-300/40"
          : status === "edited"
            ? "border-amber-300/35"
            : "border-white/10",
      )}
    >
      <div className="flex items-center gap-3 shrink-0 flex-wrap">
        <FieldLabel>version key</FieldLabel>
        <VersionKeyInput
          value={versionKey}
          isDefault={isDefault}
          takenKeys={takenKeys}
          onCommit={onRename}
        />
        {status === "added" && (
          <DiffMark
            variant="new"
            label="added"
            title="This version key isn't in the pristine example."
          />
        )}
        {status === "edited" && (
          <DiffMark
            variant="edited"
            title="Content under this version differs from the pristine example."
          />
        )}
        {isDefault && (
          <span className="font-mono text-[10px] text-white/35">
            required — used when no event overrides this layer
          </span>
        )}
        <div className="ml-auto">
          {!isDefault && (
            <Button size="sm" variant="ghost" onClick={onRemove}>
              Delete
            </Button>
          )}
        </div>
      </div>
      {children}
    </div>
  );
}

function BasePanel({
  registry,
  diff,
  onChange,
  onRename,
}: {
  registry: LayerRegistry<string>;
  diff: RegistryDiff | null;
  onChange: (next: LayerRegistry<string>) => void;
  onRename: (oldKey: string, newKey: string) => void;
}) {
  const entries = Object.entries(registry);
  const keys = entries.map(([k]) => k);
  const updateValue = (key: string, value: string) =>
    onChange(setKey(registry, key, value) as LayerRegistry<string>);
  const remove = (key: string) =>
    onChange(deleteKey(registry, key) as LayerRegistry<string>);
  const add = () => {
    const key = uniqueKey(registry, "variant");
    onChange(setKey(registry, key, "") as LayerRegistry<string>);
  };

  return (
    <div className="flex flex-col gap-4 flex-1 min-h-full">
      <Hint>
        World identity — subject, environment, and overall style. Each version
        is one paragraph of prose. Events tagged with{" "}
        <code className="text-amber-300/80">baseVersion: &quot;X&quot;</code>{" "}
        will compose against version <code>X</code> instead of{" "}
        <code>default</code>; events from incompatible base versions are
        suppressed while held.
      </Hint>
      {diff && diff.removed.size > 0 && (
        <p className="font-mono text-[11px] text-red-300/80">
          Removed from pristine: {[...diff.removed].join(", ")}.
        </p>
      )}
      <div className="flex flex-col gap-4 flex-1 min-h-0">
        {entries.map(([key, value]) => {
          const status: "added" | "edited" | null = !diff
            ? null
            : diff.added.has(key)
              ? "added"
              : diff.edited.has(key)
                ? "edited"
                : null;
          const proseChanged = status === "edited";
          return (
            <VersionCard
              key={key}
              versionKey={key}
              isDefault={key === DEFAULT_LAYER_VERSION}
              takenKeys={keys.filter((k) => k !== key)}
              onRename={(next) => onRename(key, next)}
              onRemove={() => remove(key)}
              status={status}
            >
              <div className="flex flex-col gap-1.5 flex-1 min-h-0">
                <div className="flex items-center gap-2">
                  <FieldLabel>prose</FieldLabel>
                  {proseChanged && <DiffMark variant="edited" />}
                </div>
                <ProseTextarea
                  value={value}
                  onChange={(v) => updateValue(key, v)}
                  placeholder="Subject + environment + style prose"
                  className={proseChanged ? "border-amber-300/40" : undefined}
                />
              </div>
            </VersionCard>
          );
        })}
      </div>
      <div className="shrink-0">
        <Button size="sm" variant="outline" onClick={add}>
          + Add base version
        </Button>
      </div>
    </div>
  );
}

function ShotPanel({
  layer,
  registry,
  diff,
  onChange,
  onRename,
}: {
  layer: "camera" | "movement";
  registry: LayerRegistry<ShotVariant>;
  diff: ShotRegistryDiff | null;
  onChange: (next: LayerRegistry<ShotVariant>) => void;
  onRename: (oldKey: string, newKey: string) => void;
}) {
  const entries = Object.entries(registry);
  const keys = entries.map(([k]) => k);
  const updateField = (
    key: string,
    field: keyof ShotVariant,
    value: string,
  ) => {
    const prev = registry[key];
    onChange(
      setKey(registry, key, {
        ...prev,
        [field]: value,
      }) as LayerRegistry<ShotVariant>,
    );
  };
  const remove = (key: string) =>
    onChange(deleteKey(registry, key) as LayerRegistry<ShotVariant>);
  const add = () => {
    const key = uniqueKey(registry, "variant");
    onChange(
      setKey(registry, key, {
        static: "",
        dynamic: "",
      }) as LayerRegistry<ShotVariant>,
    );
  };

  const isCamera = layer === "camera";

  return (
    <div className="flex flex-col gap-4 flex-1 min-h-full">
      <Hint>
        {isCamera ? (
          <>
            Camera framing. <strong>Static</strong> = subject locked in place,
            camera orbits on look-input. <strong>Dynamic</strong> = camera
            tracks the moving subject from astern.
          </>
        ) : (
          <>
            Subject motion. <strong>Static</strong> = subject is dead-still (no
            WASD held). <strong>Dynamic</strong> = subject moves forward (WASD
            held).
          </>
        )}
      </Hint>
      {diff && diff.removed.size > 0 && (
        <p className="font-mono text-[11px] text-red-300/80">
          Removed from pristine: {[...diff.removed].join(", ")}.
        </p>
      )}
      <div className="flex flex-col gap-4 flex-1 min-h-0">
        {entries.map(([key, variant]) => {
          const isAdded = diff?.added.has(key) ?? false;
          const staticChanged = diff?.staticChanged.has(key) ?? false;
          const dynamicChanged = diff?.dynamicChanged.has(key) ?? false;
          const status: "added" | "edited" | null = !diff
            ? null
            : isAdded
              ? "added"
              : staticChanged || dynamicChanged
                ? "edited"
                : null;
          return (
            <VersionCard
              key={key}
              versionKey={key}
              isDefault={key === DEFAULT_LAYER_VERSION}
              takenKeys={keys.filter((k) => k !== key)}
              onRename={(next) => onRename(key, next)}
              onRemove={() => remove(key)}
              status={status}
            >
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 flex-1 min-h-0">
                <div className="flex flex-col gap-1.5 min-h-0">
                  <div className="flex items-center gap-2">
                    <FieldLabel>static (no WASD held)</FieldLabel>
                    {staticChanged && !isAdded && <DiffMark variant="edited" />}
                  </div>
                  <ProseTextarea
                    value={variant.static}
                    onChange={(v) => updateField(key, "static", v)}
                    placeholder={
                      isCamera
                        ? "Camera framing when subject is still"
                        : "Subject motion when no WASD held"
                    }
                    className={
                      staticChanged && !isAdded
                        ? "border-amber-300/40"
                        : undefined
                    }
                  />
                </div>
                <div className="flex flex-col gap-1.5 min-h-0">
                  <div className="flex items-center gap-2">
                    <FieldLabel>dynamic (WASD held)</FieldLabel>
                    {dynamicChanged && !isAdded && (
                      <DiffMark variant="edited" />
                    )}
                  </div>
                  <ProseTextarea
                    value={variant.dynamic}
                    onChange={(v) => updateField(key, "dynamic", v)}
                    placeholder={
                      isCamera
                        ? "Camera framing when subject is moving"
                        : "Subject motion when WASD held"
                    }
                    className={
                      dynamicChanged && !isAdded
                        ? "border-amber-300/40"
                        : undefined
                    }
                  />
                </div>
              </div>
            </VersionCard>
          );
        })}
      </div>
      <div className="shrink-0">
        <Button size="sm" variant="outline" onClick={add}>
          + Add {layer} version
        </Button>
      </div>
    </div>
  );
}

function VersionPicker({
  value,
  options,
  onChange,
}: {
  value: string;
  options: string[];
  onChange: (next: string) => void;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="h-8 rounded-md border border-white/15 bg-white/[0.04] px-2 font-mono text-[11px] text-white focus:outline-none focus:border-white/30"
    >
      {options.map((opt) => (
        <option key={opt} value={opt} className="bg-neutral-900">
          {opt}
        </option>
      ))}
    </select>
  );
}

function EventCard({
  index,
  event,
  scene,
  diff,
  onChange,
  onRemove,
  onMove,
  isFirst,
  isLast,
}: {
  index: number;
  event: NamedEvent;
  scene: StructuredScene;
  diff: EventDiff | null;
  onChange: (next: NamedEvent) => void;
  onRemove: () => void;
  onMove: (dir: -1 | 1) => void;
  isFirst: boolean;
  isLast: boolean;
}) {
  const baseKeys = Object.keys(scene.base);
  const cameraKeys = Object.keys(scene.camera);
  const movementKeys = Object.keys(scene.movement);
  const isBranched = typeof event.detail !== "string";

  const updateString = (text: string) => onChange({ ...event, detail: text });
  const updateField = (field: "static" | "dynamic", text: string) => {
    const current: ShotVariant =
      typeof event.detail === "string"
        ? { static: event.detail, dynamic: event.detail }
        : event.detail;
    onChange({ ...event, detail: { ...current, [field]: text } });
  };
  const toggleBranched = (next: boolean) => {
    if (next === isBranched) return;
    if (next) {
      const seed = typeof event.detail === "string" ? event.detail : "";
      onChange({ ...event, detail: { static: seed, dynamic: seed } });
    } else {
      const seed =
        typeof event.detail === "string"
          ? event.detail
          : event.detail.static || event.detail.dynamic;
      onChange({ ...event, detail: seed });
    }
  };

  const isAdded = diff?.state === "added";
  const cardBorder = !diff
    ? "border-white/10"
    : isAdded
      ? "border-emerald-300/40"
      : diff.state === "edited"
        ? "border-amber-300/35"
        : "border-white/10";

  // For an added event, every sub-field is implicitly "new" — skip
  // per-field marks since the card-level "new" pill already says it.
  const showSub = (changed: boolean) => !isAdded && changed;

  return (
    <div
      className={cn(
        "rounded-md border bg-white/[0.025] p-4 flex flex-col gap-3 flex-1 min-h-[320px]",
        cardBorder,
      )}
    >
      <div className="flex items-center gap-2 flex-wrap shrink-0">
        <span className="inline-flex h-7 w-7 items-center justify-center rounded border border-white/20 bg-white/10 font-mono text-[12px] font-bold text-white/85">
          {index + 1}
        </span>
        <Input
          value={event.name}
          onChange={(e) => onChange({ ...event, name: e.target.value })}
          placeholder="Event name (shown on hold chip)"
          className={cn(
            "flex-1 min-w-[200px] font-mono text-[12px]",
            showSub(!!diff?.nameChanged) && "border-amber-300/40",
          )}
        />
        {isAdded && <DiffMark variant="new" />}
        {!isAdded && diff?.state === "edited" && <DiffMark variant="edited" />}
        {showSub(!!diff?.nameChanged) && (
          <DiffMark variant="edited" label="name" />
        )}
        <Button
          size="sm"
          variant="ghost"
          onClick={() => onMove(-1)}
          disabled={isFirst}
          title="Move up"
        >
          ↑
        </Button>
        <Button
          size="sm"
          variant="ghost"
          onClick={() => onMove(1)}
          disabled={isLast}
          title="Move down"
        >
          ↓
        </Button>
        <Button size="sm" variant="ghost" onClick={onRemove}>
          Delete
        </Button>
      </div>

      <div className="flex flex-wrap items-end gap-4 shrink-0">
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2">
            <FieldLabel>base version</FieldLabel>
            {showSub(!!diff?.baseVersionChanged) && (
              <DiffMark variant="edited" />
            )}
          </div>
          <VersionPicker
            value={event.baseVersion ?? DEFAULT_LAYER_VERSION}
            options={baseKeys}
            onChange={(v) => onChange({ ...event, baseVersion: v })}
          />
        </div>
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2">
            <FieldLabel>camera version</FieldLabel>
            {showSub(!!diff?.cameraVersionChanged) && (
              <DiffMark variant="edited" />
            )}
          </div>
          <VersionPicker
            value={event.cameraVersion ?? DEFAULT_LAYER_VERSION}
            options={cameraKeys}
            onChange={(v) => onChange({ ...event, cameraVersion: v })}
          />
        </div>
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2">
            <FieldLabel>movement version</FieldLabel>
            {showSub(!!diff?.movementVersionChanged) && (
              <DiffMark variant="edited" />
            )}
          </div>
          <VersionPicker
            value={event.movementVersion ?? DEFAULT_LAYER_VERSION}
            options={movementKeys}
            onChange={(v) => onChange({ ...event, movementVersion: v })}
          />
        </div>
        <label className="ml-auto flex items-center gap-2 font-mono text-[11px] text-white/70 cursor-pointer">
          <input
            type="checkbox"
            checked={isBranched}
            onChange={(e) => toggleBranched(e.target.checked)}
            className="accent-amber-300"
          />
          branch detail on WASD
          {showSub(!!diff?.detailTypeChanged) && (
            <DiffMark variant="edited" label="shape" />
          )}
        </label>
      </div>

      {isBranched ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 flex-1 min-h-0">
          <div className="flex flex-col gap-1.5 min-h-0">
            <div className="flex items-center gap-2">
              <FieldLabel>detail · static</FieldLabel>
              {showSub(!!diff?.detailStaticChanged) && (
                <DiffMark variant="edited" />
              )}
            </div>
            <ProseTextarea
              value={(event.detail as ShotVariant).static}
              onChange={(v) => updateField("static", v)}
              placeholder="Detail prose when no WASD held"
              className={
                showSub(!!diff?.detailStaticChanged)
                  ? "border-amber-300/40"
                  : undefined
              }
            />
          </div>
          <div className="flex flex-col gap-1.5 min-h-0">
            <div className="flex items-center gap-2">
              <FieldLabel>detail · dynamic</FieldLabel>
              {showSub(!!diff?.detailDynamicChanged) && (
                <DiffMark variant="edited" />
              )}
            </div>
            <ProseTextarea
              value={(event.detail as ShotVariant).dynamic}
              onChange={(v) => updateField("dynamic", v)}
              placeholder="Detail prose when WASD held"
              className={
                showSub(!!diff?.detailDynamicChanged)
                  ? "border-amber-300/40"
                  : undefined
              }
            />
          </div>
        </div>
      ) : (
        <div className="flex flex-col gap-1.5 flex-1 min-h-0">
          <div className="flex items-center gap-2">
            <FieldLabel>detail</FieldLabel>
            {showSub(!!diff?.detailChanged) && <DiffMark variant="edited" />}
          </div>
          <ProseTextarea
            value={event.detail as string}
            onChange={updateString}
            placeholder="Detail prose appended while held"
            className={
              showSub(!!diff?.detailChanged) ? "border-amber-300/40" : undefined
            }
          />
        </div>
      )}
    </div>
  );
}

function EventsPanel({
  scene,
  diff,
  onChange,
}: {
  scene: StructuredScene;
  diff: SceneDiff | null;
  onChange: (next: StructuredScene) => void;
}) {
  const updateEvent = (i: number, next: NamedEvent) => {
    onChange({
      ...scene,
      events: scene.events.map((e, idx) => (idx === i ? next : e)),
    });
  };
  const removeEvent = (i: number) =>
    onChange({ ...scene, events: scene.events.filter((_, idx) => idx !== i) });
  const addEvent = () => {
    if (scene.events.length >= 9) return;
    onChange({ ...scene, events: [...scene.events, { name: "", detail: "" }] });
  };
  const moveEvent = (i: number, dir: -1 | 1) => {
    const j = i + dir;
    if (j < 0 || j >= scene.events.length) return;
    const events = [...scene.events];
    [events[i], events[j]] = [events[j], events[i]];
    onChange({ ...scene, events });
  };

  return (
    <div className="flex flex-col gap-4 flex-1 min-h-full">
      <Hint>
        Each event binds to a number key by position — first event = key{" "}
        <code>1</code>, ninth = key <code>9</code>. Pressing the key holds the
        event; releasing reverts. Pick which version of base / camera / movement
        the event composes against; events whose <code>baseVersion</code>{" "}
        doesn&apos;t match the most-recently-pressed event are suppressed.
      </Hint>
      {diff && diff.eventsRemovedCount > 0 && (
        <p className="font-mono text-[11px] text-red-300/80">
          {diff.eventsRemovedCount} event
          {diff.eventsRemovedCount === 1 ? "" : "s"} removed from the pristine
          list.
        </p>
      )}
      <div className="flex flex-col gap-4 flex-1 min-h-0">
        {scene.events.map((event, i) => (
          <EventCard
            key={i}
            index={i}
            event={event}
            scene={scene}
            diff={diff ? (diff.events[i] ?? null) : null}
            onChange={(next) => updateEvent(i, next)}
            onRemove={() => removeEvent(i)}
            onMove={(dir) => moveEvent(i, dir)}
            isFirst={i === 0}
            isLast={i === scene.events.length - 1}
          />
        ))}
      </div>
      <div className="shrink-0">
        <Button
          size="sm"
          variant="outline"
          onClick={addEvent}
          disabled={scene.events.length >= 9}
        >
          + Add event ({scene.events.length}/9)
        </Button>
      </div>
    </div>
  );
}

function PreviewPanel({ scene }: { scene: StructuredScene }) {
  const [isMoving, setIsMoving] = useState(false);
  const [held, setHeld] = useState<number[]>([]);
  const composed = composePrompt(scene, isMoving, held);
  const sel = resolveSelection(scene, isMoving, held);
  const segments = composePromptSegments(sel, isMoving);
  const toggleHeld = (i: number) =>
    setHeld((h) => (h.includes(i) ? h.filter((x) => x !== i) : [...h, i]));

  return (
    <div className="flex flex-col gap-4">
      <Hint>
        Live preview of <code>composePrompt(scene, isMoving, heldSlots)</code> —
        the exact string the runtime sends to <code>set_prompt</code>. Toggle
        WASD-state and held events to see how the layered registries combine.
      </Hint>

      <div className="flex flex-wrap items-center gap-3">
        <label className="flex items-center gap-2 font-mono text-[12px] text-white/75 cursor-pointer">
          <input
            type="checkbox"
            checked={isMoving}
            onChange={(e) => setIsMoving(e.target.checked)}
            className="accent-amber-300"
          />
          WASD held (isMoving)
        </label>
        <div className="flex flex-wrap gap-1">
          {scene.events.map((event, i) => (
            <button
              key={i}
              type="button"
              onClick={() => toggleHeld(i)}
              className={cn(
                "rounded-full border px-3 py-1 font-mono text-[11px] transition-colors",
                held.includes(i)
                  ? "border-amber-300/80 bg-amber-300/25 text-amber-100"
                  : "border-white/15 bg-white/5 text-white/70 hover:bg-white/10",
              )}
            >
              {i + 1}. {event.name.trim() || "(unnamed)"}
            </button>
          ))}
        </div>
      </div>

      <div className="rounded-md border border-white/10 bg-black/40 p-4 w-full flex flex-col gap-2">
        <div className="flex items-center gap-3 flex-wrap">
          <div className="font-mono text-[10px] uppercase tracking-wider text-white/40">
            composed prompt ({composed.length} chars)
          </div>
          <div className="ml-auto flex items-center gap-2 flex-wrap font-mono text-[10px] text-white/55">
            {(["base", "camera", "movement", "event"] as SegmentKind[]).map(
              (k) => {
                const c = SEGMENT_PALETTE[k];
                return (
                  <span key={k} className="inline-flex items-center gap-1.5">
                    <span className={cn("w-2 h-2 rounded-full", c.dot)} />
                    {c.label}
                  </span>
                );
              },
            )}
          </div>
        </div>
        {segments.length === 0 ? (
          <p className="font-mono text-[12px] italic text-white/30">(empty)</p>
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
  );
}

// ---- Modal shell ----

export function LayeredSceneEditor({
  title,
  subtitle,
  scene,
  pristine,
  onChange,
  onReset,
  onClose,
  resetLabel,
}: {
  title?: string;
  subtitle?: string;
  scene: StructuredScene;
  // Pristine constant to compare the live scene against. When provided,
  // each card / tab / sub-textarea gets a diff marker pinpointing where
  // the user's edits land. Omit for the custom slot — there's no
  // pristine to compare against and every field would otherwise show
  // as "new".
  pristine?: StructuredScene;
  onChange: (next: StructuredScene) => void;
  onReset?: () => void;
  onClose: () => void;
  resetLabel?: string;
}) {
  const [tab, setTab] = useState<Tab>("base");
  const diff = pristine ? computeSceneDiff(scene, pristine) : null;

  // Lock body scroll while modal is open.
  useEffect(() => {
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, []);

  // Tracks where a mousedown landed before the next click fires. A
  // "click" event in browsers targets the common ancestor of mousedown
  // and mouseup, so dragging a text selection that starts inside the
  // dialog and releases on the backdrop would otherwise fire a click
  // on the backdrop and close the modal. We only treat a backdrop
  // click as a close request when the mousedown ALSO landed on the
  // backdrop itself.
  const mouseDownOnBackdropRef = useRef(false);

  // Close on Escape (but not while a textarea / input has focus, so the
  // user can use Escape to cancel native browser behaviours first).
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

  const renameLayerVersion = (
    layer: LayerName,
    oldKey: string,
    newKey: string,
  ) => {
    if (oldKey === DEFAULT_LAYER_VERSION || oldKey === newKey) return;
    const reg = renameKey(
      scene[layer] as Record<string, unknown>,
      oldKey,
      newKey,
    );
    const field = LAYER_TO_FIELD[layer];
    const events = scene.events.map((e) =>
      (e[field] ?? DEFAULT_LAYER_VERSION) === oldKey
        ? { ...e, [field]: newKey }
        : e,
    );
    onChange({ ...scene, [layer]: reg, events } as StructuredScene);
  };

  const setBase = (next: LayerRegistry<string>) =>
    onChange({ ...scene, base: next });
  const setCamera = (next: LayerRegistry<ShotVariant>) =>
    onChange({ ...scene, camera: next });
  const setMovement = (next: LayerRegistry<ShotVariant>) =>
    onChange({ ...scene, movement: next });
  const setJumpPrompt = (next: string) =>
    onChange({ ...scene, jumpPrompt: next });
  const setCrouchPrompt = (next: string) =>
    onChange({ ...scene, crouchPrompt: next });
  const setStandPrompt = (next: string) =>
    onChange({ ...scene, standPrompt: next });

  const counts: Record<Tab, number> = {
    base: Object.keys(scene.base).length,
    camera: Object.keys(scene.camera).length,
    movement: Object.keys(scene.movement).length,
    vertical: (scene.jumpPrompt ? 1 : 0) + (scene.crouchPrompt ? 1 : 0),
    events: scene.events.length,
    preview: 0,
  };

  // Per-tab edit counts. Counts visible cards that have any diff
  // (added or edited); the eventsRemoved-only case still surfaces a
  // tab-level mark since the Events panel renders an inline notice.
  const editCounts: Record<Tab, number> = {
    base: diff ? diff.base.added.size + diff.base.edited.size : 0,
    camera: diff
      ? diff.camera.added.size +
        diff.camera.staticChanged.size +
        diff.camera.dynamicChanged.size
      : 0,
    movement: diff
      ? diff.movement.added.size +
        diff.movement.staticChanged.size +
        diff.movement.dynamicChanged.size
      : 0,
    vertical: 0,
    events: diff
      ? diff.events.filter((e) => e.state !== "same").length +
        diff.eventsRemovedCount
      : 0,
    preview: 0,
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4 sm:p-6"
      onMouseDown={(e) => {
        mouseDownOnBackdropRef.current = e.target === e.currentTarget;
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget && mouseDownOnBackdropRef.current) {
          onClose();
        }
        mouseDownOnBackdropRef.current = false;
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        onClick={(e) => e.stopPropagation()}
        className="relative flex h-full max-h-[92vh] w-full max-w-7xl flex-col overflow-hidden rounded-xl border border-white/10 bg-neutral-950 shadow-2xl"
      >
        {/* Header */}
        <div className="flex items-center gap-4 border-b border-white/10 px-6 py-4">
          <div className="flex flex-col gap-0.5 min-w-0">
            <h2 className="font-mono text-sm uppercase tracking-widest text-white truncate">
              {title ?? "Edit layered scene"}
            </h2>
            {subtitle && (
              <span className="font-mono text-[10px] text-white/45 truncate">
                {subtitle}
              </span>
            )}
          </div>
          <span className="font-mono text-[10px] text-white/35 shrink-0">
            Auto-saved to browser
          </span>
          <div className="ml-auto flex items-center gap-2 shrink-0">
            {onReset && (
              <Button
                size="sm"
                variant="ghost"
                onClick={onReset}
                title="Discard your edits and revert this example to its pristine prompt"
              >
                {resetLabel ?? "Reset"}
              </Button>
            )}
            <Button size="sm" onClick={onClose}>
              Done
            </Button>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex shrink-0 border-b border-white/10 px-6 overflow-x-auto">
          {TABS.map((t) => {
            const active = tab === t.id;
            const edits = editCounts[t.id];
            return (
              <button
                key={t.id}
                type="button"
                onClick={() => setTab(t.id)}
                className={cn(
                  "relative px-4 py-3 font-mono text-[11px] uppercase tracking-wider transition-colors flex items-center gap-1.5",
                  active ? "text-white" : "text-white/45 hover:text-white/75",
                )}
              >
                <span>{t.label}</span>
                {t.id !== "preview" && (
                  <span
                    className={cn(
                      "inline-flex h-4 min-w-4 items-center justify-center rounded px-1 text-[9px]",
                      active
                        ? "bg-amber-300/20 text-amber-200"
                        : "bg-white/10 text-white/55",
                    )}
                  >
                    {counts[t.id]}
                  </span>
                )}
                {edits > 0 && (
                  <span
                    title={`${edits} edit${edits === 1 ? "" : "s"} vs pristine in this tab`}
                    className="inline-flex h-4 min-w-4 items-center justify-center rounded px-1 text-[9px] bg-amber-300/25 text-amber-200 border border-amber-300/40"
                  >
                    {edits}
                  </span>
                )}
                {active && (
                  <span className="absolute inset-x-2 bottom-0 h-px bg-amber-300" />
                )}
              </button>
            );
          })}
        </div>

        {/* Body — outer overflow handles the case where total content
            exceeds viewport (e.g. many version cards); the inner wrapper
            is flex-column with min-h-full so panels can stretch their
            single textarea card to fill the canvas when content is short. */}
        <div className="flex-1 min-h-0 overflow-y-auto px-6 py-5">
          <div className="flex flex-col min-h-full">
            {tab === "base" && (
              <BasePanel
                registry={scene.base}
                diff={diff?.base ?? null}
                onChange={setBase}
                onRename={(o, n) => renameLayerVersion("base", o, n)}
              />
            )}
            {tab === "camera" && (
              <ShotPanel
                layer="camera"
                registry={scene.camera}
                diff={diff?.camera ?? null}
                onChange={setCamera}
                onRename={(o, n) => renameLayerVersion("camera", o, n)}
              />
            )}
            {tab === "movement" && (
              <ShotPanel
                layer="movement"
                registry={scene.movement}
                diff={diff?.movement ?? null}
                onChange={setMovement}
                onRename={(o, n) => renameLayerVersion("movement", o, n)}
              />
            )}
            {tab === "vertical" && (
              <div className="flex flex-col gap-5 max-w-2xl">
                <Hint>
                  Sentences appended to the prompt while the vertical control is
                  held — <strong>Jump</strong> (Space, up) and{" "}
                  <strong>Crouch</strong> (C, down). Authored per scene so the
                  motion reads in context. Leave blank to append nothing.
                </Hint>
                <div className="flex flex-col gap-2">
                  <FieldLabel>Jump prompt (Space)</FieldLabel>
                  <textarea
                    value={scene.jumpPrompt ?? ""}
                    onChange={(e) => setJumpPrompt(e.target.value)}
                    rows={2}
                    placeholder="e.g. The dragon beats its wings and surges upward."
                    className="w-full rounded border border-white/15 bg-white/5 px-3 py-2 font-mono text-xs text-white/90 placeholder-white/25 focus:outline-none focus:border-white/40 resize-y"
                  />
                </div>
                <div className="flex flex-col gap-2">
                  <FieldLabel>Crouch prompt (C held)</FieldLabel>
                  <textarea
                    value={scene.crouchPrompt ?? ""}
                    onChange={(e) => setCrouchPrompt(e.target.value)}
                    rows={2}
                    placeholder="e.g. The dragon folds its wings and dives downward."
                    className="w-full rounded border border-white/15 bg-white/5 px-3 py-2 font-mono text-xs text-white/90 placeholder-white/25 focus:outline-none focus:border-white/40 resize-y"
                  />
                </div>
                <div className="flex flex-col gap-2">
                  <FieldLabel>Stand-up prompt (C release)</FieldLabel>
                  <textarea
                    value={scene.standPrompt ?? ""}
                    onChange={(e) => setStandPrompt(e.target.value)}
                    rows={2}
                    placeholder="e.g. The dragon spreads its wings and rises back up."
                    className="w-full rounded border border-white/15 bg-white/5 px-3 py-2 font-mono text-xs text-white/90 placeholder-white/25 focus:outline-none focus:border-white/40 resize-y"
                  />
                </div>
              </div>
            )}
            {tab === "events" && (
              <EventsPanel scene={scene} diff={diff} onChange={onChange} />
            )}
            {tab === "preview" && <PreviewPanel scene={scene} />}
          </div>
        </div>
      </div>
    </div>
  );
}
