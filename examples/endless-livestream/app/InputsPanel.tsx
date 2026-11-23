"use client";

import { useRef } from "react";
import {
  MAX_AUDIO,
  MAX_AUDIO_SECONDS,
  MAX_FILE_BYTES,
  MAX_IMAGES,
  MAX_VIDEOS,
  MAX_VIDEO_SECONDS,
  mediaSeconds,
  type Inputs,
  type Item,
} from "./lib/inputs";

type Tab = "references" | "frames";
type Slot = "references" | "first" | "last";

const IMAGE_TYPES = "image/jpeg,image/png,image/webp";
const ACCEPT: Record<Slot, string> = {
  references: `${IMAGE_TYPES},audio/*,video/*`,
  first: IMAGE_TYPES,
  last: IMAGE_TYPES,
};

let nextId = 0;
const newItem = (file: File, seconds?: number): Item => ({ id: `in-${++nextId}`, file, url: URL.createObjectURL(file), seconds });
const total = (items: Item[]) => items.reduce((sum, i) => sum + (i.seconds ?? 0), 0);

/**
 * Pika's raw inputs in one row: references (images, audio and video behind a
 * single +) or frames (first and last). Limits are enforced quietly: the +
 * goes away when everything is full, and a file that breaks a rule says why.
 */
export function InputsPanel({
  inputs,
  onChange,
  tab,
  onTab,
  onError,
}: {
  inputs: Inputs;
  onChange: (next: Inputs) => void;
  tab: Tab;
  onTab: (tab: Tab) => void;
  onError: (message: string) => void;
}) {
  const picker = useRef<HTMLInputElement>(null);
  const target = useRef<Slot>("references");
  const full = inputs.images.length >= MAX_IMAGES && inputs.audios.length >= MAX_AUDIO && inputs.videos.length >= MAX_VIDEOS;

  function pick(slot: Slot) {
    target.current = slot;
    const el = picker.current;
    if (!el) return;
    el.accept = ACCEPT[slot];
    el.multiple = slot === "references";
    el.value = "";
    el.click();
  }

  /** Drop files straight onto a slot. */
  const drop = (slot: Slot) => ({
    onDragOver: (e: React.DragEvent) => e.preventDefault(),
    onDrop: (e: React.DragEvent) => {
      e.preventDefault();
      void add(e.dataTransfer.files, slot);
    },
  });

  async function add(files: FileList | null, slot: Slot = target.current) {
    if (!files?.length) return;
    let next = { ...inputs };
    for (const file of Array.from(files)) {
      if (file.size > MAX_FILE_BYTES) {
        onError(`${file.name} is over 25 MB.`);
        continue;
      }
      const isImage = IMAGE_TYPES.split(",").includes(file.type);
      if (slot === "first" || slot === "last") {
        if (!isImage) onError(`${file.name} is not a JPEG, PNG or WebP image.`);
        else next = { ...next, [slot]: newItem(file) };
        continue;
      }
      if (isImage) {
        if (next.images.length >= MAX_IMAGES) onError(`Pika takes up to ${MAX_IMAGES} images.`);
        else next = { ...next, images: [...next.images, newItem(file)] };
        continue;
      }
      const kind = file.type.startsWith("audio/") ? "audios" : file.type.startsWith("video/") ? "videos" : null;
      if (!kind) {
        onError(`${file.name} is not an image, audio or video file.`);
        continue;
      }
      const [max, maxSeconds, noun] = kind === "audios" ? [MAX_AUDIO, MAX_AUDIO_SECONDS, "audio clips"] : [MAX_VIDEOS, MAX_VIDEO_SECONDS, "videos"];
      if (next[kind].length >= max) {
        onError(`Pika takes up to ${max} ${noun}.`);
        continue;
      }
      const seconds = await mediaSeconds(file, kind === "audios" ? "audio" : "video").catch(() => NaN);
      if (!Number.isFinite(seconds) || seconds < 2 || seconds > 15) {
        onError(`${file.name} must be 2–15 seconds long.`);
        continue;
      }
      if (total(next[kind]) + seconds > maxSeconds + 0.01) {
        onError(`The ${noun} can total ${maxSeconds} s; ${file.name} would go over.`);
        continue;
      }
      next = { ...next, [kind]: [...next[kind], newItem(file, seconds)] };
    }
    onChange(next);
  }

  function remove(kind: "images" | "audios" | "videos" | "first" | "last", id?: string) {
    if (kind === "first" || kind === "last") {
      const item = inputs[kind];
      if (item) URL.revokeObjectURL(item.url);
      onChange({ ...inputs, [kind]: null });
      return;
    }
    const item = inputs[kind].find((i) => i.id === id);
    if (item) URL.revokeObjectURL(item.url);
    onChange({ ...inputs, [kind]: inputs[kind].filter((i) => i.id !== id) });
  }

  return (
    <div className="inputs">
      <input ref={picker} type="file" hidden onChange={(e) => void add(e.target.files)} />
      <div className="inputs-row">
        <div role="tablist" aria-label="Inputs" className="tabs">
          <button type="button" role="tab" aria-selected={tab === "references"} className="tab mono" onClick={() => onTab("references")}>References</button>
          <button type="button" role="tab" aria-selected={tab === "frames"} className="tab mono" onClick={() => onTab("frames")}>Frames</button>
        </div>

        {tab === "references" && (
          <div className="attachments">
            {inputs.images.map((item, n) => (
              <div key={item.id} className="thumb">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={item.url} alt={`Picture ${n + 1}: ${item.file.name}`} />
                <span className="badge mono" aria-hidden="true">{n + 1}</span>
                <button type="button" className="remove" aria-label={`Remove Picture ${n + 1}`} onClick={() => remove("images", item.id)}>✕</button>
              </div>
            ))}
            {inputs.audios.map((item, n) => (
              <div key={item.id} className="media" aria-label={`Audio ${n + 1}, ${item.seconds?.toFixed(0)} seconds`}>
                <svg width="30" height="16" viewBox="0 0 30 16" aria-hidden="true" fill="currentColor"><rect x="0" y="6" width="2" height="4" rx="1" /><rect x="4" y="3" width="2" height="10" rx="1" /><rect x="8" y="5" width="2" height="6" rx="1" /><rect x="12" y="1" width="2" height="14" rx="1" /><rect x="16" y="4" width="2" height="8" rx="1" /><rect x="20" y="2" width="2" height="12" rx="1" /><rect x="24" y="6" width="2" height="4" rx="1" /><rect x="28" y="5" width="2" height="6" rx="1" /></svg>
                <span>{n + 1} · {item.seconds?.toFixed(0)} s</span>
                <button type="button" className="remove" aria-label={`Remove Audio ${n + 1}`} onClick={() => remove("audios", item.id)}>✕</button>
              </div>
            ))}
            {inputs.videos.map((item, n) => (
              <div key={item.id} className="media" aria-label={`Video ${n + 1}, ${item.seconds?.toFixed(0)} seconds`}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><rect x="2" y="5" width="15" height="14" rx="2" /><path d="m17 10 5-3v10l-5-3" /></svg>
                <span>{n + 1} · {item.seconds?.toFixed(0)} s</span>
                <button type="button" className="remove" aria-label={`Remove Video ${n + 1}`} onClick={() => remove("videos", item.id)}>✕</button>
              </div>
            ))}
            {!full && (
              <button type="button" className="add-square" aria-label="Add an image, audio or video" onClick={() => pick("references")} {...drop("references")}>+</button>
            )}
          </div>
        )}
      </div>

      {tab === "frames" && (
        <div className="frames">
          <FrameSlot label="First frame" item={inputs.first} onAdd={() => pick("first")} onRemove={() => remove("first")} dropProps={drop("first")} />
          <FrameSlot label="Last frame" item={inputs.last} onAdd={() => pick("last")} onRemove={() => remove("last")} dropProps={drop("last")} />
        </div>
      )}
    </div>
  );
}

type DropProps = { onDragOver: (e: React.DragEvent) => void; onDrop: (e: React.DragEvent) => void };

function FrameSlot({ label, item, onAdd, onRemove, dropProps }: { label: string; item: Item | null; onAdd: () => void; onRemove: () => void; dropProps: DropProps }) {
  return (
    <div className="frame-group">
      <span className="mono">{label}</span>
      {item ? (
        <div className="frame">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={item.url} alt={`${label}: ${item.file.name}`} />
          <button type="button" className="remove" aria-label={`Remove ${label.toLowerCase()}`} onClick={onRemove}>✕</button>
        </div>
      ) : (
        <button type="button" className="frame frame-add" aria-label={`Add a ${label.toLowerCase()}`} onClick={onAdd} {...dropProps}>+</button>
      )}
    </div>
  );
}
