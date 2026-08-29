"use client";

import { useEffect, useRef, useState } from "react";
import type { ReactorStatus } from "@reactor-team/js-sdk";
import { isQueued, validCommands } from "@/app/lib/machine";
import {
  FIELD_WIRE_NAME,
  FRAME_HEIGHT,
  FRAME_WIDTH,
  MAX_PROMPT_CHARS,
  MAX_SCRIPT_CHARS,
  type Ltx2UiState,
  type TakeEdit,
  type TakeFields,
} from "@/app/lib/types";
import { Panel } from "./ui";
import { CropModal } from "./CropModal";
import type { NoticeValue } from "./Notice";

// How long a free-text field must sit idle before it commits on its own.
// Long enough that ordinary typing does not fire commands mid-sentence,
// short enough that reaching for Start finds the field already sent.
const IDLE_COMMIT_MS = 600;

/**
 * What each field's local buffer holds, which is NOT always the wire type.
 *
 * The two number inputs can hold `""`: a number input reports an empty string
 * for anything it cannot parse — a cleared box, but also a lone `-` on the way
 * to a negative seed — and `Number("")` is 0. Coercing there would rewrite the
 * box to "0" under the cursor and eventually commit a value nobody typed. It
 * matters most for `duration_seconds`, where 0 is a real, meaningful value
 * ("derive the length from the script"), so an emptied box and a deliberate 0
 * have to stay distinguishable all the way to the commit.
 */
type TakeBuffers = Omit<TakeFields, "duration_seconds" | "seed"> & {
  duration_seconds: number | "";
  seed: number | "";
};

/** A number input's raw value, with "" preserved as "still editing". */
function numberBuffer(raw: string): number | "" {
  return raw === "" ? "" : Number(raw);
}

// Direct-the-take panel.
//
// Every edit commits straight to the wire, idle or not: each blur or slider
// release is a real `set_*` command, and the model accepts all of them at any
// time. While a take is generating it applies the change to the NEXT take and
// names the field in `state_update.queued_changes` — which is where the
// "queued" chips come from. The panel holds no pending-edit state of its own;
// the queue belongs to the model, and this just renders it.
//
// The one edit that cannot be sent is a blank script, which `set_script`
// refuses; see the blur handler on the textarea for what happens instead.
export function TakePanel({
  status,
  ui,
  imagePending,
  onCommit,
  onAvatarImage,
  onNotice,
}: {
  status: ReactorStatus;
  ui: Ltx2UiState;
  imagePending: boolean;
  // Each `set_*` resolves with its own acceptance message, so the union has
  // nothing narrower in common. The panel only awaits the send, never reads
  // the reply — the snapshot that follows is what it renders from.
  onCommit: (edit: TakeEdit) => Promise<unknown>;
  onAvatarImage: (file: File | Blob, name: string) => Promise<boolean>;
  onNotice: (notice: NoticeValue) => void;
}) {
  const valid = validCommands(status, ui);
  const connected = status === "ready";
  // The deployment reports its own pace bounds; they are configurable, so the
  // slider follows the snapshot rather than a constant baked in here.
  const { wpmMin, wpmMax } = ui;

  // Local editing buffers, resynced from the model's snapshot whenever a field
  // is not being edited — so the model always wins for clean
  // fields, and never clobbers a field mid-keystroke.
  const [script, setScript] = useState("");
  const [prompt, setPrompt] = useState(ui.prompt);
  const [wpm, setWpm] = useState(ui.wpm);
  const [duration, setDuration] = useState<number | "">(ui.durationSeconds);
  const [seed, setSeed] = useState<number | "">(ui.seed);
  const dirty = useRef<Set<keyof TakeFields>>(new Set());

  // Resync a field from the snapshot whenever the user is not mid-edit on it.
  // A queued value is already the value in the snapshot, so no special case is
  // needed for a run in flight: what the model will use next is what shows.
  useEffect(() => {
    const clean = (f: keyof TakeFields) => !dirty.current.has(f);
    if (clean("script")) setScript(ui.script ?? "");
    if (clean("prompt")) setPrompt(ui.prompt);
    if (clean("wpm")) setWpm(ui.wpm);
    if (clean("duration_seconds")) setDuration(ui.durationSeconds);
    if (clean("seed")) setSeed(ui.seed);
  }, [ui]);

  // The dirty set is this panel's entire concurrency story, so the rule it
  // encodes has to be exact: a field is dirty only while it is being edited,
  // and every edit ends at a resolve point. There are two, and between them
  // they have to cover every way an edit can end — `commit` sends the value,
  // `revert` abandons it and puts the model's value back — because a flag left
  // set silently suppresses the resync above for the rest of the session. The
  // field then keeps showing a value the model never received, with nothing on
  // screen to say so. Blur resolves every field unconditionally, which is what
  // makes the coverage total: focus always leaves eventually.
  function edit<K extends keyof TakeBuffers>(
    field: K,
    setter: (v: TakeBuffers[K]) => void,
  ) {
    return (value: TakeBuffers[K]) => {
      dirty.current.add(field);
      setter(value);
    };
  }

  function commit(change: TakeEdit) {
    if (!connected) {
      // Nothing can reach the model, so the edit cannot stand. Reverting
      // rather than returning early keeps the flag from outliving the
      // disconnection — the field would otherwise be frozen out of the resync
      // even after reconnecting.
      revert(change.field);
      return;
    }
    dirty.current.delete(change.field);
    void onCommit(change).catch(() => {});
  }

  // Abandon an edit. Restoring the buffer has to be explicit: clearing the
  // flag alone only re-arms the resync effect, which does not run again until
  // the next snapshot arrives, so the abandoned value would sit on screen
  // until something unrelated happened to change the model's state.
  function revert(field: keyof TakeFields) {
    dirty.current.delete(field);
    switch (field) {
      case "script":
        setScript(ui.script ?? "");
        break;
      case "prompt":
        setPrompt(ui.prompt);
        break;
      case "wpm":
        setWpm(ui.wpm);
        break;
      case "duration_seconds":
        setDuration(ui.durationSeconds);
        break;
      case "seed":
        setSeed(ui.seed);
        break;
    }
  }

  // Resolve a number field: a real number commits, an empty or unparseable box
  // is an abandoned edit rather than a value change.
  function resolveNumber(
    field: "seed" | "duration_seconds",
    value: number | "",
  ) {
    if (!dirty.current.has(field)) return;
    if (typeof value !== "number" || !Number.isFinite(value)) {
      revert(field);
      return;
    }
    commit(
      field === "seed"
        ? { field: "seed", value }
        : { field: "duration_seconds", value },
    );
  }

  // Commit the pace on release rather than on every drag step: each commit is
  // a real command on the wire, and a drag would otherwise send one per pixel.
  // Keyboard use needs its own release — arrow keys fire `change` with no
  // mouse or touch end at all — and keyup is the right one: holding a key
  // auto-repeats `change` but produces a single keyup for the whole burst, so
  // a held arrow still costs one command. Blur is the backstop for anything
  // that moves focus before a release lands.
  function commitWpm() {
    if (dirty.current.has("wpm")) commit({ field: "wpm", value: wpm });
  }

  // Blur alone is not a sound commit point for the free-text fields, and for
  // the script it deadlocks outright. `start` is gated on the model's `ready`,
  // which needs the script to have been SENT — but the natural next action
  // after typing one is to press Start, and a disabled button swallows the
  // mousedown that would have blurred the textarea. The script never commits,
  // `ready` never turns true, and the button stays dead no matter how many
  // times it is clicked; the only escape is clicking some unrelated element
  // first. Committing after a short idle breaks that without sending a
  // command per keystroke. Blur still commits, so leaving a field early is
  // still immediate.
  useEffect(() => {
    if (!connected) return;
    const pending = (["script", "prompt"] as const).filter((f) =>
      dirty.current.has(f),
    );
    if (pending.length === 0) return;
    const timer = setTimeout(() => {
      for (const field of pending) {
        if (!dirty.current.has(field)) continue;
        if (field === "script") {
          // A blank script is the one value `set_script` refuses outright, and
          // this timer fires mid-edit: someone who just cleared the box to
          // retype it is still editing, not asking for anything. So leave the
          // field dirty and let blur resolve it, which is the resolve point
          // that cannot be skipped.
          if (!script.trim()) continue;
          commit({ field: "script", value: script });
        } else {
          commit({ field: "prompt", value: prompt });
        }
      }
    }, IDLE_COMMIT_MS);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [script, prompt, connected]);

  // The picked file goes through the crop modal first: the model fits uploads
  // to its 640×352 canvas, which decapitates a tall portrait.
  const [cropFile, setCropFile] = useState<File | null>(null);

  function onPortraitPicked(file: File | undefined) {
    if (!file) return;
    // `set_avatar_image` is valid whenever there is a session — mid-run it
    // queues for the next take like any other condition — so in practice this
    // only fires when the session has gone away underneath the panel.
    if (!valid.has("set_avatar_image")) {
      onNotice({
        kind: "error",
        text: "Not connected — press Connect before setting an avatar image.",
      });
      return;
    }
    setCropFile(file);
  }

  // One upload at a time. Each `set_avatar_image` gets its own correlated
  // reply, so two in flight would not confuse each other's confirmations —
  // but `imagePending` is one boolean, and the first upload to finish lowers
  // it while the second is still decoding, releasing the hold on Start early.
  // Holding the control shut is what makes the single-in-flight assumption
  // the flag is written against true rather than merely hoped for.
  const canUpload = valid.has("set_avatar_image") && !imagePending;

  const words = script.split(/\s+/).filter(Boolean).length;
  const derivedSeconds = wpm > 0 ? (words / wpm) * 60 : 0;
  const queuedCount = ui.queuedChanges.length;
  // While the duration box sits empty the model is still holding its own
  // value, so the hint under it describes the model, not the empty box.
  const pinnedSeconds = duration === "" ? ui.durationSeconds : duration;

  return (
    <Panel label="Direct the take">
      <div className="flex flex-col gap-3.5">
        {/* Avatar image */}
        <div>
          <FieldLabel label="Avatar image" />
          <div className="mt-1.5 flex items-center gap-2">
            <span
              className={`h-2 w-2 rounded-full ${
                ui.hasAvatarImage ? "bg-brand" : "bg-white/20"
              }`}
            />
            <span className="text-xs text-zinc-400">
              {imagePending
                ? "Confirming…"
                : ui.hasAvatarImage
                  ? "Image set"
                  : "No image yet"}
            </span>
            <label
              className={`ml-auto rounded-md border border-edge px-2 py-1 text-xs ${
                canUpload
                  ? "cursor-pointer text-zinc-300 hover:border-zinc-600"
                  : "cursor-not-allowed opacity-30"
              }`}
            >
              Upload
              <input
                type="file"
                accept="image/*"
                className="hidden"
                disabled={!canUpload}
                onChange={(e) => {
                  onPortraitPicked(e.target.files?.[0]);
                  e.target.value = "";
                }}
              />
            </label>
          </div>
          <p className="mt-1 text-xs text-zinc-600">
            Cropped to {FRAME_WIDTH}×{FRAME_HEIGHT} — drag to frame.
          </p>
        </div>

        {/* Script */}
        <div>
          <FieldLabel
            label="Script"
            queued={isQueued(ui, FIELD_WIRE_NAME.script)}
          />
          <textarea
            value={script}
            maxLength={MAX_SCRIPT_CHARS}
            disabled={!connected}
            onChange={(e) => edit("script", setScript)(e.target.value)}
            onBlur={() => {
              if (!dirty.current.has("script")) return;
              if (!script.trim()) {
                // `set_script` answers a blank script with `command_error`, so
                // an emptied box is not a state the model can be put into. The
                // model is still holding the old words and would speak them at
                // the next Start, so show what it actually holds and say why
                // the text came back rather than leave the two disagreeing.
                const held = ui.script ?? "";
                revert("script");
                if (held.trim()) {
                  onNotice({
                    kind: "info",
                    text: "The model does not accept an empty script, so the previous one is still set — type over it to replace it.",
                  });
                }
                return;
              }
              commit({ field: "script", value: script });
            }}
            rows={6}
            placeholder="What the avatar says…"
            className="mt-1.5 w-full resize-y rounded-md border border-edge bg-black/40 px-2.5 py-2 text-sm leading-relaxed outline-none placeholder:text-zinc-600 focus:border-brand/50 disabled:opacity-40"
          />
          <p className="mt-1 flex justify-between text-xs text-zinc-500">
            <span>{words} words</span>
            <span>
              ≈{derivedSeconds.toFixed(0)}s at {wpm} wpm
              {ui.durationSeconds > 0 &&
                ` · pinned ${ui.durationSeconds.toFixed(0)}s`}
            </span>
          </p>
        </div>

        {/* Prompt */}
        <div>
          <FieldLabel
            label="Scene prompt"
            queued={isQueued(ui, FIELD_WIRE_NAME.prompt)}
          />
          <textarea
            value={prompt}
            maxLength={MAX_PROMPT_CHARS}
            disabled={!connected}
            onChange={(e) => edit("prompt", setPrompt)(e.target.value)}
            onBlur={() => {
              if (dirty.current.has("prompt"))
                commit({ field: "prompt", value: prompt });
            }}
            rows={2}
            className="mt-1.5 w-full resize-y rounded-md border border-edge bg-black/40 px-2.5 py-2 text-sm leading-relaxed outline-none focus:border-brand/50 disabled:opacity-40"
          />
        </div>

        {/* WPM */}
        <div>
          <FieldLabel
            label={`Pace — ${wpm} wpm`}
            queued={isQueued(ui, FIELD_WIRE_NAME.wpm)}
          />
          <input
            type="range"
            min={wpmMin}
            max={wpmMax}
            value={wpm}
            disabled={!connected}
            onChange={(e) => edit("wpm", setWpm)(Number(e.target.value))}
            onMouseUp={commitWpm}
            onTouchEnd={commitWpm}
            onKeyUp={commitWpm}
            onBlur={commitWpm}
            className="mt-2 w-full disabled:opacity-40"
          />
          <div className="flex justify-between text-xs text-zinc-600">
            <span>{wpmMin}</span>
            <span>{wpmMax}</span>
          </div>
        </div>

        {/* Seed + duration */}
        <div className="grid grid-cols-2 gap-3">
          <div>
            <FieldLabel
              label="Seed"
              queued={isQueued(ui, FIELD_WIRE_NAME.seed)}
            />
            <div className="mt-1.5 flex gap-1.5">
              <input
                type="number"
                value={seed}
                disabled={!connected}
                onChange={(e) =>
                  edit("seed", setSeed)(numberBuffer(e.target.value))
                }
                onBlur={() => resolveNumber("seed", seed)}
                className="w-full rounded-md border border-edge bg-black/40 px-2 py-1.5 font-mono text-xs outline-none focus:border-brand/50 disabled:opacity-40"
              />
              <button
                disabled={!connected}
                title="Roll a new seed — another take, same setup"
                onClick={() => {
                  const next = Math.floor(Math.random() * 2 ** 31);
                  setSeed(next);
                  commit({ field: "seed", value: next });
                }}
                className="rounded-md border border-edge px-2 text-sm text-zinc-300 hover:border-zinc-600 disabled:opacity-30"
              >
                ⚄
              </button>
            </div>
          </div>
          <div>
            <FieldLabel
              label="Duration"
              queued={isQueued(ui, FIELD_WIRE_NAME.duration_seconds)}
            />
            <input
              type="number"
              min={0}
              value={duration}
              disabled={!connected}
              title="0 = derive the length from the script at the current pace"
              onChange={(e) =>
                edit(
                  "duration_seconds",
                  setDuration,
                )(numberBuffer(e.target.value))
              }
              onBlur={() => resolveNumber("duration_seconds", duration)}
              className="mt-1.5 w-full rounded-md border border-edge bg-black/40 px-2 py-1.5 font-mono text-xs outline-none focus:border-brand/50 disabled:opacity-40"
            />
            <p className="mt-1 text-xs text-zinc-600">
              {pinnedSeconds > 0 ? "Pinned" : "Auto from script"}
            </p>
          </div>
        </div>

        {/* No apply button: the model holds the queue, and it applies on its
            own at the next `start`. Nothing here is waiting to be flushed. */}
        {queuedCount > 0 && (
          <p className="border-t border-edge pt-2.5 text-xs leading-relaxed text-zinc-500">
            <span className="text-brand-light">
              {queuedCount} change{queuedCount > 1 ? "s" : ""} queued
            </span>{" "}
            — the take you are watching is unchanged. These apply when you press
            Start for the next one.
          </p>
        )}
      </div>

      {cropFile && (
        <CropModal
          file={cropFile}
          onConfirm={(blob, name) => {
            setCropFile(null);
            // Fire and forget, but not fire and ignore. `onAvatarImage`
            // reports a refusal or an unconfirmed upload itself and resolves
            // false; a rejection is the one outcome it cannot describe — the
            // upload or the file read failed before there was anything to
            // confirm — and swallowing it leaves the avatar unchanged with no
            // sign that anything went wrong.
            void onAvatarImage(blob, name).catch(() => {
              onNotice({
                kind: "error",
                text: "The avatar image failed to upload — check the connection and try again.",
              });
            });
          }}
          onCancel={() => setCropFile(null)}
        />
      )}
    </Panel>
  );
}

function FieldLabel({ label, queued }: { label: string; queued?: boolean }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-zinc-400">{label}</span>
      {queued && (
        <span className="rounded-full border border-brand/50 px-1.5 py-px text-[10px] text-brand-light">
          queued
        </span>
      )}
    </div>
  );
}
