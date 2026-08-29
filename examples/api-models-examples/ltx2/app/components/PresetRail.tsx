"use client";

import { useEffect, useState } from "react";
import type { ReactorStatus } from "@reactor-team/js-sdk";
import { PRESETS, type Preset } from "@/app/lib/presets";
import type { Ltx2UiState } from "@/app/lib/types";
import { Panel } from "./ui";

// Preset rail. Clicking a row fires the real command sequence — watch each
// command land in the take panel below as it goes. Presets are macros over the
// same form the take panel exposes, so there is nothing here you cannot also
// send by hand. See lib/presets.ts for the sequence.
export function PresetRail({
  status,
  ui,
  presetPending,
  onRun,
}: {
  status: ReactorStatus;
  ui: Ltx2UiState;
  presetPending: string | null;
  onRun: (preset: Preset) => void;
}) {
  const enabled =
    status === "ready" && !ui.generating && presetPending === null;

  return (
    <Panel label="Presets">
      <div className="flex flex-col">
        {PRESETS.map((preset) => (
          <PresetRow
            key={preset.id}
            preset={preset}
            disabled={!enabled}
            busy={presetPending === preset.id}
            onClick={() => onRun(preset)}
          />
        ))}
      </div>
    </Panel>
  );
}

/**
 * Probe the portrait client-side rather than relying on `<img onError>`: an
 * SSR'd img can fail before hydration, in which case the error event fires
 * before React is listening and the fallback never renders. Portraits are not
 * committed to this repo — see public/presets/README.md.
 */
function usePortraitOk(src: string): boolean {
  const [ok, setOk] = useState(false);
  useEffect(() => {
    let cancelled = false;
    const probe = new Image();
    probe.onload = () => !cancelled && setOk(true);
    probe.onerror = () => !cancelled && setOk(false);
    probe.src = src;
    return () => {
      cancelled = true;
    };
  }, [src]);
  return ok;
}

function PresetRow({
  preset,
  disabled,
  busy,
  onClick,
}: {
  preset: Preset;
  disabled: boolean;
  busy: boolean;
  onClick: () => void;
}) {
  const portraitOk = usePortraitOk(preset.portrait);

  return (
    <button
      onClick={onClick}
      disabled={disabled || busy}
      title={disabled ? "Stop the current take first" : `Direct ${preset.name}`}
      className="-mx-1.5 flex items-center gap-2.5 rounded-md px-1.5 py-1.5 text-left transition-colors hover:bg-white/5 disabled:opacity-40 disabled:hover:bg-transparent"
    >
      <span className="flex h-9 w-9 shrink-0 items-center justify-center overflow-hidden rounded-md bg-black/40">
        {portraitOk ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={preset.portrait}
            alt=""
            className="h-full w-full object-cover"
          />
        ) : (
          <span className="font-mono text-xs text-zinc-500">
            {preset.monogram}
          </span>
        )}
      </span>
      <span className="min-w-0">
        <span className="block truncate text-sm text-zinc-200">
          {busy ? "Preparing…" : preset.name}
        </span>
        <span className="block truncate text-xs text-zinc-500">
          {preset.hook}
        </span>
      </span>
    </button>
  );
}
