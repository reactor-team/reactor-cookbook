"use client";

import { useHelios } from "@reactor-models/helios";

// The status badge teaches the four-state connection machine:
//   disconnected → connecting → waiting → ready
//
// We surface every state visibly. Beginners learning the SDK should
// *see* the transitions, not have them hidden behind a spinner.
const TONE: Record<string, { dot: string; label: string }> = {
  disconnected: { dot: "bg-zinc-500", label: "Disconnected" },
  connecting: { dot: "bg-amber-400 animate-pulse", label: "Connecting…" },
  waiting: { dot: "bg-amber-400 animate-pulse", label: "Waiting for GPU…" },
  ready: { dot: "bg-active", label: "Connected" },
};

export function StatusBadge() {
  const { status, lastError, connect, disconnect } = useHelios();
  const tone = TONE[status] ?? TONE.disconnected;
  const idle = status === "disconnected";

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className={`h-2 w-2 rounded-full ${tone.dot}`} />
          <span className="text-sm text-zinc-200">{tone.label}</span>
        </div>
        {idle ? (
          <button
            onClick={() => connect()}
            className="rounded-md bg-brand px-3 py-1 text-xs font-medium text-brand-fg hover:opacity-90"
          >
            Connect
          </button>
        ) : (
          <button
            onClick={() => disconnect()}
            className="rounded-md border border-zinc-700 px-3 py-1 text-xs text-zinc-300 hover:bg-zinc-800"
          >
            Disconnect
          </button>
        )}
      </div>

      {lastError && (
        <p className="mt-2 text-xs text-red-400">{lastError.message}</p>
      )}
    </div>
  );
}
