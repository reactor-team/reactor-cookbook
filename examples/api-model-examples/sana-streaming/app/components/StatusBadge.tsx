"use client";

import { useSanaStreaming } from "@reactor-models/sana-streaming";
import { Button, Panel, cn } from "./ui";

// The status badge teaches the four-state connection machine:
//   disconnected -> connecting -> waiting -> ready
//
// We surface every state visibly. Beginners learning the SDK should
// *see* the transitions, not have them hidden behind a spinner.
const TONE: Record<string, { dot: string; label: string }> = {
  disconnected: { dot: "bg-zinc-500", label: "Disconnected" },
  connecting: { dot: "bg-blue-500 animate-pulse", label: "Connecting…" },
  waiting: { dot: "bg-blue-500 animate-pulse", label: "Waiting for GPU…" },
  ready: { dot: "bg-active animate-pulse", label: "Connected" },
};

export function StatusBadge() {
  const { status, lastError, connect, disconnect } = useSanaStreaming();

  const tone = TONE[status] ?? TONE.disconnected;
  const idle = status === "disconnected";

  return (
    <Panel>
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className={cn("h-2 w-2 rounded-full", tone.dot)} />
          <span className="text-sm text-zinc-200">{tone.label}</span>
        </div>
        {idle ? (
          <Button variant="primary" size="sm" onClick={() => connect()}>
            Connect
          </Button>
        ) : (
          <Button variant="secondary" size="sm" onClick={() => disconnect()}>
            Disconnect
          </Button>
        )}
      </div>

      {lastError && (
        <p className="mt-2 text-xs text-red-400">{lastError.message}</p>
      )}
    </Panel>
  );
}
