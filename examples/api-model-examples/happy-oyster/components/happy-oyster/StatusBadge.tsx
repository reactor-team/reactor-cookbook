"use client";

import { Button } from "@/components/ui/button";
import { useHappyOysterClient } from "./ho-client";

// The connection badge every Reactor example carries at the top of its
// sidebar: a visible session state plus an explicit Connect/Disconnect,
// so the disconnected → connecting → connected transitions are seen,
// not hidden behind a spinner.
//
// Connecting here is optional — picking a world connects automatically —
// but pre-connecting takes the session handshake out of the build wait.
const TONE: Record<string, { dot: string; label: string }> = {
  idle: { dot: "bg-white/30", label: "Disconnected" },
  connecting: { dot: "bg-amber-400 animate-pulse", label: "Connecting…" },
  connected: { dot: "bg-emerald-400", label: "Connected" },
  ended: { dot: "bg-white/30", label: "Disconnected" },
  failed: { dot: "bg-red-400", label: "Connection failed" },
};

export function StatusBadge({
  onDisconnect,
}: {
  /** Override the plain disconnect, e.g. to also drop the pending intent. */
  onDisconnect?: () => void;
}) {
  const { phase, lastError, connect, disconnect } = useHappyOysterClient();
  const tone = TONE[phase] ?? TONE.connected; // streaming phases read as connected
  const idle = phase === "idle" || phase === "ended" || phase === "failed";

  return (
    <div className="flex flex-col gap-2 rounded-xl border border-white/10 bg-white/[0.03] p-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className={`h-2 w-2 rounded-full ${tone.dot}`} />
          <span className="text-sm text-white/80">{tone.label}</span>
        </div>
        {idle ? (
          <Button size="sm" onClick={() => void connect().catch(() => {})}>
            Connect
          </Button>
        ) : (
          <Button
            size="sm"
            variant="ghost"
            onClick={() =>
              onDisconnect ? onDisconnect() : void disconnect().catch(() => {})
            }
          >
            Disconnect
          </Button>
        )}
      </div>
      {lastError && (
        <p className="break-words text-xs leading-relaxed text-red-300/90">
          {lastError}
        </p>
      )}
    </div>
  );
}
