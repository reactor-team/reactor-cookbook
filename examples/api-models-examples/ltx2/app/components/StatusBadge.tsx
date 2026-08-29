"use client";

import { useLtx2 } from "@reactor-models/ltx2";
import type { ReactorStatus } from "@reactor-team/js-sdk";
import { Button, Panel, cn } from "./ui";

// The connection badge every Reactor example carries at the top of its
// sidebar, teaching the four-state machine:
//
//   disconnected → connecting → waiting → ready
//
// Here it is also the app's entry point. The provider does not autoConnect
// (see Ltx2App): a session holds a whole B200, so merely opening the
// page should not claim one, and the click is the user gesture the browser's
// autoplay policy requires before <video> may play with sound.
//
// A deployment may serve only one session at a time, so a second visitor gets
// a refused connect. The error is shown verbatim rather than pattern-matched
// into a friendlier guess — when you are learning the SDK, the real message is
// the useful one.
const TONE: Record<ReactorStatus, { dot: string; label: string }> = {
  disconnected: { dot: "bg-zinc-600", label: "Disconnected" },
  connecting: { dot: "bg-brand/60 animate-pulse", label: "Connecting…" },
  waiting: { dot: "bg-brand/60 animate-pulse", label: "Waiting for GPU…" },
  ready: { dot: "bg-brand", label: "Connected" },
};

export function StatusBadge({ status }: { status: ReactorStatus }) {
  const { lastError, connect, disconnect } = useLtx2();

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
          <Button variant="primary" size="sm" onClick={() => void connect()}>
            Connect
          </Button>
        ) : (
          <Button
            variant="secondary"
            size="sm"
            onClick={() => void disconnect()}
          >
            Disconnect
          </Button>
        )}
      </div>

      {lastError && (
        <p className="mt-2 text-xs leading-relaxed text-red-400">
          {lastError.message}
        </p>
      )}
    </Panel>
  );
}
