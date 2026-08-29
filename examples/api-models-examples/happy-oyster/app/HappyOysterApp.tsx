"use client";

// The fixed shell: header on top, control sidebar beside the content screen —
// the layout every Reactor example uses, and it never changes shape. Nothing
// here navigates or goes full screen: useWorldSession() reduces the SDK's
// authoritative snapshot to one AppView (lib/view.ts) and the two regions
// switch what they show off it, so the page always mirrors the session's
// state machine.
//
// The experience mode is fixed for the life of a session (each mode is its own
// Reactor model), so the pending intent — and the mode it implies — is owned
// here, above the provider, and the provider is keyed on the mode: picking a
// world of the other experience remounts a fresh session. Nothing connects
// until you hit Connect or pick a world.

import { useCallback, useState } from "react";
import type { HappyOysterMode } from "@reactor-models/happy-oyster";
import type { WorldIntent } from "@/lib/worlds";
import { Header } from "@/components/Header";
import { LiveClientProvider } from "@/components/happy-oyster/ho-client";
import { useWorldSession } from "@/components/happy-oyster/use-world-session";
import { Sidebar } from "@/components/happy-oyster/Sidebar";
import { Screen } from "@/components/happy-oyster/Screen";

export function HappyOysterApp() {
  const [mode, setMode] = useState<HappyOysterMode>("adventure");
  const [intent, setIntent] = useState<WorldIntent | null>(null);

  // A new intent sets both the mode (which model to connect to) and the intent.
  // Leaving an intent keeps the mode, so returning to the same experience
  // reuses the session instead of remounting it.
  const run = useCallback((next: WorldIntent) => {
    setMode(next.mode);
    setIntent(next);
  }, []);
  const clearIntent = useCallback(() => setIntent(null), []);

  return (
    <LiveClientProvider mode={mode} key={mode}>
      <Shell intent={intent} onRun={run} onClearIntent={clearIntent} />
    </LiveClientProvider>
  );
}

function Shell({
  intent,
  onRun,
  onClearIntent,
}: {
  intent: WorldIntent | null;
  onRun: (intent: WorldIntent) => void;
  onClearIntent: () => void;
}) {
  const session = useWorldSession({ intent, onRun, onClearIntent });
  return (
    <div className="flex h-dvh flex-col bg-zinc-950">
      <Header />
      <main className="flex w-full min-h-0 flex-1 flex-col gap-4 p-4 max-lg:overflow-y-auto sm:p-6 lg:flex-row lg:gap-6">
        <Sidebar session={session} />
        <Screen session={session} />
      </main>
    </div>
  );
}
