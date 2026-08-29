"use client";

// The content screen — the app's fixed sandbox, reserved for the world:
// every view renders into the same plain frame the travel video plays in
// (the generated first frame lives in the sidebar), and the live world stream
// plays on top.

import type { ReactNode } from "react";
import type { JourneyStep } from "@/lib/view";
import { useVideoSlot } from "./ho-client";
import type { WorldSession } from "./use-world-session";
import { Eyebrow, Spinner, WorldIdChip } from "./ui";

export function Screen({ session }: { session: WorldSession }) {
  const { view } = session;
  const videoSlot = useVideoSlot();

  // Loading spans every state between "picked a world" and "video is up";
  // the journey pane tracks the API's own machine through all of them.
  const loading =
    view.kind === "connecting" ||
    view.kind === "building" ||
    (view.kind === "traveling" && !view.live);

  // Idle it reads as another quiet panel (the sidebar's surface); it only
  // goes black once a world is loading or playing in it.
  const idle = view.kind === "browse";

  return (
    <section
      className={`relative order-1 aspect-video w-full overflow-hidden rounded-xl border lg:order-2 lg:aspect-auto lg:min-h-0 lg:min-w-0 lg:flex-1 ${
        idle
          ? "border-white/10 bg-white/[0.03]"
          : "border-white/[0.08] bg-black"
      }`}
    >
      {view.kind === "traveling" && videoSlot}
      {loading && <JourneyPane session={session} />}
      {view.kind === "ready" && <EndScene session={session} />}
      {view.kind === "error" && (
        <Overlay>
          <Eyebrow>
            {view.buildFailed ? "World build failed" : "Something broke"}
          </Eyebrow>
          <p className="max-w-md break-words px-6 text-sm leading-relaxed text-red-300/90">
            {view.message}
          </p>
        </Overlay>
      )}
    </section>
  );
}

// ── the loading journey ──────────────────────────────────────────────────────

// Not a spinner: the API's own machine, live. Each row is one step of the
// journey lib/view.ts derives from the session snapshot.
function JourneyPane({ session }: { session: WorldSession }) {
  const { journey } = session;
  return (
    <Overlay>
      <div className="flex w-full max-w-sm flex-col gap-3 px-6">
        {journey.map((step) => (
          <JourneyRow key={step.key} step={step} />
        ))}
      </div>
    </Overlay>
  );
}

function JourneyRow({ step }: { step: JourneyStep }) {
  return (
    <div className="flex items-center gap-3 text-left">
      <span className="flex h-4 w-4 shrink-0 items-center justify-center">
        {step.status === "done" ? (
          <svg
            viewBox="0 0 16 16"
            className="h-4 w-4 text-primary/90"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d="M3 8.5 6.5 12 13 4.5" />
          </svg>
        ) : step.status === "active" ? (
          <Spinner />
        ) : (
          <span className="h-1.5 w-1.5 rounded-full bg-white/20" />
        )}
      </span>
      <span
        className={`text-sm ${
          step.status === "active"
            ? "text-white"
            : step.status === "done"
              ? "text-white/60"
              : "text-white/30"
        }`}
      >
        {step.label}
      </span>
    </div>
  );
}

// ── the end scene ────────────────────────────────────────────────────────────

// Shown when a travel ends: the world outlives it. Its encrypted_world_id is
// a capability — save it and attach it anytime to skip the build.
function EndScene({ session }: { session: WorldSession }) {
  const worldId = session.client.worldState?.encrypted_world_id;
  return (
    <Overlay>
      <Eyebrow>Travel ended</Eyebrow>
      {session.intent && (
        <span className="text-lg font-medium tracking-tight text-white">
          {session.intent.title}
        </span>
      )}
      <p className="max-w-sm px-6 text-sm leading-relaxed text-white/50">
        This world is permanent. Save its id and attach it anytime to jump
        straight back in — no build wait.
      </p>
      {worldId && <WorldIdChip worldId={worldId} />}
    </Overlay>
  );
}

// ── shared layers ────────────────────────────────────────────────────────────

function Overlay({
  children,
  dim = true,
}: {
  children: ReactNode;
  dim?: boolean;
}) {
  return (
    <div
      className={`absolute inset-0 flex flex-col items-center justify-center gap-3 text-center ${
        dim ? "bg-black/40" : ""
      }`}
    >
      {children}
    </div>
  );
}
