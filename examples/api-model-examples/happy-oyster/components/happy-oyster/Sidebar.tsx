"use client";

// The control rail — always on screen, beside the content screen. What it
// offers tracks the reducer's view:
//   browse    → connection badge, featured worlds, composer, attach-by-id
//   building  → the in-flight intent, and Cancel
//   ready     → the world card (id capability, travel limit), Start travel
//   traveling → the countdown and the mode-matched control deck
//   error     → Try again / Back to worlds
// The countdown runs on the budget HappyOyster granted the live travel, sized
// from TRAVEL_SECONDS while the stream opens; at zero the travel ends
// client-side and the world stays ready for another run.

import { useEffect, useRef, useState } from "react";
import { TRAVEL_SECONDS } from "@/lib/worlds";
import { Button } from "@/components/ui/button";
import type { WorldSession } from "./use-world-session";
import { StatusBadge } from "./StatusBadge";
import { Gallery } from "./Gallery";
import { CustomCompose, AttachById } from "./Composer";
import { AdventureControls } from "./AdventureControls";
import { DirectingControls } from "./DirectingControls";
import { ModeBadge, SectionLabel, Spinner, WorldIdChip } from "./ui";

export function Sidebar({ session }: { session: WorldSession }) {
  const { view } = session;
  return (
    <aside className="order-2 flex w-full flex-col gap-4 lg:order-1 lg:w-80 lg:shrink-0 lg:overflow-y-auto lg:pr-1">
      {/* Disconnecting also drops the pending intent — back to browsing. */}
      <StatusBadge onDisconnect={session.exit} />
      {view.kind === "browse" && (
        <>
          <Gallery onIntent={session.run} />
          <CustomCompose onIntent={session.run} />
          <AttachById onIntent={session.run} />
        </>
      )}
      {(view.kind === "connecting" || view.kind === "building") && (
        <IntentCard session={session} />
      )}
      {view.kind === "ready" && <ReadyCard session={session} />}
      {view.kind === "traveling" && <TravelDeck session={session} />}
      {view.kind === "error" && <ErrorCard session={session} />}
      {view.kind !== "browse" &&
        session.client.worldState?.encrypted_world_id && (
          <WorldIdCard worldId={session.client.worldState.encrypted_world_id} />
        )}
      {view.kind !== "browse" &&
        view.kind !== "traveling" &&
        session.seedFrame && <SeedFrameCard src={session.seedFrame} />}
    </aside>
  );
}

// The id lands in world_state snapshots while the build is still running —
// the world is claimable before it's even done generating.
function WorldIdCard({ worldId }: { worldId: string }) {
  return (
    <div className="flex flex-col gap-2 rounded-xl border border-white/10 bg-white/[0.03] p-3">
      <SectionLabel>World id</SectionLabel>
      <WorldIdChip worldId={worldId} />
    </div>
  );
}

// The generated image that seeds the world — the model reports it as
// world_state.first_frame during the build, and every travel opens on it.
function SeedFrameCard({ src }: { src: string }) {
  return (
    <div className="flex flex-col gap-2 rounded-xl border border-white/10 bg-white/[0.03] p-3">
      <SectionLabel>First frame</SectionLabel>
      <img
        src={src}
        alt="The world's generated first frame"
        className="w-full rounded-md border border-white/[0.06]"
      />
    </div>
  );
}

// ── building ─────────────────────────────────────────────────────────────────

function IntentCard({ session }: { session: WorldSession }) {
  const { view, intent, client } = session;
  if (!intent) return null;
  const mode = intent.mode === "directing" ? 2 : 1;
  const prompt =
    client.worldState?.prompt ??
    (intent.kind === "create" ? intent.params.prompt : null);
  // The same journey the loading pane traces — one source for "where are we".
  const status =
    session.journey.find((step) => step.status === "active")?.label ??
    "Working…";
  return (
    <div className="flex flex-col gap-3 rounded-xl border border-white/10 bg-white/[0.03] p-4">
      <SectionLabel>
        {view.kind === "building" && view.restoring
          ? "Restoring world"
          : "Building world"}
      </SectionLabel>
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm font-medium text-white">{intent.title}</span>
        {mode != null && <ModeBadge mode={mode} />}
      </div>
      <div className="flex items-center gap-2.5 text-white/60">
        <Spinner />
        <span className="font-mono text-[11px] uppercase tracking-tight">
          {status}
        </span>
      </div>
      {prompt && (
        <p className="font-mono text-xs leading-relaxed text-white/35">
          {prompt}
        </p>
      )}
      <Button variant="ghost" onClick={session.exit}>
        Cancel
      </Button>
    </div>
  );
}

// ── between travels ──────────────────────────────────────────────────────────

// The travel is over but the world stays ready (the screen shows the end
// scene with its attach id); this card is just "what next".
function ReadyCard({ session }: { session: WorldSession }) {
  const { client, starting } = session;
  const worldState = client.worldState;
  const mode = (worldState?.mode === 2 ? 2 : 1) as 1 | 2;
  return (
    <div className="flex flex-col gap-3 rounded-xl border border-white/10 bg-white/[0.03] p-4">
      <div className="flex items-center justify-between gap-2">
        <SectionLabel>Travel ended</SectionLabel>
        <ModeBadge mode={worldState?.mode ?? null} />
      </div>
      <span className="font-mono text-[11px] uppercase tracking-tight text-white/40">
        {formatLimit(TRAVEL_SECONDS[mode])} per travel
      </span>
      <div className="flex flex-col gap-2">
        <Button onClick={session.beginTravel} disabled={starting}>
          {starting ? "Starting…" : "Travel again"}
        </Button>
        <Button variant="ghost" onClick={session.exit}>
          Back to worlds
        </Button>
      </div>
    </div>
  );
}

function formatLimit(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  return minutes >= 1 ? `${minutes} min` : `${seconds}s`;
}

// ── traveling ────────────────────────────────────────────────────────────────

function TravelDeck({ session }: { session: WorldSession }) {
  const { view, client } = session;
  const live = view.kind === "traveling" && view.live;
  const mode = (client.worldState?.mode === 2 ? 2 : 1) as 1 | 2;
  // The budget HappyOyster granted this travel, so the countdown expires with
  // the stream. It lands with the streaming phase — the same signal `live`
  // reads — so TRAVEL_SECONDS sizes the clock while the stream opens, and for
  // Directing travels.
  const totalSeconds = client.maxExperienceTimeSec ?? TRAVEL_SECONDS[mode];
  const secondsLeft = useTravelTimer(live, totalSeconds, () => {
    void client.endTravelSession().catch(() => {});
  });

  return (
    <>
      <div className="flex items-center justify-between gap-2 rounded-xl border border-white/10 bg-white/[0.03] p-3">
        <div className="flex items-center gap-2">
          <ModeBadge mode={mode} />
          <TravelClock secondsLeft={secondsLeft} />
        </div>
        <button
          onClick={() => void client.endTravelSession().catch(() => {})}
          className="inline-flex h-8 items-center justify-center rounded-md border border-white/15 bg-white/10 px-2.5 text-sm font-medium text-white/70 transition hover:border-red-400/40 hover:bg-red-500/20 hover:text-red-300"
        >
          End travel
        </button>
      </div>
      {session.seedFrame && <SeedFrameCard src={session.seedFrame} />}
      {live && (mode === 2 ? <DirectingControls /> : <AdventureControls />)}
    </>
  );
}

function useTravelTimer(
  active: boolean,
  totalSeconds: number,
  onExpire: () => void,
): number {
  const deadline = useRef<number | null>(null);
  const expired = useRef(false);
  const [left, setLeft] = useState(totalSeconds);
  const expireRef = useRef(onExpire);
  expireRef.current = onExpire;

  useEffect(() => {
    if (!active) {
      deadline.current = null;
      expired.current = false;
      setLeft(totalSeconds);
      return;
    }
    if (!deadline.current) deadline.current = Date.now() + totalSeconds * 1000;
    const tick = () => {
      const remaining = Math.max(
        0,
        Math.round((deadline.current! - Date.now()) / 1000),
      );
      setLeft(remaining);
      if (remaining <= 0 && !expired.current) {
        expired.current = true;
        expireRef.current();
      }
    };
    tick();
    const id = setInterval(tick, 500);
    return () => clearInterval(id);
  }, [active, totalSeconds]);

  return left;
}

function TravelClock({ secondsLeft }: { secondsLeft: number }) {
  const minutes = Math.floor(secondsLeft / 60);
  const seconds = String(secondsLeft % 60).padStart(2, "0");
  const warn = secondsLeft <= 10;
  return (
    <span
      className={`rounded px-2 py-0.5 font-mono text-sm font-medium tabular-nums ${
        warn
          ? "bg-red-500/90 text-white"
          : "bg-primary/90 text-primary-foreground"
      }`}
    >
      {minutes}:{seconds}
    </span>
  );
}

// ── error ────────────────────────────────────────────────────────────────────

function ErrorCard({ session }: { session: WorldSession }) {
  const { view } = session;
  if (view.kind !== "error") return null;
  return (
    <div className="flex flex-col gap-3 rounded-xl border border-red-400/20 bg-red-500/[0.06] p-4">
      <SectionLabel>
        {view.buildFailed ? "World build failed" : "Something broke"}
      </SectionLabel>
      <p className="break-words text-xs leading-relaxed text-red-300/90">
        {view.message}
      </p>
      <div className="flex flex-col gap-2">
        <Button onClick={session.retry}>Try again</Button>
        <Button variant="ghost" onClick={session.exit}>
          Back to worlds
        </Button>
      </div>
    </div>
  );
}
