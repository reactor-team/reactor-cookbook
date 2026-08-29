"use client";

// Directing (mode 2) control deck: steer the story with free-text instructions
// plus pause / resume / rewind transport, over the live instruction timeline
// the runtime reconciles into travel_state. Rewind needs the session paused and
// snaps to multiples of 4 seconds (the server rounds down).

import { useState } from "react";
import { useHappyOysterClient } from "./ho-client";
import { SectionLabel } from "./ui";

export function DirectingControls() {
  const { instruct, pause, resume, rewind, travelState, travelStatus } =
    useHappyOysterClient();
  const [text, setText] = useState("");
  const [rewindSec, setRewindSec] = useState(4);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const paused = travelStatus === "paused";

  const run = (action: () => Promise<unknown>) => {
    setError(null);
    setBusy(true);
    void action()
      .catch((cause) => setError(String(cause)))
      .finally(() => setBusy(false));
  };

  const send = () => {
    if (!text.trim()) return;
    run(() => instruct(text.trim()).then(() => setText("")));
  };

  const instructions = travelState?.user_instructions ?? [];
  const chapters = travelState?.chapters ?? [];

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-3 rounded-xl border border-white/10 bg-white/[0.03] p-4">
        <div className="flex items-center justify-between">
          <SectionLabel>Direct the story</SectionLabel>
          <span className="font-mono text-[10px] uppercase tracking-tight text-white/35">
            {travelStatus}
          </span>
        </div>
        <div className="flex gap-1.5">
          <input
            value={text}
            onChange={(event) => setText(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") send();
            }}
            placeholder="Steer the next scene… “A storm rolls in”"
            className="min-w-0 flex-1 rounded-md border border-white/10 bg-black/30 px-3 py-2 font-mono text-sm text-white/85 outline-none transition placeholder:text-white/25 focus:border-white/30 focus:ring-2 focus:ring-primary/20"
          />
          <button
            disabled={busy || text.trim().length === 0}
            onClick={send}
            className="shrink-0 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition hover:brightness-95 disabled:opacity-40"
          >
            Instruct
          </button>
        </div>
        <div className="grid grid-cols-2 gap-1.5">
          {paused ? (
            <TransportButton disabled={busy} onClick={() => run(resume)}>
              ▶ Resume
            </TransportButton>
          ) : (
            <TransportButton disabled={busy} onClick={() => run(pause)}>
              ⏸ Pause
            </TransportButton>
          )}
          <div className="flex gap-1.5">
            <input
              type="number"
              min={0}
              step={4}
              value={rewindSec}
              onChange={(event) =>
                setRewindSec(Math.max(0, Number(event.target.value)))
              }
              className="w-16 rounded-md border border-white/10 bg-black/30 px-2 py-1.5 font-mono text-sm text-white/85 outline-none focus:border-white/30"
            />
            <TransportButton
              disabled={busy || !paused}
              title={paused ? "Rewind to this second" : "Pause first to rewind"}
              onClick={() => run(() => rewind(rewindSec))}
            >
              ⏪ Rewind
            </TransportButton>
          </div>
        </div>
        <p className="text-[11px] leading-relaxed text-white/30">
          Instructions steer the next chunk. Rewind takes multiples of 4s and
          needs the session paused first.
        </p>
        {error && <p className="text-xs text-red-400">{error}</p>}
      </div>

      <div className="flex flex-col gap-3 rounded-xl border border-white/10 bg-white/[0.03] p-4">
        <SectionLabel>Story timeline</SectionLabel>
        {instructions.length === 0 && chapters.length === 0 ? (
          <p className="text-sm text-white/30">
            Your instructions appear here with the window HappyOyster schedules
            them into on the video timeline.
          </p>
        ) : (
          <div className="flex max-h-40 flex-col gap-1.5 overflow-y-auto pr-1">
            {instructions.map((instruction, index) => (
              <div
                key={`${instruction.instruction}-${index}`}
                className="flex items-baseline justify-between gap-3 rounded-md border border-white/[0.06] bg-black/20 px-3 py-1.5"
              >
                <span className="min-w-0 truncate text-sm text-white/75">
                  {instruction.instruction}
                </span>
                <span className="shrink-0 font-mono text-[10px] tabular-nums text-white/35">
                  {instruction.start_time != null &&
                  instruction.end_time != null
                    ? `${instruction.start_time}s–${instruction.end_time}s`
                    : (instruction.status ?? "scheduled")}
                </span>
              </div>
            ))}
            {chapters.map((chapter, index) => (
              <div
                key={`chapter-${chapter.chapter_id ?? index}`}
                className="flex items-baseline justify-between gap-3 rounded-md border border-primary/20 bg-primary/[0.06] px-3 py-1.5"
              >
                <span className="min-w-0 truncate text-sm text-primary/90">
                  {chapter.title ??
                    `Chapter ${chapter.chapter_id ?? index + 1}`}
                </span>
                <span className="shrink-0 font-mono text-[10px] tabular-nums text-white/35">
                  {chapter.start_time != null && chapter.end_time != null
                    ? `${chapter.start_time}s–${chapter.end_time}s`
                    : ""}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function TransportButton({
  children,
  onClick,
  disabled,
  title,
}: {
  children: React.ReactNode;
  onClick: () => void;
  disabled?: boolean;
  title?: string;
}) {
  return (
    <button
      disabled={disabled}
      title={title}
      onClick={onClick}
      className="flex-1 rounded-md border border-white/15 bg-white/[0.06] px-3 py-1.5 text-sm text-white/70 transition hover:bg-white/[0.12] hover:text-white/90 disabled:opacity-40"
    >
      {children}
    </button>
  );
}
