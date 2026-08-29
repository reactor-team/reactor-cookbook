import { cn, EYEBROW } from "./ui";

export function Header() {
  return (
    <header className="flex items-center justify-between border-b border-zinc-800 bg-zinc-900/40 px-4 py-3 lg:px-6">
      <div className="flex items-baseline gap-3">
        <h1 className="text-sm font-semibold tracking-tight text-zinc-100">
          SANA-Streaming
        </h1>
        <span
          className={cn(
            EYEBROW,
            "hidden border-l border-zinc-800 pl-3 sm:inline",
          )}
        >
          Real-time video-to-video
        </span>
      </div>
      <span className={EYEBROW}>Powered by Reactor</span>
    </header>
  );
}
