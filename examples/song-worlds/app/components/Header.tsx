export function Header() {
  return (
    <header className="flex items-center justify-between border-b border-zinc-800 bg-zinc-900/40 px-4 py-3 lg:px-6">
      <div className="flex items-baseline gap-3">
        <h1 className="text-sm font-semibold tracking-tight text-zinc-100">
          Helios Interactive
        </h1>
        <span className="hidden border-l border-zinc-800 pl-3 text-[11px] uppercase tracking-wider text-zinc-500 sm:inline">
          Real-time interactive video generation
        </span>
      </div>
      <span className="text-[11px] uppercase tracking-wider text-zinc-500">
        Powered by Reactor
      </span>
    </header>
  );
}
