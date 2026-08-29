// Server Component: no hooks, no "use client". Rendered by app/page.tsx when
// REACTOR_API_KEY is missing, so a fresh clone explains itself instead of
// failing at connect time with an opaque 401.
export function SetupRequired() {
  return (
    <main className="flex min-h-screen items-center justify-center p-6">
      <div className="w-full max-w-md rounded-xl border border-edge bg-surface p-8">
        <p className="font-mono text-[11px] uppercase tracking-wider text-brand">
          Setup required
        </p>
        <h1 className="mt-2 text-xl font-medium">REACTOR_API_KEY is not set</h1>
        <p className="mt-3 text-sm leading-relaxed text-zinc-400">
          This app mints session tokens server-side, so it needs a Reactor API
          key in the environment. Copy{" "}
          <code className="font-mono text-brand-light">.env.example</code> to{" "}
          <code className="font-mono text-brand-light">.env.local</code>, set{" "}
          <code className="font-mono text-brand-light">REACTOR_API_KEY</code>,
          then restart the dev server.
        </p>
        <p className="mt-4 text-sm leading-relaxed text-zinc-500">
          ltx2 is not listed publicly yet, so a key reaches it only once its
          account has been granted access. Ask the Reactor team if{" "}
          <code className="font-mono text-brand-light">connect</code> fails with
          a 404.
        </p>
      </div>
    </main>
  );
}
