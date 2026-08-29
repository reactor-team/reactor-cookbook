import { Header } from "./components/Header";

const ACCOUNT_API_KEYS_URL = "https://www.reactor.inc/account/api-keys";

// Server Component shown when REACTOR_API_KEY is missing from the
// environment. Pure markup — no hooks, no UI-lib components, so it
// can stay server-rendered. Brand alignment comes from the font and
// `text-brand` accent color exposed via Tailwind's @theme in
// globals.css.
export function SetupRequired() {
  return (
    <div className="flex min-h-screen flex-col">
      <Header />
      <main className="flex flex-1 items-center justify-center p-6">
        <div className="w-full max-w-md rounded-lg border border-zinc-800 bg-zinc-900/40 p-6">
          <h2 className="text-base font-semibold text-zinc-100">
            Setup required
          </h2>
          <p className="mt-1 text-sm text-zinc-400">
            This app needs a Reactor API key to mint session tokens. You only
            need to do this once.
          </p>

          <ol className="mt-5 space-y-4 text-sm text-zinc-300">
            <li className="flex gap-3">
              <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-zinc-700 text-[11px] text-zinc-400">
                1
              </span>
              <span>
                Create an API key at{" "}
                <a
                  href={ACCOUNT_API_KEYS_URL}
                  target="_blank"
                  rel="noreferrer"
                  className="text-brand underline-offset-2 hover:underline"
                >
                  reactor.inc/dashboard
                </a>
                . It starts with <code className="text-zinc-200">rk_</code>.
              </span>
            </li>
            <li className="flex gap-3">
              <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-zinc-700 text-[11px] text-zinc-400">
                2
              </span>
              <div className="min-w-0 flex-1">
                <p>
                  Save it to <code className="text-zinc-200">.env</code> in the
                  project root:
                </p>
                <pre className="mt-2 overflow-x-auto rounded-md border border-zinc-800 bg-zinc-950 p-2 font-mono text-xs text-zinc-300">
                  REACTOR_API_KEY=rk_your_key_here
                </pre>
              </div>
            </li>
            <li className="flex gap-3">
              <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-zinc-700 text-[11px] text-zinc-400">
                3
              </span>
              <span>
                Restart the dev server (
                <code className="text-zinc-200">pnpm dev</code>) so the new
                variable is picked up.
              </span>
            </li>
          </ol>

          <p className="mt-6 border-t border-zinc-800 pt-4 text-[11px] text-zinc-500">
            The key is read only on the server — the browser only ever sees
            short-lived JWTs minted from it.
          </p>
        </div>
      </main>
    </div>
  );
}
