import { Header } from "@/components/Header";

const ACCOUNT_API_KEYS_URL = "https://www.reactor.inc/account/api-keys";

// Server Component shown when REACTOR_API_KEY is missing.
// Pure markup, no hooks, no client components, so it stays server-rendered.
export function SetupRequired() {
  return (
    <div className="flex min-h-screen flex-col bg-zinc-950">
      <Header />
      <main className="flex flex-1 items-center justify-center p-6">
        <div className="w-full max-w-md rounded-xl border border-white/[0.08] bg-white/[0.02] p-6">
          <h2 className="text-base font-semibold text-zinc-100">
            Setup required
          </h2>
          <p className="mt-1 text-sm text-zinc-400">
            This app needs a Reactor API key to mint session tokens. You only
            need to do this once.
          </p>

          <ol className="mt-5 space-y-4 text-sm text-zinc-300">
            <li className="flex gap-3">
              <Step>1</Step>
              <span>
                Create an API key at{" "}
                <a
                  href={ACCOUNT_API_KEYS_URL}
                  target="_blank"
                  rel="noreferrer"
                  className="text-primary underline-offset-2 hover:underline"
                >
                  reactor.inc/account/api-keys
                </a>
                . It starts with <code className="text-zinc-200">rk_</code>.
              </span>
            </li>
            <li className="flex gap-3">
              <Step>2</Step>
              <div className="min-w-0 flex-1">
                <p>
                  Save it to <code className="text-zinc-200">.env.local</code>{" "}
                  in the project root:
                </p>
                <pre className="mt-2 overflow-x-auto rounded-md border border-zinc-800 bg-zinc-900 p-2 font-mono text-xs text-zinc-300">
                  REACTOR_API_KEY=rk_your_key_here
                </pre>
              </div>
            </li>
            <li className="flex gap-3">
              <Step>3</Step>
              <span>
                Restart the dev server (
                <code className="text-zinc-200">pnpm dev</code>) so the new
                variable is picked up.
              </span>
            </li>
          </ol>
        </div>
      </main>
    </div>
  );
}

function Step({ children }: { children: React.ReactNode }) {
  return (
    <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-zinc-700 text-[11px] text-zinc-400">
      {children}
    </span>
  );
}
