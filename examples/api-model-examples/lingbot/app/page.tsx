import { LingbotApp } from "./LingbotApp";
import { SetupRequired } from "./SetupRequired";

// The page is a Server Component. Its only job is to check whether
// the app is configured (REACTOR_API_KEY is present) and render the
// right tree:
//   - missing key  → friendly <SetupRequired /> landing
//   - present      → <LingbotApp />, which fetches the JWT itself
//
// We don't mint the token here. Token minting lives behind
// /api/reactor/token so the same client-side flow works whether the
// frontend is Next.js, Vite, CRA, or anything else — the route is
// the framework-agnostic contract.
//
// `dynamic = "force-dynamic"` skips static prerendering so the env
// check runs per-request.
export const dynamic = "force-dynamic";

export default function Page() {
  const hasKey = !!process.env.REACTOR_API_KEY;
  return hasKey ? <LingbotApp /> : <SetupRequired />;
}
