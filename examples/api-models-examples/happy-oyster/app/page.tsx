import { HappyOysterApp } from "./HappyOysterApp";
import { SetupRequired } from "./SetupRequired";

// Server Component gate. Ways to render:
//   - NEXT_PUBLIC_HO_LOCAL_RUNTIME=1 → the live app against a local runtime (no key)
//   - REACTOR_API_KEY set            → the live app, which mints JWTs server-side
//   - none                           → the <SetupRequired /> landing
//
// `force-dynamic` skips static prerendering so the env check runs per-request.
export const dynamic = "force-dynamic";

export default function Page() {
  if (process.env.NEXT_PUBLIC_HO_LOCAL_RUNTIME === "1")
    return <HappyOysterApp />;
  const hasKey = !!process.env.REACTOR_API_KEY;
  return hasKey ? <HappyOysterApp /> : <SetupRequired />;
}
