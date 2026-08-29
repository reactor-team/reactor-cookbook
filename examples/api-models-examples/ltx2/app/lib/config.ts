// The Reactor API the SDK talks to. The browser-side <Ltx2Provider> and the
// server-side token mint (app/api/reactor/token) must both point at the same
// environment — a JWT minted against one environment is not valid on another,
// so they read this single value.
//
// ltx2 runs on production. It is not listed publicly yet, so the key's account
// still needs to be granted access, but the API endpoint is the normal one.
// Override with NEXT_PUBLIC_REACTOR_API_URL.
export const REACTOR_API_URL =
  process.env.NEXT_PUBLIC_REACTOR_API_URL || "https://api.reactor.inc";
