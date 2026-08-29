// The Reactor API the SDK talks to. The browser-side <X2Provider> and the
// server-side token mint (app/api/reactor/token) must both point at the same
// environment - a JWT minted against one environment is not valid on another,
// so they read this single value. Override with NEXT_PUBLIC_REACTOR_API_URL
// for local / staging; defaults to production.
export const REACTOR_API_URL =
  process.env.NEXT_PUBLIC_REACTOR_API_URL || "https://api.reactor.inc";
