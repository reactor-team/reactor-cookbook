import { NextResponse } from "next/server";

// How long we ask Reactor to make the JWT valid for. The server caps
// this at its configured maximum (currently 6h), so asking for more
// is harmless — you just get the server max back.
const TOKEN_LIFETIME_SECONDS = 6 * 60 * 60;

// Safety margin on the cache lifetime so an in-flight request doesn't
// race with the real expiry.
const CACHE_SKEW_SECONDS = 60;

// Mint a Reactor JWT and return it with a `Cache-Control` header
// that lets the browser reuse it for the rest of its lifetime.
//
// Why GET and not POST?
//   POST responses are not cached by browsers. We expose this route
//   as GET so the browser's HTTP cache can serve repeat calls
//   transparently — no localStorage, no JWT parsing in client code.
//   The route handler still POSTs to Reactor internally.
//
// Why `private`?
//   Tells shared caches (CDNs, corporate proxies) not to store the
//   response. JWTs are per-user and must never be reused across users.
//
// Why derive `max-age` from `expires_at`?
//   Reactor decides the actual token lifetime (it caps the request
//   at its server max). Reading `expires_at` off the response means
//   the cache window is always in sync with whatever the server
//   actually granted, with a one-minute safety skew baked in.
export async function GET() {
  const apiKey = process.env.REACTOR_API_KEY;
  if (!apiKey) {
    return NextResponse.json(
      { error: "REACTOR_API_KEY is not set on the server" },
      { status: 500 },
    );
  }

  const res = await fetch("https://api.reactor.inc/tokens", {
    method: "POST",
    headers: {
      "Reactor-API-Key": apiKey,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ expires_after: TOKEN_LIFETIME_SECONDS }),
  });

  if (!res.ok) {
    return NextResponse.json(
      { error: `Reactor /tokens returned ${res.status}` },
      { status: 502 },
    );
  }

  const { jwt, expires_at } = (await res.json()) as {
    jwt: string;
    expires_at: number;
  };

  const nowSeconds = Math.floor(Date.now() / 1000);
  const maxAge = Math.max(0, expires_at - nowSeconds - CACHE_SKEW_SECONDS);

  return NextResponse.json(
    { jwt },
    {
      headers: {
        "Cache-Control": `private, max-age=${maxAge}`,
      },
    },
  );
}
