import { NextResponse } from "next/server";

// How long we ask Reactor to make the JWT valid for. The server caps
// this at its configured maximum (currently 6h), so asking for more
// is harmless, you just get the server max back.
const TOKEN_LIFETIME_SECONDS = 6 * 60 * 60;

// Mint a Reactor JWT and return it together with its `expires_at`, so the
// client can memoize it for exactly its lifetime.
//
// Why `no-store`?
//   The client owns the cache (see fetchToken in
//   components/happy-oyster/ho-client.tsx). Keeping the token in the app
//   rather than in the browser's HTTP cache makes its lifetime observable,
//   and keeps a cache miss from silently minting a second token part-way
//   through a session.
//
// Why GET and not POST?
//   Nothing about the request varies, and a GET reads as the lookup it is.
//   The route handler still POSTs to Reactor internally.
export async function GET() {
  const apiKey = process.env.REACTOR_API_KEY;
  if (!apiKey) {
    return NextResponse.json(
      { error: "REACTOR_API_KEY is not set on the server" },
      { status: 500 },
    );
  }

  const baseUrl =
    process.env.NEXT_PUBLIC_REACTOR_API_URL || "https://api.reactor.inc";

  const res = await fetch(`${baseUrl}/tokens`, {
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

  // `expires_at` (unix seconds, decided by the server) lets the client
  // memoize the token for exactly its real lifetime.
  return NextResponse.json(
    { jwt, expires_at },
    {
      headers: {
        "Cache-Control": "private, no-store",
      },
    },
  );
}
