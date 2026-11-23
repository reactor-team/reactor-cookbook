import { NextResponse } from "next/server";
import { MODEL_NAME } from "@reactor-models/pika-h3-flex/core";
import { KEY_HEADERS } from "../../lib/keys";
import { keyFor } from "../../lib/server/keys";

// Trade the server-side API key for a short-lived JWT scoped to this model.
// The browser connects with the JWT; it never sees REACTOR_API_KEY.
export async function GET(req: Request) {
  const apiKey = keyFor(req, "reactor");
  if (!apiKey) {
    return NextResponse.json({ error: "No Reactor key: paste one under Keys, or set REACTOR_API_KEY in .env.local" }, { status: 500 });
  }
  const res = await fetch("https://api.reactor.inc/tokens", {
    method: "POST",
    headers: { "Reactor-API-Key": apiKey, "Content-Type": "application/json" },
    body: JSON.stringify({
      expires_after: 60 * 60,
      authorization_details: [
        {
          type: "session",
          resources: { models: { match: [MODEL_NAME] } },
          constraints: { max_sessions: 10 },
        },
      ],
    }),
  });
  if (!res.ok) {
    const pasted = Boolean(req.headers.get(KEY_HEADERS.reactor)?.trim());
    const reason =
      res.status === 401 || res.status === 403
        ? pasted
          ? "Reactor rejected the pasted API key. Fix or clear it under API keys"
          : "Reactor rejected the server's API key"
        : "Reactor could not mint a session token";
    return NextResponse.json({ error: `${reason} (${res.status}).` }, { status: 502 });
  }
  const { jwt } = (await res.json()) as { jwt: string };
  return NextResponse.json({ jwt }, { headers: { "Cache-Control": "no-store" } });
}
