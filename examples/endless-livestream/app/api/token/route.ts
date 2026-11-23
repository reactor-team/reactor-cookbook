import { NextResponse } from "next/server";
import { MODEL_NAME } from "@reactor-models/pika-h3-flex/core";

// Trade the server-side API key for a short-lived JWT scoped to this model.
// The browser connects with the JWT; it never sees REACTOR_API_KEY.
export async function GET() {
  const apiKey = process.env.REACTOR_API_KEY;
  if (!apiKey) {
    return NextResponse.json({ error: "REACTOR_API_KEY not set in .env.local" }, { status: 500 });
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
    return NextResponse.json({ error: `Reactor returned ${res.status}` }, { status: 502 });
  }
  const { jwt } = (await res.json()) as { jwt: string };
  return NextResponse.json({ jwt }, { headers: { "Cache-Control": "no-store" } });
}
