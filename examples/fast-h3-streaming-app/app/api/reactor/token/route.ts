import { NextResponse } from "next/server";
import { FAST_H3_MODEL } from "@/lib/h3-contract";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const TOKEN_URL = "https://api.reactor.inc/tokens";

function credentialFrom() {
  const credential = process.env.REACTOR_API_KEY?.trim();
  if (!credential) throw new Error("Missing REACTOR_API_KEY. Add it to .env.local and restart the server.");
  return credential;
}

function upstreamMessage(value: unknown) {
  if (!value || typeof value !== "object") return null;
  const record = value as Record<string, unknown>;
  if (typeof record.message === "string") return record.message;
  if (typeof record.error === "string") return record.error;
  if (record.error && typeof record.error === "object") {
    const nested = record.error as Record<string, unknown>;
    if (typeof nested.message === "string") return nested.message;
  }
  return null;
}

export async function POST() {
  try {
    const response = await fetch(TOKEN_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Reactor-API-Key": credentialFrom(),
      },
      body: JSON.stringify({
        expires_after: 3_600,
        authorization_details: [{
          type: "session",
          resources: { models: { match: [FAST_H3_MODEL] } },
          constraints: { max_sessions: 1, max_session_duration_seconds: 3_600 },
        }],
      }),
      cache: "no-store",
    });
    const body = await response.json().catch(() => ({})) as Record<string, unknown>;
    if (!response.ok || typeof body.jwt !== "string") {
      return NextResponse.json(
        { error: upstreamMessage(body) || `Reactor authentication returned ${response.status}.` },
        { status: response.status >= 400 && response.status <= 599 ? response.status : 502 },
      );
    }
    return NextResponse.json(
      { jwt: body.jwt, expiresAt: body.expires_at },
      { headers: { "Cache-Control": "no-store, max-age=0" } },
    );
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not create a Reactor session token." },
      { status: 500 },
    );
  }
}
