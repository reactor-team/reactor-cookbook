import { NextResponse } from "next/server";

const MODEL_NAME = "reactor/lingbot-world-2";
const MAX_SESSIONS = 10;
const TOKEN_LIFETIME_SECONDS = 60 * 60;
const CACHE_SKEW_SECONDS = 60;

export async function GET() {
  const apiKey = process.env.REACTOR_API_KEY;
  if (!apiKey) {
    return NextResponse.json(
      { error: "Live session credentials are not set on the server." },
      { status: 500 },
    );
  }

  const baseUrl =
    process.env.NEXT_PUBLIC_COORDINATOR_URL ?? "https://api.reactor.inc";
  try {
    const response = await fetch(`${baseUrl}/tokens`, {
      method: "POST",
      headers: {
        "Reactor-API-Key": apiKey,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        expires_after: TOKEN_LIFETIME_SECONDS,
        authorization_details: [
          {
            type: "session",
            resources: { models: { match: [MODEL_NAME] } },
            constraints: { max_sessions: MAX_SESSIONS },
          },
        ],
      }),
    });

    const payload = (await response.json().catch(() => ({}))) as {
      jwt?: string;
      expires_at?: number;
      error?: string;
      message?: string;
    };
    if (!response.ok || !payload.jwt || !payload.expires_at) {
      return NextResponse.json(
        {
          error:
            payload.error ??
            payload.message ??
            `Session token request returned ${response.status}.`,
        },
        { status: 502 },
      );
    }

    const nowSeconds = Math.floor(Date.now() / 1000);
    const maxAge = Math.max(
      0,
      payload.expires_at - nowSeconds - CACHE_SKEW_SECONDS,
    );
    return NextResponse.json(
      { jwt: payload.jwt },
      { headers: { "Cache-Control": `private, max-age=${maxAge}` } },
    );
  } catch (error) {
    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "The live session service is unavailable.",
      },
      { status: 502 },
    );
  }
}
