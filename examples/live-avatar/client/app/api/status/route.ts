import { NextResponse } from "next/server";

import { describeBackend } from "@/lib/reactor";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/** Which avatar backend and LiveKit server the page will use (no secrets). */
export function GET() {
  let livekit: string | null = null;
  try {
    livekit = new URL(process.env.LIVEKIT_URL ?? "").host;
  } catch {}
  return NextResponse.json({ avatar: describeBackend(), livekit, agent: process.env.AVATAR_AGENT_NAME ?? null });
}
