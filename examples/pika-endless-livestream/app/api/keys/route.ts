import { NextResponse } from "next/server";

// Which keys the server already has, so the page can say a field is optional. Never the keys themselves.
export function GET() {
  return NextResponse.json(
    {
      reactor: Boolean(process.env.REACTOR_API_KEY),
      anthropic: Boolean(process.env.ANTHROPIC_API_KEY),
      openai: Boolean(process.env.OPENAI_API_KEY),
    },
    { headers: { "Cache-Control": "no-store" } },
  );
}
