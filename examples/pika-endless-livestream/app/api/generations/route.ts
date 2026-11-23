import { NextResponse } from "next/server";
import { MODEL_NAME } from "@reactor-models/pika-h3-flex/core";
import { reactor } from "../../lib/server/reactor";

// Starts one clip on the generation API. Only Pika H3 Flex, whatever the page sends.
export async function POST(req: Request) {
  const body = (await req.json().catch(() => null)) as { input?: unknown } | null;
  if (!body?.input || typeof body.input !== "object") {
    return NextResponse.json({ error: "Bad generation request." }, { status: 400 });
  }
  return reactor(req, "/generations", { model: MODEL_NAME, input: body.input });
}
