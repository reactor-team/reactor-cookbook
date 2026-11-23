import Anthropic from "@anthropic-ai/sdk";
import { NextResponse } from "next/server";
import { z } from "zod";
import { BeatSchema } from "../../lib/story";
import { DIRECTOR_SYSTEM_PROMPT, RECENT_PROMPTS, directorMessage, withoutAlignmentLine } from "../../lib/director";

// Writes the next clip's prompt. Runs on the server so the Anthropic key never
// reaches the browser. Without a key the client sends the steer text as is.
export async function POST(req: Request) {
  const body = (await req.json().catch(() => null)) as { direction?: unknown; recent?: unknown; fresh?: unknown; story?: unknown; format?: unknown } | null;
  const formatParsed = ClipFormatSchema.safeParse(body?.format);
  const format = formatParsed.success ? formatParsed.data : ({ kind: "t2va" } as const);
  const fresh = body?.fresh === true;
  const parsed = StoryBeatContext.safeParse(body?.story);
  const story = parsed.success ? parsed.data : undefined;
  const direction = typeof body?.direction === "string" ? body.direction.slice(0, 2000) : "";
  const recent = Array.isArray(body?.recent)
    ? body.recent.filter((p): p is string => typeof p === "string").slice(-RECENT_PROMPTS).map((p) => p.slice(0, 2000))
    : [];

  try {
    // Built per request: the constructor throws when it finds no credentials.
    const client = new Anthropic();
    const response = await client.messages.create({
      // A short, frequent writing job: Haiku is fast and cheap enough to run every clip.
      model: "claude-haiku-5-5",
      max_tokens: 16000,
      // The next clip should be written well before the current one ends; low effort keeps it quick.
      output_config: { effort: "low" },
      system: DIRECTOR_SYSTEM_PROMPT,
      messages: [{ role: "user", content: directorMessage({ direction, recent, fresh, story, format }) }],
    });

    if (response.stop_reason === "refusal") {
      return NextResponse.json({ error: "The director declined this direction." }, { status: 422 });
    }
    const prompt = withoutAlignmentLine(
      response.content.map((block) => (block.type === "text" ? block.text : "")).join(""),
    );
    if (!prompt) return NextResponse.json({ error: "The director returned no prompt." }, { status: 502 });
    return NextResponse.json({ prompt });
  } catch (error) {
    if (error instanceof Anthropic.AuthenticationError) {
      return NextResponse.json({ error: "Director not configured: set ANTHROPIC_API_KEY in .env.local" }, { status: 503 });
    }
    if (error instanceof Anthropic.RateLimitError) {
      return NextResponse.json({ error: "Director rate limited." }, { status: 429 });
    }
    if (error instanceof Anthropic.APIError) {
      return NextResponse.json({ error: `Director error ${error.status}: ${error.message}` }, { status: 502 });
    }
    if (error instanceof Error && /api key|authToken|credentials/i.test(error.message)) {
      // The SDK throws before any request when it finds no credentials at all.
      return NextResponse.json({ error: "Director not configured: set ANTHROPIC_API_KEY in .env.local" }, { status: 503 });
    }
    return NextResponse.json({ error: "Director failed." }, { status: 500 });
  }
}

const StoryBeatContext = z.object({
  title: z.string().max(300),
  bible: z.string().max(4000),
  beat: BeatSchema,
  number: z.number().int().positive(),
  total: z.number().int().positive(),
  next: z.string().max(2000).optional(),
});

const ClipFormatSchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("t2va") }),
  z.object({ kind: z.literal("fl2va"), frames: z.enum(["first", "last", "both"]) }),
  z.object({
    kind: z.literal("ref2va"),
    pictures: z.number().int().min(0).max(9),
    audios: z.number().int().min(0).max(3),
    videos: z.number().int().min(0).max(3),
    tail: z.number().int().min(1).max(4).nullable(),
  }),
]);
