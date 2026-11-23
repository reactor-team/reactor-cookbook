import Anthropic from "@anthropic-ai/sdk";
import { zodOutputFormat } from "@anthropic-ai/sdk/helpers/zod";
import { NextResponse } from "next/server";
import { STORY_SYSTEM_PROMPT, StorySchema, storyMessage, type StoryRequest } from "../../lib/story";

// Plans an episode (or the next one, or a rewrite after a steer). Runs on the
// server so the Anthropic key never reaches the browser.
export async function POST(req: Request) {
  const body = (await req.json().catch(() => null)) as Record<string, unknown> | null;
  const request = parseRequest(body);
  if (!request) return NextResponse.json({ error: "Bad story request." }, { status: 400 });

  // The reference pictures, shown to the showrunner so the bible can describe them.
  const images: Anthropic.ImageBlockParam[] = parseImages(body?.images);

  try {
    // Built per request: the constructor throws when it finds no credentials.
    const client = new Anthropic();
    const response = await client.messages.parse({
      model: "claude-sonnet-5-5",
      max_tokens: 16000,
      // An episode should land in a few seconds; low effort keeps the stream from waiting.
      output_config: { effort: "low", format: zodOutputFormat(StorySchema) },
      system: STORY_SYSTEM_PROMPT,
      messages: [{ role: "user", content: [...images, { type: "text", text: storyMessage(request) }] }],
    });
    if (response.stop_reason === "refusal") {
      return NextResponse.json({ error: "The showrunner declined this concept." }, { status: 422 });
    }
    const story = response.parsed_output;
    if (!story || !story.beats.length) {
      return NextResponse.json({ error: "The showrunner returned no episode." }, { status: 502 });
    }
    // A steer keeps the beats already written; trust ours, not the model's copy.
    if (request.mode === "steer") {
      const kept = request.story.beats.slice(0, request.done);
      const rest = story.beats.slice(kept.length);
      story.beats = [...kept, ...rest];
    }
    return NextResponse.json({ story });
  } catch (error) {
    if (error instanceof Anthropic.AuthenticationError) {
      return NextResponse.json({ error: "Story writer not configured: set ANTHROPIC_API_KEY in .env.local" }, { status: 503 });
    }
    if (error instanceof Anthropic.RateLimitError) {
      return NextResponse.json({ error: "Story writer rate limited." }, { status: 429 });
    }
    if (error instanceof Anthropic.APIError) {
      return NextResponse.json({ error: `Story writer error ${error.status}: ${error.message}` }, { status: 502 });
    }
    return NextResponse.json({ error: "Story writer failed." }, { status: 500 });
  }
}

function parseImages(value: unknown): Anthropic.ImageBlockParam[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter((v): v is string => typeof v === "string" && /^[A-Za-z0-9+/=]+$/.test(v) && v.length < 1_500_000)
    .slice(0, 9)
    .map((data) => ({ type: "image", source: { type: "base64", media_type: "image/jpeg", data } }));
}

function parseRequest(body: Record<string, unknown> | null): StoryRequest | null {
  if (!body) return null;
  const references = typeof body.references === "string" ? body.references.slice(0, 4000) : undefined;
  const story = StorySchema.safeParse(body.story);
  if (body.mode === "new" && typeof body.premise === "string" && body.premise.trim()) {
    return { mode: "new", premise: body.premise.slice(0, 2000), references };
  }
  if (body.mode === "next" && story.success) return { mode: "next", story: story.data, references };
  if (body.mode === "steer" && story.success && typeof body.direction === "string" && typeof body.done === "number") {
    return { mode: "steer", story: story.data, done: Math.max(0, Math.min(body.done, story.data.beats.length)), direction: body.direction.slice(0, 2000), references };
  }
  return null;
}
