import { NextResponse } from "next/server";
import { writeEpisode, writerError } from "../../lib/server/writer";
import { STORY_SYSTEM_PROMPT, StorySchema, storyMessage, type StoryRequest } from "../../lib/story";

// Plans an episode (or the next one, or a rewrite after a steer). Runs on the
// server; keys from .env.local never reach the browser.
export async function POST(req: Request) {
  const body = (await req.json().catch(() => null)) as Record<string, unknown> | null;
  const request = parseRequest(body);
  if (!request) return NextResponse.json({ error: "Bad story request." }, { status: 400 });

  // The reference pictures, shown to the showrunner so the bible can describe them.
  const images = parseImages(body?.images);

  try {
    const story = await writeEpisode(req, STORY_SYSTEM_PROMPT, storyMessage(request), images);
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
    return writerError(error, "Story writer");
  }
}

function parseImages(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.filter((v): v is string => typeof v === "string" && /^[A-Za-z0-9+/=]+$/.test(v) && v.length < 1_500_000).slice(0, 9);
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
