import { NextResponse } from "next/server";
import { z } from "zod";
import { BeatSchema, CastSchema, PlaceSchema, PropSchema, fillLabels } from "../../lib/story";
import { writeClip, writerError } from "../../lib/server/writer";
import { DIRECTOR_SYSTEM_PROMPT, RECENT_PROMPTS, directorMessage, withoutAlignmentLine } from "../../lib/director";

// Writes the next clip's prompt. Runs on the server; keys from .env.local never
// reach the browser. Without a key the client sends the steer text as is.
export async function POST(req: Request) {
  const body = (await req.json().catch(() => null)) as { direction?: unknown; recent?: unknown; fresh?: unknown; story?: unknown; format?: unknown; independent?: unknown; retake?: unknown } | null;
  const independent = body?.independent === true;
  const retakeParsed = RetakeSchema.safeParse(body?.retake);
  const retake = retakeParsed.success ? retakeParsed.data : undefined;
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
    const prompt = withoutAlignmentLine(
      await writeClip(req, DIRECTOR_SYSTEM_PROMPT, directorMessage({ direction, recent, fresh, story, format, independent, retake })),
    );
    // Each @label becomes the cast member's fixed look, word for word.
    const filled = story ? fillLabels(prompt, story) : prompt;
    if (!filled) return NextResponse.json({ error: "The director returned no prompt." }, { status: 502 });
    return NextResponse.json({ prompt: filled });
  } catch (error) {
    return writerError(error, "Director");
  }
}

const StoryBeatContext = z.object({
  title: z.string().max(300),
  bible: z.string().max(4000),
  beat: BeatSchema,
  number: z.number().int().positive(),
  total: z.number().int().positive(),
  next: z.string().max(2000).optional(),
  cast: z.array(CastSchema).max(12),
  props: z.array(PropSchema).max(4),
  place: PlaceSchema.optional(),
});

const RetakeSchema = z.object({
  prompt: z.string().max(6000),
  missed: z.string().max(40),
  reason: z.string().max(1000),
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
