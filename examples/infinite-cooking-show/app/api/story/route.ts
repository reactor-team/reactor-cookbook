import { NextRequest, NextResponse } from "next/server";
import {
  CEREBRAS_MAX_IMAGE_PAYLOAD,
  CEREBRAS_MAX_IMAGES,
  CEREBRAS_STORY_MODEL,
  type StoryHistoryItem,
  type StoryPlanRequest,
} from "@/lib/cerebras-contract";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions";
const VISION_DATA_URL = /^data:image\/(?:jpeg|png);base64,[A-Za-z0-9+/=]+$/;

function credentialFrom() {
  const credential = process.env.CEREBRAS_API_KEY?.trim();
  if (!credential) throw new Error("Missing CEREBRAS_API_KEY.");
  return credential;
}

function historyFrom(value: unknown): StoryHistoryItem[] {
  if (!Array.isArray(value)) return [];
  return value.slice(-6).flatMap((item) => {
    if (!item || typeof item !== "object") return [];
    const record = item as Record<string, unknown>;
    return [{
      sceneSummary: typeof record.sceneSummary === "string" ? record.sceneSummary.slice(0, 1_200) : "",
      dialogue: typeof record.dialogue === "string" ? record.dialogue.slice(0, 500) : "",
      videoPrompt: typeof record.videoPrompt === "string" ? record.videoPrompt.slice(0, 2_000) : "",
    }];
  });
}

function imagesFrom(value: unknown) {
  if (!Array.isArray(value)) return [];
  const images = value.slice(0, CEREBRAS_MAX_IMAGES).filter((item): item is string => (
    typeof item === "string" && VISION_DATA_URL.test(item)
  ));
  if (images.reduce((sum, image) => sum + image.length, 0) > CEREBRAS_MAX_IMAGE_PAYLOAD) {
    throw new Error("Cerebras image context is too large.");
  }
  return images;
}

function propsFrom(value: unknown) {
  if (!Array.isArray(value)) return [];
  return value.slice(0, 4).flatMap((item) => (
    typeof item === "string" && item.trim() ? [item.trim().slice(0, 500)] : []
  ));
}

function upstreamMessage(value: unknown) {
  if (!value || typeof value !== "object") return null;
  const record = value as Record<string, unknown>;
  if (typeof record.message === "string") return record.message;
  if (record.error && typeof record.error === "object") {
    const nested = record.error as Record<string, unknown>;
    if (typeof nested.message === "string") return nested.message;
  }
  return null;
}

export function GET() {
  return NextResponse.json(
    { enabled: Boolean(process.env.CEREBRAS_API_KEY?.trim()) },
    { headers: { "Cache-Control": "no-store, max-age=0" } },
  );
}

export async function POST(request: NextRequest) {
  try {
    const credential = credentialFrom();
    const body = (await request.json()) as Partial<StoryPlanRequest>;
    const direction = typeof body.direction === "string" ? body.direction.trim().slice(0, 4_000) : "";
    if (!direction) return NextResponse.json({ error: "A story direction is required." }, { status: 400 });
    const requestedDuration = typeof body.duration === "number" ? body.duration : 10;
    const duration = Math.min(14, Math.max(6, Math.round(requestedDuration)));

    const history = historyFrom(body.history);
    const images = imagesFrom(body.images);
    const props = propsFrom(body.props);
    const visualNote = images.length
      ? "Use the supplied image only as visual planning context."
      : "No reference image is available; rely on the written continuity history and textual scene cues.";
    const userText = [
      `Operator direction: ${direction}`,
      visualNote,
      props.length ? `Persistent prop requirements: ${props.join(" | ")}` : "No persistent prop is currently required.",
      `Recent accepted story history, oldest to newest: ${JSON.stringify(history)}`,
      `Plan only the next ${duration} seconds.`,
    ].join("\n\n");

    const content: Array<Record<string, unknown>> = [{ type: "text", text: userText }];
    for (const image of images) {
      content.push({ type: "image_url", image_url: { url: image } });
    }

    const startedAt = Date.now();
    const response = await fetch(CEREBRAS_URL, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${credential}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: CEREBRAS_STORY_MODEL,
        messages: [
          {
            role: "system",
            content: [
              "You are the live showrunner for an infinite audiovisual stream built from continuous short video clips.",
              "Continue the story causally from the newest history and visual frame while preserving character identity, location, lighting, screen direction, props, and camera language.",
              "Treat the operator direction as the requested next beat. Treat every named persistent prop as mandatory scene state and visibly incorporate it into the established scene in the most intuitive way.",
              "All newly added objects and images arrive as textual scene cues. Incorporate each named cue intuitively without cutting away from the established scene. Keep every named persistent prop visible or logically present across frames; subjects may receive, notice, pick up, prepare, move, or cook with them, but the props must not vanish.",
              `Write a concrete FastH3 generation prompt describing visible action over ${duration} seconds, camera behavior, ambience, sound effects, and a spoken line when dialogue helps the story.`,
              `Keep dialogue speakable within ${duration} seconds and quote the exact words inside videoPrompt. Do not mention prompts, source images, chunks, models, or continuity systems.`,
            ].join(" "),
          },
          { role: "user", content },
        ],
        temperature: 0.8,
        max_completion_tokens: 650,
        response_format: {
          type: "json_schema",
          json_schema: {
            name: "next_story_beat",
            strict: true,
            schema: {
              type: "object",
              properties: {
                videoPrompt: { type: "string" },
                sceneSummary: { type: "string" },
                dialogue: { type: "string" },
              },
              required: ["videoPrompt", "sceneSummary", "dialogue"],
              additionalProperties: false,
            },
          },
        },
      }),
      cache: "no-store",
      signal: AbortSignal.timeout(10_000),
    });
    const data = await response.json().catch(() => ({})) as Record<string, unknown>;
    if (!response.ok) {
      return NextResponse.json(
        { error: upstreamMessage(data) || `Cerebras returned ${response.status}.` },
        { status: response.status >= 400 && response.status <= 599 ? response.status : 502 },
      );
    }

    const choices = Array.isArray(data.choices) ? data.choices : [];
    const first = choices[0] as { message?: { content?: unknown } } | undefined;
    const rawContent = first?.message?.content;
    if (typeof rawContent !== "string") throw new Error("Cerebras returned no story plan.");
    const plan = JSON.parse(rawContent) as Record<string, unknown>;
    const videoPrompt = typeof plan.videoPrompt === "string" ? plan.videoPrompt.trim().slice(0, 8_000) : "";
    if (!videoPrompt) throw new Error("Cerebras returned an empty video prompt.");

    const timeInfo = data.time_info as Record<string, unknown> | undefined;
    return NextResponse.json({
      videoPrompt,
      sceneSummary: typeof plan.sceneSummary === "string" ? plan.sceneSummary.trim().slice(0, 2_000) : "",
      dialogue: typeof plan.dialogue === "string" ? plan.dialogue.trim().slice(0, 500) : "",
      model: CEREBRAS_STORY_MODEL,
      latencyMs: typeof timeInfo?.total_time === "number"
        ? Math.round(timeInfo.total_time * 1_000)
        : Date.now() - startedAt,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Could not plan the next story beat.";
    const status = message.startsWith("Missing CEREBRAS_API_KEY") ? 503 : 502;
    return NextResponse.json({ error: message }, { status });
  }
}
