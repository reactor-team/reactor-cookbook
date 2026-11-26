import Anthropic from "@anthropic-ai/sdk";
import { zodOutputFormat } from "@anthropic-ai/sdk/helpers/zod";
import OpenAI from "openai";
import { zodTextFormat } from "openai/helpers/zod";
import { NextResponse } from "next/server";
import { StorySchema, type Story } from "../story";
import { VerdictSchema, type Verdict } from "../check";
import { writerFor } from "./keys";

// The showrunner plans episodes on the larger model; the director writes every
// clip, so it runs on the small, fast one.
const MODELS = {
  claude: { story: "claude-sonnet-5-5", clip: "claude-haiku-5-5" },
  openai: { story: "gpt-5.1", clip: "gpt-5.1-mini" },
} as const;

export class NotConfigured extends Error {}
export class Refused extends Error {}

/** The director: one clip's prompt as plain text. */
export async function writeClip(req: Request, system: string, user: string): Promise<string> {
  const w = writerFor(req);
  if (!w) throw new NotConfigured();
  if (w.writer === "claude") {
    const response = await new Anthropic({ apiKey: w.key }).messages.create({
      model: MODELS.claude.clip,
      max_tokens: 16000,
      // The next clip should be written well before the current one ends.
      output_config: { effort: "low" },
      system,
      messages: [{ role: "user", content: user }],
    });
    if (response.stop_reason === "refusal") throw new Refused();
    return response.content.map((b) => (b.type === "text" ? b.text : "")).join("");
  }
  const response = await new OpenAI({ apiKey: w.key }).responses.create({
    model: MODELS.openai.clip,
    reasoning: { effort: "low" },
    instructions: system,
    input: user,
  });
  return response.output_text;
}

/** The showrunner: an episode, looking at the reference pictures (base64 JPEGs). */
export async function writeEpisode(req: Request, system: string, user: string, images: string[]): Promise<Story | null> {
  const w = writerFor(req);
  if (!w) throw new NotConfigured();
  if (w.writer === "claude") {
    const response = await new Anthropic({ apiKey: w.key }).messages.parse({
      model: MODELS.claude.story,
      max_tokens: 16000,
      // An episode should land in a few seconds so the stream doesn't wait.
      output_config: { effort: "low", format: zodOutputFormat(StorySchema) },
      system,
      messages: [
        {
          role: "user",
          content: [
            ...images.map((data) => ({ type: "image" as const, source: { type: "base64" as const, media_type: "image/jpeg" as const, data } })),
            { type: "text", text: user },
          ],
        },
      ],
    });
    if (response.stop_reason === "refusal") throw new Refused();
    return response.parsed_output ?? null;
  }
  const response = await new OpenAI({ apiKey: w.key }).responses.parse({
    model: MODELS.openai.story,
    reasoning: { effort: "low" },
    instructions: system,
    input: [
      {
        role: "user",
        content: [
          ...images.map((data) => ({ type: "input_image" as const, image_url: `data:image/jpeg;base64,${data}`, detail: "low" as const })),
          { type: "input_text" as const, text: user },
        ],
      },
    ],
    text: { format: zodTextFormat(StorySchema, "episode") },
  });
  return response.output_parsed ?? null;
}

export type Picture = { data: string; mediaType: "image/jpeg" | "image/png" | "image/webp" };

/** The shot check: a verdict on a built clip's last frame, on the fast model. */
export async function writeVerdict(req: Request, system: string, user: string, pictures: Picture[]): Promise<Verdict | null> {
  const w = writerFor(req);
  if (!w) throw new NotConfigured();
  if (w.writer === "claude") {
    const response = await new Anthropic({ apiKey: w.key }).messages.parse({
      model: MODELS.claude.clip,
      max_tokens: 4000,
      output_config: { effort: "low", format: zodOutputFormat(VerdictSchema) },
      system,
      messages: [
        {
          role: "user",
          content: [
            ...pictures.map((p) => ({ type: "image" as const, source: { type: "base64" as const, media_type: p.mediaType, data: p.data } })),
            { type: "text", text: user },
          ],
        },
      ],
    });
    return response.parsed_output ?? null;
  }
  const response = await new OpenAI({ apiKey: w.key }).responses.parse({
    model: MODELS.openai.clip,
    reasoning: { effort: "low" },
    instructions: system,
    input: [
      {
        role: "user",
        content: [
          ...pictures.map((p) => ({ type: "input_image" as const, image_url: `data:${p.mediaType};base64,${p.data}`, detail: "low" as const })),
          { type: "input_text" as const, text: user },
        ],
      },
    ],
    text: { format: zodTextFormat(VerdictSchema, "verdict") },
  });
  return response.output_parsed ?? null;
}

/** One error reply for both providers. `who` names the writer in the message. */
export function writerError(error: unknown, who: string) {
  const notSet = NextResponse.json(
    { error: `${who} not configured: paste a Claude or OpenAI key under Keys, or set one in .env.local` },
    { status: 503 },
  );
  if (error instanceof NotConfigured) return notSet;
  if (error instanceof Refused) return NextResponse.json({ error: `The ${who.toLowerCase()} declined.` }, { status: 422 });
  if (error instanceof Anthropic.AuthenticationError || error instanceof OpenAI.AuthenticationError) {
    return NextResponse.json({ error: `${who}: the key was rejected.` }, { status: 503 });
  }
  if (error instanceof Anthropic.RateLimitError || error instanceof OpenAI.RateLimitError) {
    return NextResponse.json({ error: `${who} rate limited.` }, { status: 429 });
  }
  if (error instanceof Anthropic.APIError || error instanceof OpenAI.APIError) {
    return NextResponse.json({ error: `${who} error ${error.status}: ${error.message}` }, { status: 502 });
  }
  return NextResponse.json({ error: `${who} failed.` }, { status: 500 });
}
