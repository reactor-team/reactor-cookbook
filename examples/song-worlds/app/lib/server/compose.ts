import "server-only";
import OpenAI from "openai";
import {
  NEMOTRON_BASE_URL,
  NEMOTRON_MAX_TOKENS,
  NEMOTRON_MODEL,
  NEMOTRON_REASONING_BUDGET,
  NEMOTRON_TEMPERATURE,
  NEMOTRON_TOP_P,
  getNemotronApiKey,
} from "./config";
import { ACTIVE_TEMPLATE } from "./promptTemplate";
import { resolveSeedId, type SeedEntry } from "./seedCatalog";
import type { MusicalSummary, TokenUsage, WorldEvent } from "../world/types";

/**
 * Stage 2 — the one Nemotron composition call.
 *
 * Server-side ONLY: NVIDIA_NEMO_KEY must never reach the browser, so
 * this module is imported exclusively from API routes. The frontend
 * only ever sees the finished event series.
 *
 * Nemotron specifics handled here:
 *  - OpenAI-compatible API at integrate.api.nvidia.com (OpenAI SDK with
 *    a custom baseURL) — different provider, same wire shape as Groq was.
 *  - Nemotron is a REASONING model: it emits a reasoning trace before
 *    the final answer. Depending on server config the trace arrives in
 *    `message.reasoning_content` (separate field) or inline inside
 *    `message.content` as a <think>…</think> block. We take content,
 *    strip any think block, then extract the JSON object defensively —
 *    the model may wrap or precede it with prose.
 *  - Sampling per NVIDIA guidance for reasoning models (temp ~1.0 /
 *    top_p ~0.95), exposed as config in server/config.ts.
 */

export class ComposeError extends Error {
  constructor(
    message: string,
    readonly status: number = 502,
  ) {
    super(message);
  }
}

export interface ComposeOutput {
  events: WorldEvent[];
  interpretation: string;
  tokenUsage: TokenUsage | null;
  model: string;
}

// ── reasoning-trace-safe JSON extraction ──

/** Strip <think>…</think> (or an unterminated <think>… prefix) that some
 *  deployments inline into content instead of reasoning_content. */
export function stripReasoning(content: string): string {
  let out = content.replace(/<think>[\s\S]*?<\/think>/gi, "");
  // Unclosed think block: everything from <think> onward is reasoning
  // unless a JSON object follows it — keep from the last '{' if so.
  const openIdx = out.search(/<think>/i);
  if (openIdx !== -1) out = out.slice(0, openIdx);
  return out.trim();
}

/** Find the first balanced top-level JSON object in the text. Never
 *  assume the whole completion is clean JSON. */
export function extractJsonObject(text: string): unknown {
  const start = text.indexOf("{");
  if (start === -1) throw new ComposeError("Model response contains no JSON object");
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === "\\") escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === "{") depth++;
    else if (ch === "}") {
      depth--;
      if (depth === 0) {
        const candidate = text.slice(start, i + 1);
        try {
          return JSON.parse(candidate);
        } catch {
          // Balanced but unparseable — try from the next '{'.
          const rest = text.slice(start + 1);
          return extractJsonObject(rest);
        }
      }
    }
  }
  throw new ComposeError("Model response has an unterminated JSON object");
}

// ── event validation ──

/** Anchor timestamps the LLM was allowed to place events on. */
export function collectAnchors(summary: MusicalSummary): number[] {
  const anchors = new Set<number>();
  for (const s of summary.sections) {
    anchors.add(s.start);
    anchors.add(s.end);
  }
  for (const m of summary.notableMoments) anchors.add(m.time);
  return [...anchors].sort((a, b) => a - b);
}

const ANCHOR_SNAP_TOLERANCE_SEC = 1.0;
const MIN_EVENT_GAP_SEC = 2.0;

/**
 * Single trust boundary for the event series: whatever the LLM said,
 * the client receives events that are sorted, snapped to real anchors
 * (within tolerance; off-anchor inventions are dropped), min-gap
 * enforced, in-range, starting at the first section boundary — and
 * carrying only seedIds that exist in the catalog (unknown/misspelled
 * ids are snapped or fall back, mirroring the timestamp rule; a broken
 * ref must never reach the world model).
 */
export function sanitizeEvents(
  raw: unknown,
  summary: MusicalSummary,
  seeds: SeedEntry[],
): WorldEvent[] {
  if (!Array.isArray(raw)) return [];
  const anchors = collectAnchors(summary);
  if (anchors.length === 0 || seeds.length === 0) return [];

  let previousSeedId: string | null = null;
  const events: WorldEvent[] = [];
  for (const item of raw) {
    if (typeof item !== "object" || item === null) continue;
    const o = item as Record<string, unknown>;
    const timestamp = Number(o.timestamp);
    const prompt = typeof o.prompt === "string" ? o.prompt.trim() : "";
    if (!prompt || !Number.isFinite(timestamp)) continue;
    if (timestamp < 0 || timestamp >= summary.durationSec) continue;

    // Snap to the nearest real anchor; drop events that aren't near any
    // (the LLM invented an arbitrary time — the exact failure mode the
    // anchor rule exists to prevent).
    let best = anchors[0];
    for (const a of anchors) {
      if (Math.abs(a - timestamp) < Math.abs(best - timestamp)) best = a;
    }
    if (Math.abs(best - timestamp) > ANCHOR_SNAP_TOLERANCE_SEC) continue;

    const seedId = resolveSeedId(o.seedId, seeds, previousSeedId);
    previousSeedId = seedId;

    events.push({
      timestamp: best,
      seedId,
      prompt,
      transition: o.transition === "cut" ? "cut" : "morph",
    });
  }

  events.sort((a, b) => a.timestamp - b.timestamp);

  // Dedup + min gap (keep the earlier event).
  const spaced: WorldEvent[] = [];
  for (const e of events) {
    const prev = spaced[spaced.length - 1];
    if (prev && e.timestamp - prev.timestamp < MIN_EVENT_GAP_SEC) continue;
    spaced.push(e);
  }

  // The world needs an opening state at the very start of the song.
  if (spaced.length > 0) {
    spaced[0] = { ...spaced[0], timestamp: summary.sections[0]?.start ?? 0 };
  }
  return spaced;
}

// ── the call ──

export async function composeWithNemotron(
  summary: MusicalSummary,
  seeds: SeedEntry[],
): Promise<ComposeOutput> {
  const apiKey = getNemotronApiKey();
  if (!apiKey) {
    throw new ComposeError("NVIDIA_NEMO_KEY is not set on the server", 503);
  }
  if (seeds.length === 0) {
    throw new ComposeError(
      "Seed catalog is empty — check SEED_IMAGES_DIR (image-conditioned composition needs at least one seed)",
      503,
    );
  }

  const client = new OpenAI({ baseURL: NEMOTRON_BASE_URL, apiKey });

  const systemPrompt = ACTIVE_TEMPLATE.buildSystemPrompt(summary, seeds);
  const userMessage = ACTIVE_TEMPLATE.buildUserMessage(summary, seeds);

  let completion: OpenAI.Chat.Completions.ChatCompletion;
  try {
    // NVIDIA extension fields (chat_template_kwargs, reasoning_budget)
    // ride inside the params object — the OpenAI SDK passes unknown
    // fields through to the request body. Thinking stays on, but
    // budgeted so the reasoning trace can't consume the whole completion.
    const params = {
      model: NEMOTRON_MODEL,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userMessage },
      ],
      temperature: NEMOTRON_TEMPERATURE,
      top_p: NEMOTRON_TOP_P,
      max_tokens: NEMOTRON_MAX_TOKENS,
      chat_template_kwargs: { enable_thinking: true },
      reasoning_budget: NEMOTRON_REASONING_BUDGET,
    } as unknown as OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming;
    completion = await client.chat.completions.create(params);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    throw new ComposeError(`Nemotron call failed: ${msg}`);
  }

  const message = completion.choices?.[0]?.message as
    | (OpenAI.Chat.Completions.ChatCompletionMessage & {
        reasoning_content?: string;
      })
    | undefined;
  // The final answer is message.content; reasoning_content (when the
  // server splits it out) is the trace and is deliberately IGNORED.
  const content = message?.content ?? "";
  if (!content.trim()) {
    throw new ComposeError(
      "Nemotron returned an empty final answer (reasoning may have consumed the token budget)",
    );
  }

  const parsed = extractJsonObject(stripReasoning(content)) as {
    interpretation?: unknown;
    events?: unknown;
    timeline?: unknown; // tolerate the older field name
  };

  const events = sanitizeEvents(parsed.events ?? parsed.timeline, summary, seeds);
  if (events.length === 0) {
    throw new ComposeError("Nemotron returned no valid, anchor-aligned events");
  }

  // Token counts per song — sanity check that the summary is compact.
  const usage = completion.usage;
  const tokenUsage: TokenUsage | null = usage
    ? {
        promptTokens: usage.prompt_tokens,
        completionTokens: usage.completion_tokens,
        totalTokens: usage.total_tokens,
      }
    : null;
  console.info(
    `[compose] ${summary.songId}: model=${completion.model} ` +
      `prompt_tokens=${tokenUsage?.promptTokens ?? "?"} ` +
      `completion_tokens=${tokenUsage?.completionTokens ?? "?"} ` +
      `events=${events.length}`,
  );

  return {
    events,
    interpretation:
      typeof parsed.interpretation === "string" ? parsed.interpretation : "",
    tokenUsage,
    model: completion.model ?? NEMOTRON_MODEL,
  };
}
