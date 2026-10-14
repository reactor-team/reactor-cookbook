import type { Reactor, ReactorError, ReactorMessage } from "@reactor-team/js-sdk";

export const FAST_H3_MODEL = "reactor/fast-h3";
export const FAST_H3_DISPLAY_NAME = "FastH3";
export const FAST_H3_CANVAS = "1344×768";
export const FAST_H3_DEFAULT_CLIP_SECONDS = 10;
export const FAST_H3_MAX_PROMPT_LENGTH = 800;

export type FastH3ClipSeconds = 6 | 10 | 14;

export type FastH3ClipInfo = {
  clip_id: string;
  prompt: string;
  metadata?: string;
  frames: number;
  seconds: number;
  seed: number;
  continue_from_clip_id: string | null;
  has_starting_frame: boolean;
  ready: boolean;
};

export type FastH3Queue = {
  generation: FastH3ClipInfo[];
  playout: FastH3ClipInfo[];
  history: FastH3ClipInfo[];
};

export function clipFromMessageData(data: unknown): FastH3ClipInfo | null {
  if (!data || typeof data !== "object") return null;
  const clip = (data as { clip?: unknown }).clip;
  if (!clip || typeof clip !== "object") return null;
  const candidate = clip as Partial<FastH3ClipInfo>;
  return typeof candidate.clip_id === "string" && typeof candidate.prompt === "string"
    ? candidate as FastH3ClipInfo
    : null;
}

export function queueFromMessageData(data: unknown): FastH3Queue | null {
  if (!data || typeof data !== "object") return null;
  const candidate = data as Partial<FastH3Queue>;
  if (!Array.isArray(candidate.generation) || !Array.isArray(candidate.playout) || !Array.isArray(candidate.history)) {
    return null;
  }
  return candidate as FastH3Queue;
}

/** Shown when FastH3 refuses a command and gives no reason. */
export const FAST_H3_GENERIC_REFUSAL = "The model refused this command.";

export type FastH3CommandResult =
  | { ok: true; reply: ReactorMessage | undefined }
  | { ok: false; code: string | null; reason: string };

const REFUSAL_PATTERN = /^command '[^']*' failed \([^)]*\): ([\s\S]*)$/;

/**
 * Send one FastH3 command and read its refusal from that same call.
 *
 * `sendCommand` never rejects. When FastH3 refuses a command, the call
 * resolves `undefined` and the SDK emits a `ReactorError` on the `error`
 * event, with `operation: "sendCommand"` and the model's code (for example
 * `queue_full`). Only the client that sent the command is told.
 *
 * A FastH3 release that reports no code still refuses a command that declares
 * a reply by resolving `undefined`. Set `expectsReply` for such a command, so
 * that case reads as a generic refusal instead of a success.
 */
export async function sendFastH3Command(
  reactor: Reactor,
  command: string,
  data: Record<string, unknown> = {},
  { expectsReply = false }: { expectsReply?: boolean } = {},
): Promise<FastH3CommandResult> {
  let refusal: ReactorError | null = null;
  const onError = (error: ReactorError) => {
    if (error.operation === "sendCommand" && error.message.startsWith(`command '${command}' failed`)) refusal = error;
  };
  reactor.on("error", onError);
  let reply: ReactorMessage | undefined;
  try {
    reply = await reactor.sendCommand(command, data);
  } finally {
    reactor.off("error", onError);
  }
  const error = refusal as ReactorError | null;
  if (error) {
    const reason = REFUSAL_PATTERN.exec(error.message)?.[1] || FAST_H3_GENERIC_REFUSAL;
    return { ok: false, code: error.code, reason };
  }
  if (expectsReply && reply === undefined) return { ok: false, code: null, reason: FAST_H3_GENERIC_REFUSAL };
  return { ok: true, reply };
}
