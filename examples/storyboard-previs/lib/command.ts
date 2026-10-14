import type { Reactor, ReactorError, ReactorMessage } from '@reactor-team/js-sdk';

/** Shown when the model refuses a command and gives no reason. */
export const GENERIC_REFUSAL = 'The model refused this command.';

// FastH3 Omni commands that answer with a reply message on success.
const REPLY_COMMANDS = new Set(['enqueue', 'set_canvas', 'pop', 'move', 'set_seed', 'set_clip_seconds', 'set_autoplay', 'set_flush_on_clip_end']);
const MODEL_REFUSAL = /^command '([^']*)' failed \(([^)]*)\): ([\s\S]*)$/;

/** A command the model, or the SDK, refused. `code` is the model's code when it sent one. */
export class CommandRefused extends Error {
  constructor(readonly command: string, readonly code: string | null, reason: string) {
    super(reason);
    this.name = 'CommandRefused';
  }
}

/**
 * Send one command and read its refusal from that same call.
 *
 * `sendCommand` never rejects. When the model refuses a command, the call
 * resolves `undefined` and the SDK emits a `ReactorError` on the `error` event,
 * with `operation: "sendCommand"` and the model's code (for example
 * `queue_full`). Only the client that sent the command is told. This listens
 * for the length of the call and throws `CommandRefused` with the model's
 * reason.
 *
 * A release that reports no code still refuses a command that declares a reply
 * by resolving `undefined`; that case throws the generic refusal.
 */
export async function sendModelCommand(reactor: Reactor, name: string, data: Record<string, unknown> = {}): Promise<ReactorMessage | undefined> {
  let refusal: ReactorError | null = null;
  const onError = (error: ReactorError) => {
    if (error.operation !== 'sendCommand') return;
    const refused = MODEL_REFUSAL.exec(error.message);
    if (!refused || refused[1] === name) refusal = error;
  };
  reactor.on('error', onError);
  let reply: ReactorMessage | undefined;
  try {
    reply = await reactor.sendCommand(name, data);
  } finally {
    reactor.off('error', onError);
  }
  const error = refusal as ReactorError | null;
  if (error) throw new CommandRefused(name, error.code, MODEL_REFUSAL.exec(error.message)?.[3] || error.message || GENERIC_REFUSAL);
  if (reply === undefined && REPLY_COMMANDS.has(name)) throw new CommandRefused(name, null, GENERIC_REFUSAL);
  return reply;
}
