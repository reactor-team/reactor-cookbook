import { KEY_HEADERS, type Writer } from "../keys";

const ENV = { reactor: "REACTOR_API_KEY", anthropic: "ANTHROPIC_API_KEY", openai: "OPENAI_API_KEY" } as const;

/** A key pasted in the page wins over the one in .env.local. */
export function keyFor(req: Request, name: keyof typeof ENV): string | undefined {
  const pasted = req.headers.get(KEY_HEADERS[name])?.trim();
  return pasted || process.env[ENV[name]] || undefined;
}

/** The writer the page asked for, or whichever one has a key. */
export function writerFor(req: Request): { writer: Writer; key: string } | null {
  const asked = req.headers.get(KEY_HEADERS.writer) === "openai" ? "openai" : "claude";
  const order: Writer[] = asked === "openai" ? ["openai", "claude"] : ["claude", "openai"];
  for (const writer of order) {
    const key = keyFor(req, writer === "claude" ? "anthropic" : "openai");
    if (key) return { writer, key };
  }
  return null;
}

export const REACTOR_API = "https://api.reactor.inc";
