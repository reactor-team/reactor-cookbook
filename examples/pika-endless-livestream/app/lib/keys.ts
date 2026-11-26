// Keys pasted into the page. They live in this tab's sessionStorage and travel
// only to this app's own route handlers, which use them in place of the ones in
// .env.local. The Reactor key is still exchanged on the server: the browser
// never calls Reactor with it.

export type Writer = "claude" | "openai";
export type Keys = { reactor: string; anthropic: string; openai: string; writer: Writer };

export const EMPTY_KEYS: Keys = { reactor: "", anthropic: "", openai: "", writer: "claude" };

export const KEY_HEADERS = {
  reactor: "x-reactor-api-key",
  anthropic: "x-anthropic-api-key",
  openai: "x-openai-api-key",
  writer: "x-writer",
} as const;

const STORAGE = "pika-endless-livestream:keys";
let current: Keys = EMPTY_KEYS;

export function loadKeys(): Keys {
  try {
    const raw = sessionStorage.getItem(STORAGE);
    if (raw) current = { ...EMPTY_KEYS, ...(JSON.parse(raw) as Partial<Keys>) };
  } catch {
    // Storage blocked: keys last until the page reloads.
  }
  return current;
}

export function saveKeys(keys: Keys) {
  current = keys;
  try {
    sessionStorage.setItem(STORAGE, JSON.stringify(keys));
  } catch {
    // Storage blocked: keys last until the page reloads.
  }
}

// Keys the server has in its environment. A key set there has no field, so a
// copy saved in the tab earlier is never sent: the server's key wins.
let onServer: Partial<Record<"reactor" | "anthropic" | "openai", boolean>> = {};
export function setServerKeys(keys: typeof onServer) {
  onServer = keys;
}

/** Headers carrying the pasted keys to this app's route handlers. */
export function keyHeaders(): Record<string, string> {
  const h: Record<string, string> = { [KEY_HEADERS.writer]: current.writer };
  if (current.reactor && !onServer.reactor) h[KEY_HEADERS.reactor] = current.reactor;
  if (current.anthropic && !onServer.anthropic) h[KEY_HEADERS.anthropic] = current.anthropic;
  if (current.openai && !onServer.openai) h[KEY_HEADERS.openai] = current.openai;
  return h;
}
