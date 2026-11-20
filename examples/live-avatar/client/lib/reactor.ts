import "server-only";

/**
 * Starting and stopping the avatar model's session, which joins the LiveKit room as the avatar.
 *
 * Two backends, picked by the environment:
 * - a runtime you serve yourself (`REACTOR_RUNTIME_URL`, e.g. `reactor run` on your GPU machine);
 * - a model deployed on Reactor (`REACTOR_API_KEY` and `REACTOR_MODEL`), through the Reactor API.
 *
 * Either way the start is the same three moves: start a session whose starting input runs `set_avatar` with the
 * portrait (when one was picked), upload the portrait, and ask the session to join the room with the avatar's token.
 */

export interface Portrait {
  name: string;
  type: string;
  bytes: Uint8Array;
}

export interface AvatarStart {
  roomUrl: string;
  avatarToken: string;
  portrait: Portrait | null;
}

/** A started session, enough to stop it. */
export interface AvatarSession {
  backend: "runtime" | "reactor";
  sessionId: string;
}

const API_VERSION = "1";
const JOIN_TIMEOUT_MS = 60_000;

function backend(): "runtime" | "reactor" {
  if (process.env.REACTOR_RUNTIME_URL?.trim()) return "runtime";
  if (process.env.REACTOR_API_KEY?.trim() && process.env.REACTOR_MODEL?.trim()) return "reactor";
  throw new Error("set REACTOR_RUNTIME_URL, or REACTOR_API_KEY and REACTOR_MODEL");
}

/** Which backend serves the avatar, for the page's status line (no secret). */
export function describeBackend(): string {
  try {
    return backend() === "runtime"
      ? `runtime at ${process.env.REACTOR_RUNTIME_URL}`
      : `${process.env.REACTOR_MODEL} on the Reactor API`;
  } catch (error) {
    return (error as Error).message;
  }
}

async function call(url: string, init: RequestInit & { json?: unknown } = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  let body = init.body;
  if (init.json !== undefined) {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(init.json);
  }
  return fetch(url, { ...init, headers, body, signal: AbortSignal.timeout(30_000) });
}

async function expect(resp: Response, ok: number[], what: string): Promise<Record<string, unknown>> {
  const text = await resp.text();
  if (!ok.includes(resp.status)) throw new Error(`${what}: HTTP ${resp.status} ${text.slice(0, 300)}`);
  try {
    return text ? (JSON.parse(text) as Record<string, unknown>) : {};
  } catch {
    return {};
  }
}

/** The starting input that sets the portrait, referencing an upload by id. */
function startingInput(uploadId: string) {
  return { commands: [{ command: "set_avatar", data: { image: { upload_id: uploadId } } }] };
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/** Wait until the session's room connection is open; throw with its reason when the join failed. */
async function waitForJoin(connectionUrl: string, headers: HeadersInit): Promise<void> {
  const deadline = Date.now() + JOIN_TIMEOUT_MS;
  while (Date.now() < deadline) {
    const resp = await call(connectionUrl, { headers });
    if (resp.status === 200) {
      const status = (await resp.json()) as { state: string; reason?: string };
      if (status.state === "open") return;
      throw new Error(`the avatar could not join the room: ${status.state} ${status.reason ?? ""}`);
    }
    await resp.body?.cancel();
    await sleep(250);
  }
  throw new Error("the avatar did not join the room in time");
}

// -- a runtime you serve ---------------------------------------------------------------------------------------------

async function startOnRuntime({ roomUrl, avatarToken, portrait }: AvatarStart): Promise<AvatarSession> {
  const base = process.env.REACTOR_RUNTIME_URL!.trim().replace(/\/$/, "");
  const uploadId = portrait ? crypto.randomUUID() : null;
  const started = await expect(
    await call(`${base}/start_session`, { method: "POST", json: uploadId ? { starting_input: startingInput(uploadId) } : {} }),
    [200],
    "the runtime refused the session",
  );
  const sessionId = String(started.session_id);
  try {
    if (portrait && uploadId) {
      // The starting `set_avatar` waits for these bytes: the runtime holds a starting command up to 10 s for its upload.
      const slot = await expect(
        await call(`${base}/sessions/${sessionId}/uploads`, {
          method: "POST",
          json: { name: portrait.name, size: portrait.bytes.length, mime_type: portrait.type, upload_id: uploadId },
        }),
        [201],
        "the runtime refused the upload",
      );
      await expect(
        await call(String(slot.presigned_url), {
          method: "PUT",
          headers: { "Content-Type": portrait.type },
          body: portrait.bytes as BodyInit,
        }),
        [200, 201, 204],
        "the portrait upload failed",
      );
    }
    const accepted = await expect(
      await call(`${base}/sessions/${sessionId}/transport/livekit/connections`, {
        method: "POST",
        json: { url: roomUrl, token: avatarToken },
      }),
      [202],
      "the runtime refused the room connection",
    );
    await waitForJoin(`${base}/sessions/${sessionId}/transport/livekit/connections/${accepted.connection_id}`, {});
  } catch (error) {
    await stopOnRuntime().catch(() => undefined);
    throw error;
  }
  return { backend: "runtime", sessionId };
}

async function stopOnRuntime(): Promise<void> {
  const base = process.env.REACTOR_RUNTIME_URL!.trim().replace(/\/$/, "");
  const resp = await call(`${base}/stop_session`, { method: "POST", json: { reason: "call ended" } });
  await resp.body?.cancel();
}

// -- a model deployed on Reactor -------------------------------------------------------------------------------------

function reactorApi() {
  const url = (process.env.REACTOR_API_URL?.trim() || "https://api.reactor.inc").replace(/\/$/, "");
  const headers = { Authorization: `Bearer ${process.env.REACTOR_API_KEY!.trim()}`, "Reactor-API-Version": API_VERSION };
  return { url, headers };
}

async function startOnReactor({ roomUrl, avatarToken, portrait }: AvatarStart): Promise<AvatarSession> {
  const { url, headers } = reactorApi();
  const extra: Record<string, string> = {};
  if (portrait) {
    const file = await expect(
      await call(`${url}/files`, {
        method: "POST",
        headers,
        json: { name: portrait.name, size: portrait.bytes.length, mime_type: portrait.type },
      }),
      [201],
      "the Reactor API refused the file",
    );
    await expect(
      await call(String(file.presigned_url), {
        method: "PUT",
        headers: { "Content-Type": portrait.type },
        body: portrait.bytes as BodyInit,
      }),
      [200, 201, 204],
      "the portrait upload failed",
    );
    extra.file_ids = JSON.stringify([file.file_id]);
    extra.starting_input = JSON.stringify(startingInput(String(file.file_id)));
  }
  const created = await expect(
    await call(`${url}/sessions`, {
      method: "POST",
      headers,
      json: { model: { name: process.env.REACTOR_MODEL!.trim() }, ...(portrait ? { extra_configs: extra } : {}) },
    }),
    [201],
    "the Reactor API refused the session",
  );
  const sessionId = String(created.session_id);
  try {
    const deadline = Date.now() + JOIN_TIMEOUT_MS;
    for (;;) {
      const info = await expect(await call(`${url}/sessions/${sessionId}`, { headers }), [200], "reading the session");
      if (info.selected_transport) break;
      if (Date.now() > deadline) throw new Error("the session did not get a transport in time");
      await sleep(250);
    }
    const accepted = await expect(
      await call(`${url}/sessions/${sessionId}/transport/livekit/connections`, {
        method: "POST",
        headers,
        json: { url: roomUrl, token: avatarToken },
      }),
      [202],
      "the Reactor API refused the room connection",
    );
    await waitForJoin(`${url}/sessions/${sessionId}/transport/livekit/connections/${accepted.connection_id}`, headers);
  } catch (error) {
    await stopOnReactor(sessionId).catch(() => undefined);
    throw error;
  }
  return { backend: "reactor", sessionId };
}

async function stopOnReactor(sessionId: string): Promise<void> {
  const { url, headers } = reactorApi();
  const resp = await call(`${url}/sessions/${sessionId}`, { method: "DELETE", headers, json: { reason: "call ended" } });
  await resp.body?.cancel();
}

// -- either --------------------------------------------------------------------------------------------------------------

/** Start the avatar's session and have it join the room; resolves once the avatar is in. */
export function startAvatar(start: AvatarStart): Promise<AvatarSession> {
  return backend() === "runtime" ? startOnRuntime(start) : startOnReactor(start);
}

/** End the avatar's session; the avatar leaves the room. */
export function stopAvatar(session: AvatarSession): Promise<void> {
  return session.backend === "runtime" ? stopOnRuntime() : stopOnReactor(session.sessionId);
}
