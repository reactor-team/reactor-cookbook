import "server-only";

import { AccessToken, AgentDispatchClient, RoomServiceClient } from "livekit-server-sdk";

/** The identity the avatar joins the room with. */
export const AVATAR_IDENTITY = "avatar";
/** Rooms this app creates are named `avatar-<10 hex>`. */
export const ROOM_NAME = /^avatar-[0-9a-f]{10}$/;
/** `ParticipantInfo.Kind.AGENT` in LiveKit's protocol. */
const PARTICIPANT_KIND_AGENT = 4;

function credentials() {
  const url = process.env.LIVEKIT_URL?.trim();
  const apiKey = process.env.LIVEKIT_API_KEY?.trim();
  const apiSecret = process.env.LIVEKIT_API_SECRET?.trim();
  if (!url || !apiKey || !apiSecret) throw new Error("LIVEKIT_URL, LIVEKIT_API_KEY or LIVEKIT_API_SECRET is not set");
  const ws = url.replace(/^http(s?):\/\//, "ws$1://");
  return { ws, http: ws.replace(/^ws(s?):\/\//, "http$1://"), apiKey, apiSecret };
}

function rooms() {
  const { http, apiKey, apiSecret } = credentials();
  return new RoomServiceClient(http, apiKey, apiSecret);
}

/** The URL participants join the room server with. */
export function roomUrl(): string {
  return credentials().ws;
}

/** Create a fresh room for one call. */
export async function createRoom(): Promise<string> {
  const name = `avatar-${crypto.randomUUID().replaceAll("-", "").slice(0, 10)}`;
  await rooms().createRoom({ name, emptyTimeout: 60, departureTimeout: 20 });
  return name;
}

/** Delete the room: every participant leaves it, the avatar included. */
export async function deleteRoom(room: string): Promise<void> {
  await rooms().deleteRoom(room);
}

/**
 * Dispatch the voice agent named `AVATAR_AGENT_NAME` into `room` and wait until it has joined. Returns its identity,
 * which the avatar's token names as the participant it publishes on behalf of.
 */
export async function dispatchAgent(room: string, metadata: Record<string, string>, timeoutMs = 20_000): Promise<string> {
  const { http, apiKey, apiSecret } = credentials();
  const agentName = process.env.AVATAR_AGENT_NAME?.trim();
  if (!agentName) throw new Error("AVATAR_AGENT_NAME is not set");
  await new AgentDispatchClient(http, apiKey, apiSecret).createDispatch(room, agentName, {
    metadata: JSON.stringify(metadata),
  });
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const agent = (await rooms().listParticipants(room)).find(
      (p) => p.kind === PARTICIPANT_KIND_AGENT && p.identity !== AVATAR_IDENTITY,
    );
    if (agent) return agent.identity;
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error(`the voice agent ${agentName} did not join within ${timeoutMs / 1000} s`);
}

/**
 * The avatar's token, as LiveKit's avatar plugins mint it: kind `agent`, publishing audio, video and data on behalf of
 * the voice agent (`lk.publish_on_behalf`), so the user sees one participant speaking.
 */
export async function mintAvatarToken(room: string, agentIdentity: string): Promise<string> {
  const { apiKey, apiSecret } = credentials();
  const at = new AccessToken(apiKey, apiSecret, { identity: AVATAR_IDENTITY, ttl: "2h" });
  at.kind = "agent";
  at.attributes = { "lk.publish_on_behalf": agentIdentity };
  at.addGrant({ roomJoin: true, room, canPublish: true, canPublishData: true, canSubscribe: false });
  return at.toJwt();
}

/** The user's token for `room`. */
export async function mintUserToken(room: string): Promise<{ url: string; token: string }> {
  const { ws, apiKey, apiSecret } = credentials();
  const at = new AccessToken(apiKey, apiSecret, {
    identity: `user-${crypto.randomUUID().slice(0, 6)}`,
    name: "You",
    ttl: "2h",
  });
  at.addGrant({ roomJoin: true, room, canPublish: true, canSubscribe: true, canPublishData: true });
  return { url: ws, token: await at.toJwt() };
}
