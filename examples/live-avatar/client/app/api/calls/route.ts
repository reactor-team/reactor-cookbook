import { readFile } from "node:fs/promises";
import path from "node:path";

import { NextResponse } from "next/server";

import { calls } from "@/lib/calls";
import { createRoom, deleteRoom, dispatchAgent, mintAvatarToken, mintUserToken, roomUrl } from "@/lib/livekit";
import { isPreset } from "@/lib/presets";
import { type Portrait, startAvatar } from "@/lib/reactor";

export const runtime = "nodejs";

const MAX_PORTRAIT_BYTES = 10 * 1024 * 1024;

/**
 * Start a call: create a room, bring the voice agent in, start the avatar model's session with the portrait and have
 * it join, then hand the browser its own token. The avatar is in the room before the user.
 *
 * Form fields, all optional: `preset` (one of the portraits in `assets/`) or `portrait` (an uploaded image); without
 * either, the model's default portrait. `voice` (an OpenAI Realtime voice) and `instructions` (the agent's system
 * prompt).
 */
export async function POST(request: Request) {
  const form = await request.formData();
  const file = form.get("portrait");
  const preset = form.get("preset");
  let portrait: Portrait | null = null;
  if (typeof preset === "string" && preset) {
    if (!isPreset(preset)) return NextResponse.json({ error: "unknown preset" }, { status: 400 });
    const bytes = await readFile(path.join(process.cwd(), "assets", `${preset}.jpg`));
    portrait = { name: `${preset}.jpg`, type: "image/jpeg", bytes: new Uint8Array(bytes) };
  } else if (file instanceof File && file.size > 0) {
    if (!file.type.startsWith("image/") || file.size > MAX_PORTRAIT_BYTES) {
      return NextResponse.json({ error: "the portrait must be an image of at most 10 MB" }, { status: 400 });
    }
    portrait = { name: file.name || "portrait", type: file.type, bytes: new Uint8Array(await file.arrayBuffer()) };
  }
  const metadata: Record<string, string> = {};
  for (const key of ["voice", "instructions"]) {
    const value = form.get(key);
    if (typeof value === "string" && value.trim()) metadata[key] = value.trim().slice(0, 4000);
  }

  let room: string | null = null;
  const started = Date.now();
  try {
    room = await createRoom();
    const agent = await dispatchAgent(room, metadata);
    const session = await startAvatar({
      roomUrl: roomUrl(),
      avatarToken: await mintAvatarToken(room, agent),
      portrait,
    });
    calls.set(room, session);
    const user = await mintUserToken(room);
    console.log(JSON.stringify({ message: "call started", room, session: session.sessionId, ms: Date.now() - started }));
    return NextResponse.json({ room, ...user });
  } catch (error) {
    console.error(JSON.stringify({ message: "call failed", room, error: (error as Error).message }));
    if (room) await deleteRoom(room).catch(() => undefined);
    return NextResponse.json({ error: (error as Error).message }, { status: 502 });
  }
}
