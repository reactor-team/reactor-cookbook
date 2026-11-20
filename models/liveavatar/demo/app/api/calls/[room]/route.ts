import { NextResponse } from "next/server";

import { calls } from "@/lib/calls";
import { ROOM_NAME, deleteRoom } from "@/lib/livekit";
import { stopAvatar } from "@/lib/reactor";

export const runtime = "nodejs";

/** End a call: stop the avatar's session and delete the room. */
export async function DELETE(_request: Request, { params }: { params: Promise<{ room: string }> }) {
  const { room } = await params;
  if (!ROOM_NAME.test(room)) return NextResponse.json({ error: "unknown room" }, { status: 404 });
  const session = calls.get(room);
  calls.delete(room);
  const errors: string[] = [];
  if (session) await stopAvatar(session).catch((error: Error) => errors.push(error.message));
  await deleteRoom(room).catch((error: Error) => errors.push(error.message));
  console.log(JSON.stringify({ message: "call stopped", room, errors }));
  return new NextResponse(null, { status: 204 });
}
