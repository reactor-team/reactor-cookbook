import { NextResponse } from "next/server";
import { reactor } from "../../../lib/server/reactor";

// A generation's status, and its video file once it has succeeded.
export async function GET(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  if (!/^[0-9a-f-]{36}$/i.test(id)) return NextResponse.json({ error: "Generation not found." }, { status: 404 });
  return reactor(req, `/generations/${id}`);
}
