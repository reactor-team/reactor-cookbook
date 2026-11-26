import { NextResponse } from "next/server";
import { z } from "zod";
import { reactor } from "../../lib/server/reactor";

// Reserves an upload for one reference file. The answer carries a file_id for
// the generation and a presigned URL the browser PUTs the bytes to.
const FileRequest = z.object({
  name: z.string().min(1).max(255),
  size: z.number().int().positive().max(25 * 1024 * 1024),
  mime_type: z.string().regex(/^(image|audio|video)\/[\w.+-]+$/),
});

export async function POST(req: Request) {
  const parsed = FileRequest.safeParse(await req.json().catch(() => null));
  if (!parsed.success) return NextResponse.json({ error: "Bad file request." }, { status: 400 });
  return reactor(req, "/files", parsed.data);
}
