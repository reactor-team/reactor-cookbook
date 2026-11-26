import { NextResponse } from "next/server";
import { z } from "zod";
import { CHECK_SYSTEM_PROMPT, checkMessage } from "../../lib/check";
import { writeVerdict, writerError, type Picture } from "../../lib/server/writer";

const CheckBody = z.object({
  beat: z.string().max(2000),
  prompt: z.string().max(8000),
  cast: z.array(z.object({ label: z.string().max(60), look: z.string().max(600) })).max(12),
  frame: z.url({ protocol: /^https$/ }),
  earlier: z.url({ protocol: /^https$/ }).optional(),
});

// Only Reactor's file storage: the route fetches these URLs server-side.
const allowed = (url: string) => {
  const host = new URL(url).hostname;
  return host.endsWith(".amazonaws.com") || host === "reactor.inc" || host.endsWith(".reactor.inc");
};

async function picture(url: string): Promise<Picture | null> {
  if (!allowed(url)) return null;
  const res = await fetch(url, { cache: "no-store", redirect: "error" }).catch(() => null);
  if (!res?.ok) return null;
  const type = res.headers.get("content-type")?.split(";")[0] ?? "image/png";
  const mediaType = (["image/jpeg", "image/png", "image/webp"].includes(type) ? type : "image/png") as Picture["mediaType"];
  const bytes = Buffer.from(await res.arrayBuffer());
  if (bytes.length > 4_500_000) return null;
  return { data: bytes.toString("base64"), mediaType };
}

// Judges a built clip's last frame against its beat and cast (generation API only).
export async function POST(req: Request) {
  const parsed = CheckBody.safeParse(await req.json().catch(() => null));
  if (!parsed.success) return NextResponse.json({ error: "Bad check request." }, { status: 400 });
  const body = parsed.data;
  const [frame, earlier] = await Promise.all([picture(body.frame), body.earlier ? picture(body.earlier) : Promise.resolve(null)]);
  if (!frame) return NextResponse.json({ error: "Could not read the clip's last frame." }, { status: 502 });
  try {
    const pictures = earlier ? [earlier, frame] : [frame];
    const verdict = await writeVerdict(req, CHECK_SYSTEM_PROMPT, checkMessage(body, Boolean(earlier)), pictures);
    if (!verdict) return NextResponse.json({ error: "The check returned no verdict." }, { status: 502 });
    return NextResponse.json({ verdict });
  } catch (error) {
    return writerError(error, "Shot check");
  }
}
