// Turns a short "what happens next" beat into full H3 Omni direction with Claude Haiku.
// The key stays on the server; the browser only ever sees the expanded prompt.
type Body = {
  beat: string;
  mode: 'continue' | 'cut';
  story: { title: string; shots: { title: string; prompt: string }[] };
  source: { title: string; prompt: string } | null;
  pictures: string[];
};

const SYSTEM = `You write shot direction for FastH3 Omni, a video model that generates one clip of 8 seconds with synchronized sound from a text prompt plus reference pictures. Follow the model's prompt guide exactly.

The subject and picture binding lines are added by the app from the previous shot; do not write them. Use the same <Subject N> and <Picture N> labels as the previous shot's prompt. Never invent names or new people.

Format, one item per line, in this order:
1. One line starting "In one shot," describing a single clear beat that fits 8 seconds, using the <Subject N> labels.
2. "Camera:" one move or framing.
3. "Dialogue:" either None, or exactly one short line in this form: She says, “…” (under 10 words).
4. "Soundscape:" two or three concrete sounds.
5. "Music:" None, unless the beat clearly calls for it.

Keep the whole prompt under 110 words. Plain sentences. No headings, bullets, or quotes around the prompt.

If the mode is "continue", the clip starts from the final frame of the previous shot: begin from that exact moment and place, keep the same people on screen, and name what carries over (appearance, light, camera motion). If the mode is "cut", it is a new angle or a new moment in the same story and location.

Return only the prompt text. Start directly with "In one shot," and never add a preamble.`;

export async function POST(request: Request) {
  const origin = request.headers.get('origin');
  if (origin && origin !== new URL(request.url).origin) return Response.json({ error: 'Origin not allowed' }, { status: 403 });
  const key = process.env.ANTHROPIC_API_KEY;
  if (!key) return Response.json({ error: 'Set ANTHROPIC_API_KEY on the server to use the director.' }, { status: 503 });
  let body: Body;
  try { body = await request.json() as Body; } catch { return Response.json({ error: 'Bad request' }, { status: 400 }); }
  const beat = (body.beat || '').trim().slice(0, 500);
  if (!beat) return Response.json({ error: 'Empty beat' }, { status: 400 });

  const pictures = (body.pictures || []).slice(0, 9).map((name, i) => `<Picture ${i + 1}> = ${name}`).join('\n') || '(no pictures)';
  const script = (body.story?.shots || []).slice(0, 12).map((s, i) => `${i + 1}. ${s.title}: ${s.prompt}`).join('\n');
  const user = `Story: ${body.story?.title || 'Untitled'}

Script so far:
${script || '(empty)'}

Pictures available for this clip, in order:
${pictures}

Previous shot (the clip on screen now): ${body.source ? `${body.source.title}: ${body.source.prompt}` : '(none)'}

Mode: ${body.mode === 'cut' ? 'cut' : 'continue'}

The director's beat for the next clip: ${beat}

Write the prompt.`;

  try {
    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: { 'x-api-key': key, 'anthropic-version': '2023-06-01', 'content-type': 'application/json' },
      body: JSON.stringify({ model: process.env.DIRECTOR_MODEL || 'claude-haiku-5-5', max_tokens: 1500, system: SYSTEM, messages: [{ role: 'user', content: user }] }),
      signal: AbortSignal.timeout(15000),
    });
    if (!response.ok) {
      const detail = await response.text().catch(() => '');
      return Response.json({ error: `Director unavailable (${response.status}). ${detail.slice(0, 200)}` }, { status: 502 });
    }
    const data = await response.json() as { content?: { type: string; text?: string }[] };
    const prompt = (data.content || []).filter(c => c.type === 'text').map(c => c.text || '').join('').trim().replace(/^["“]|["”]$/g, '');
    if (!prompt) return Response.json({ error: `Director returned nothing (${(data.content || []).map(c => c.type).join(',') || 'no content'}; stop ${(data as { stop_reason?: string }).stop_reason ?? '?'}).` }, { status: 502 });
    return Response.json({ prompt }, { headers: { 'Cache-Control': 'no-store' } });
  } catch (e) {
    return Response.json({ error: `Director failed: ${e instanceof Error ? e.message : String(e)}` }, { status: 502 });
  }
}
