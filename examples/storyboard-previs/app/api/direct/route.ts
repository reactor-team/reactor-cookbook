import { isAllowedOrigin } from '@/lib/request-origin';
// Expand a beat using the creative brief, recent observations, and planned story.
// Both provider credentials and external-feed evidence stay out of generated code.
import { parseDirectorInput, directorMessage, DIRECTOR_SYSTEM } from '@/lib/director';

export async function POST(request: Request) {
  if (!isAllowedOrigin(request, process.env.PREVIS_PUBLIC_ORIGIN)) return Response.json({ error: 'Origin not allowed.' }, { status: 403 });
  let body;
  try {
    const reader = request.body?.getReader();
    if (!reader) throw new Error('Missing request body.');
    const chunks: Uint8Array[] = []; let size = 0;
    while (true) {
      const { done, value } = await reader.read(); if (done) break;
      size += value.length;
      if (size > 100000) { await reader.cancel(); return Response.json({ error: 'Request too large.' }, { status: 413 }); }
      chunks.push(value);
    }
    body = parseDirectorInput(JSON.parse(Buffer.concat(chunks).toString('utf8')));
  } catch (e) { return Response.json({ error: e instanceof Error ? e.message : 'Bad request' }, { status: 400 }); }
  const key = process.env.ANTHROPIC_API_KEY;
  if (!key) return Response.json({ error: 'Set ANTHROPIC_API_KEY on the server to use the director.' }, { status: 503 });
  try {
    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: { 'x-api-key': key, 'anthropic-version': '2023-06-01', 'content-type': 'application/json' },
      body: JSON.stringify({ model: process.env.DIRECTOR_MODEL || 'claude-haiku-5-5', max_tokens: 1500, system: DIRECTOR_SYSTEM, messages: [{ role: 'user', content: directorMessage(body) }] }),
      signal: AbortSignal.any([request.signal, AbortSignal.timeout(15000)]),
    });
    if (!response.ok) return Response.json({ error: `Director unavailable (${response.status}). Check the server configuration and provider limits.` }, { status: 502 });
    const data = await response.json() as { content?: { type: string; text?: string }[] };
    const prompt = (data.content || []).filter(c => c.type === 'text').map(c => c.text || '').join('').trim().replace(/^["“]|["”]$/g, '');
    if (!prompt || prompt.length > 5000 || !prompt.startsWith('In one shot,')) return Response.json({ error: 'Director did not return valid shot direction. Try again.' }, { status: 502 });
    return Response.json({ prompt }, { headers: { 'Cache-Control': 'no-store' } });
  } catch {
    return Response.json({ error: 'Director timed out or could not reach the provider. Try again.' }, { status: 502 });
  }
}
