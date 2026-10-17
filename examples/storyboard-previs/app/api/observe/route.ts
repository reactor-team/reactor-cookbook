import { isAllowedOrigin } from '@/lib/request-origin';
import { decisionAnswers, decisionConfig, decisionRequest, parseEvaluation } from '@/lib/decisions';

export async function GET() {
  const { provider, key, model } = decisionConfig(process.env);
  return Response.json({ provider, model, configured: !!key, director: !!process.env.ANTHROPIC_API_KEY }, { headers: { 'Cache-Control': 'no-store' } });
}

export async function POST(request: Request) {
  if (!isAllowedOrigin(request, process.env.PREVIS_PUBLIC_ORIGIN)) return Response.json({ error: 'Origin not allowed.' }, { status: 403 });
  let input;
  try {
    // Enforce the limit even when Content-Length is missing or false.
    const reader = request.body?.getReader();
    if (!reader) throw new Error('Missing request body.');
    const chunks: Uint8Array[] = []; let size = 0;
    while (true) {
      const { done, value } = await reader.read(); if (done) break;
      size += value.byteLength;
      if (size > 1700000) { await reader.cancel(); return Response.json({ error: 'Request too large.' }, { status: 413 }); }
      chunks.push(value);
    }
    input = parseEvaluation(JSON.parse(Buffer.concat(chunks).toString('utf8')));
  } catch (e) {
    return Response.json({ error: e instanceof Error ? e.message : 'Invalid evaluation.' }, { status: 400 });
  }
  try {
    const { provider, key } = decisionConfig(process.env);
    if (!key) return Response.json({ error: 'Set OPENAI_API_KEY on the server to start monitoring.' }, { status: 503 });
    const started = performance.now();
    const response = await fetch('https://api.openai.com/v1/decisions', {
      method: 'POST',
      headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' },
      body: JSON.stringify(decisionRequest(input)),
      signal: AbortSignal.any([request.signal, AbortSignal.timeout(8000)]),
      cache: 'no-store',
    });
    if (!response.ok) return Response.json({ error: `Visual monitoring unavailable (${response.status}). Check the OpenAI key, access and limits.` }, { status: response.status === 429 ? 429 : 502 });
    const answers = decisionAnswers(await response.json(), input.questions);
    return Response.json({ answers, provider, latencyMs: Math.round(performance.now() - started) }, { headers: { 'Cache-Control': 'no-store' } });
  } catch {
    return Response.json({ error: 'Visual monitoring timed out or could not reach OpenAI. No actions were triggered.' }, { status: 502 });
  }
}
