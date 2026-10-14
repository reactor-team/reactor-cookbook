export async function POST(request: Request) {
  const origin = request.headers.get('origin');
  if (origin && origin !== new URL(request.url).origin) return Response.json({ error: 'Origin not allowed' }, { status: 403 });
  const key = process.env.REACTOR_API_KEY;
  if (!key) return Response.json({ error: 'Set REACTOR_API_KEY on the server to connect.' }, { status: 503 });
  try {
    const response = await fetch('https://api.reactor.inc/tokens', {
      method: 'POST', headers: { 'Reactor-API-Key': key, 'Content-Type': 'application/json' },
      body: JSON.stringify({ authorization_details: [{ type: 'session', resources: { models: { match: ['reactor/fast-h3-omni'] } }, constraints: { max_sessions: 1 } }] }),
      signal: AbortSignal.timeout(20000),
    });
    if (!response.ok) return Response.json({ error: `Reactor authentication failed (${response.status}). Check model access and your API key.` }, { status: 502 });
    const { jwt } = await response.json() as { jwt: string };
    return Response.json({ jwt }, { headers: { 'Cache-Control': 'no-store' } });
  } catch { return Response.json({ error: 'Unable to reach Reactor. Try connecting again.' }, { status: 502 }); }
}
