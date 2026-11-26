import { NextResponse } from "next/server";
import { REACTOR_API, keyFor } from "./keys";

/** Forward one call to Reactor's API with the API key, and pass its answer back. */
export async function reactor(req: Request, path: string, body?: unknown): Promise<Response> {
  const apiKey = keyFor(req, "reactor");
  if (!apiKey) {
    return NextResponse.json({ error: "No Reactor key: paste one under Keys, or set REACTOR_API_KEY in .env.local" }, { status: 500 });
  }
  try {
    const res = await fetch(`${REACTOR_API}${path}`, {
      method: body === undefined ? "GET" : "POST",
      headers: { "Reactor-API-Key": apiKey, "Content-Type": "application/json" },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
      cache: "no-store",
    });
    const data = (await res.json().catch(() => null)) as unknown;
    return NextResponse.json(data ?? { error: `Reactor returned ${res.status}` }, { status: res.status, headers: { "Cache-Control": "no-store" } });
  } catch {
    return NextResponse.json({ error: "Could not reach Reactor." }, { status: 502 });
  }
}
