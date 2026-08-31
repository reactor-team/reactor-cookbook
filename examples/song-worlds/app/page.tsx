import { SongWorldApp } from "./SongWorldApp";

// Server Component: confirms which keys actually load, then hands only
// BOOLEANS to the client so it can pick real vs mock paths per subsystem:
//   REACTOR_API_KEY missing  → mock world renderer (forced)
//   NVIDIA_NEMO_KEY missing  → composition disabled warning (no fallback)
// The keys themselves never reach the client — only booleans do. The
// Nemotron call runs exclusively in POST /api/compose (server route).
//
// The original create-reactor-app Helios demo is preserved at /demo.
export const dynamic = "force-dynamic";

export default function Page() {
  return (
    <SongWorldApp
      hasReactorKey={!!process.env.REACTOR_API_KEY}
      hasNemotronKey={!!process.env.NVIDIA_NEMO_KEY}
    />
  );
}
