# Recipe setup

- Follow `README.md` for setup and architecture.
- Keep `REACTOR_API_KEY` and `NVIDIA_NEMO_KEY` server-side in `.env.local`;
  never place either in a `NEXT_PUBLIC_` variable, never import
  `app/lib/server/**` from a client component, and never commit the file.
- If the user asks you to run the demo and no keys are available, finish
  the local setup and mock-mode composition/playback path, then ask for
  the missing key(s) before attempting a live Reactor session or a real
  composition call.
- Composing a song calls an LLM exactly once, before playback starts (see
  `app/api/compose/route.ts`). Never add a code path that calls it again
  during playback, and never let live audio or mouse input reach a
  `set_prompt` / `set_image` call — see the README's "Non-goals" section
  before touching `app/SongWorldApp.tsx` or `app/lib/world/worldSession.ts`.
- Run `npm run typecheck` and `npm run build` after code changes.
