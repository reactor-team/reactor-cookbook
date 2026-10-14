# Workspace instructions

- `REACTOR_API_KEY` and the optional `ANTHROPIC_API_KEY` live in `.env.local` and are read only by route handlers under `app/api/`; never send a raw key to the browser or commit the file.
- Scoped session JWTs are minted in `app/api/token/route.ts` for `reactor/fast-h3-omni` only. Keep that the only auth path.
- The stories and their prompts live in `lib/storyboard.ts`. Prompts follow the model's prompt guide: subject lines binding `<Subject N>` to `<Picture N>`, one "In one shot," beat, then Camera, Dialogue, Soundscape, Music.
- Reference pictures are uploaded once per session and cached by asset id; every shot passes its pictures in the order its `refs` lists them, which is the `<Picture N>` order in the prompt.
- Generate storyboard uses the model's playout queue with `set_autoplay` and `set_flush_on_clip_end: false`; the app records the WebRTC stream, it does not splice video in the browser.
- A continuation keeps the previous shot's binding lines and pictures, states what carries over, and sends `continue_from_clip_id` with `continue_mode: "frame"`, falling back to `latent` if the model refuses.
- A refused command fails its own call. `command()` in `app/page.tsx` goes through `sendModelCommand` in `lib/command.ts`, which throws the model's reason; never listen for a `command_error` message, and never show an `error` event with `operation: "sendCommand"` as a connection error.

<!-- BEGIN:nextjs-agent-rules -->

# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` (resolved from this file's directory; in monorepos the `next` package may not be visible from the repo root) before writing any code. Heed deprecation notices.

This block is written and re-added by `next dev` — verify at `node_modules/next/dist/server/lib/generate-agent-files.js`. Removing it from a diff only re-creates the uncommitted change; committing it with your work keeps the tree clean.

<!-- END:nextjs-agent-rules -->
