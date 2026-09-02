# Workspace instructions

- Keep `REACTOR_API_KEY` and the optional `CEREBRAS_API_KEY` server-side in `.env.local`; never send raw keys to the browser or commit the file.
- Keep FastH3 command and event details isolated in `lib/h3-contract.ts`; mint scoped session JWTs in `app/api/reactor/token/route.ts`.
- The selected library image is the opening `starting_frame`. Every continuation uses Reactor's native `continue_from_clip_id`; do not extract final frames in the browser.
- FastH3 does not accept reference images yet. Mid-stream images and all props must become textual prompt context rather than uploads.
- A live prompt, image cue, or prop change applies only to the next uncommitted clip. It must not discard prefetched clips or mutate the clip already playing.

<!-- BEGIN:nextjs-agent-rules -->

# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` (resolved from this file's directory; in monorepos the `next` package may not be visible from the repo root) before writing any code. Heed deprecation notices.

This block is written and re-added by `next dev` — verify at `node_modules/next/dist/server/lib/generate-agent-files.js`. Removing it from a diff only re-creates the uncommitted change; committing it with your work keeps the tree clean.

<!-- END:nextjs-agent-rules -->
