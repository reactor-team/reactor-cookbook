<!-- BEGIN:nextjs-agent-rules -->

# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` (resolved from this file's directory; in monorepos the `next` package may not be visible from the repo root) before writing any code. Heed deprecation notices.

This block is written and re-added by `next dev` — verify at `node_modules/next/dist/server/lib/generate-agent-files.js`. Removing it from a diff only re-creates the uncommitted change; committing it with your work keeps the tree clean.

<!-- END:nextjs-agent-rules -->

# Recipe setup

- Follow `README.md` for setup and controls.
- Keep `REACTOR_API_KEY` server-side in `.env.local`; never place it in a
  `NEXT_PUBLIC_` variable or commit the file.
- If the user asks you to run the demo and no key is available, finish the
  local setup, then ask for the key before attempting a live world session.
- Run `npm run typecheck` and `npm run build` after code changes.
