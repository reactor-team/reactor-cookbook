# Deploying the demo

`npm run build` turns the demo into a static site in `dist/`: one `index.html`, hashed
assets with relative paths, the SDK's WebAssembly, the game covers and the seven games
under `local-games/`. It works from any static host, at the domain root or under a
sub-path. No server code runs and the build carries no credentials.

## Vercel CLI (recommended)

From `demo/`:

```sh
npm ci
npm run build
npx vercel deploy dist          # preview URL
npx vercel deploy dist --prod   # production URL
```

Deploying `dist/` uploads exactly what was built. The first run asks you to log in and to
name the project.

## Vercel dashboard (deploy from git)

Set **Root Directory** to `models/eden-live/demo`. The build needs the original
game sources: prepare the source-only checkout from the model README in the build
environment and set `EDEN_GAMES_ROOT` to its `realtime/games/` directory.
Weights and a GPU are not needed to build the demo. `vercel.json` supplies
`npm ci`, `npm run build` and output `dist`.

For a deployment without a custom source-checkout step, build locally with
`EDEN_BUNDLE` set and upload `dist/` using the Vercel CLI above.

Do not set `VITE_REACTOR_URL` for a public deploy: it only prefills the Local server URL and
makes Local the default.

## What a visitor does

1. Picks **DEV** (`https://api.rea.live`) or **PROD** (`https://api.reactor.inc`) in the
   header and types their own Reactor API key for that platform (`rk_…`). DEV and PROD keys
   are separate.
2. Leaves the model as `reactor/eden-live`. The platform serves its deployed release; the version
   it served appears next to the status once connected.
3. Clicks **Connect**, then plays as usual.

**Local** takes a model server URL instead of a key, for a server run with `reactor run` or
Docker. A page served over https can only reach an `https://` server or one on
`http://localhost` (for example through an SSH tunnel); for a plain-http server elsewhere,
run the demo locally with `npm run dev`.

## How the key is used

- The browser sends the key once per **Connect**, as the `Reactor-API-Key` header of
  `POST <platform>/tokens`, and nowhere else. The platform returns a short-lived JWT scoped
  to the chosen model (at most three sessions); the session runs on that JWT. Near expiry
  the page mints a replacement bound to the running session.
- The key stays in the tab's `sessionStorage` and is gone when the tab closes. **Remember
  on this device** moves it to `localStorage` instead. It never appears in a URL or a log.
- Reactor's docs recommend minting tokens on a server so that an app's own key never
  reaches a browser. That applies to a key the site owner embeds; here each visitor brings
  their own key, so do not put a team key into this site in any form.

## What the page reports

| Situation | Message |
| --- | --- |
| Wrong key, or a DEV key on PROD | `DEV rejected this API key (Invalid API key)…` |
| Model not visible to the key | `reactor/eden-live is not available to this API key on DEV…` |
| Model not deployed, or every instance busy | `No capacity: DEV has no free reactor/eden-live instance right now…` |
| Platform unreachable | `Could not reach DEV (https://api.rea.live)…` |
