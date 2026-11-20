# FastH3 livestream frontend

This Next.js application connects to the FastH3 Reactor model, plays its
synchronized WebRTC video and audio tracks, and exposes the controls needed for
an interactive livestream.

The interface provides:

- local and hosted Reactor connections
- initial and queued prompt submission
- start, stop, pause, resume, reset, canvas, seed, and clip-length controls
- automatic-story status and control
- Bilibili connection state, viewer backlog, and recent requests
- current playback metadata with `manual`, `bilibili`, or `ai` attribution
- the original viewer name and comment for Bilibili-driven clips

## Local development

Start the FastH3 model first, then configure the frontend:

```sh
cp .env.example .env
pnpm install --frozen-lockfile
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000). The default model URL is
`http://localhost:8080`. Set another local endpoint in `.env` when needed:

```dotenv
REACTOR_LOCAL_URL=http://localhost:18104
```

Connect, submit an initial prompt, and select **Start**. Later manual prompts
join the primary FIFO. Bilibili comments containing `!Prompt:` are rewritten
into complete English scenes and share that FIFO, while Infinite story supplies
fallback scenes from the latest seven-scene context.

## Production process

```sh
pnpm typecheck
pnpm build
pnpm start --hostname 0.0.0.0 --port 18105
```

## Hosted Reactor model

Set the following values on the Next.js server:

```dotenv
REACTOR_API_KEY=replace_with_your_reactor_api_key
REACTOR_MODEL_NAME=your-org/fasth3
REACTOR_API_URL=https://api.reactor.inc
```

The server exchanges the API key for a short-lived, model-scoped browser token.
The browser receives the scoped token through `/api/token` and connects through
`@reactor-team/js-sdk`.
