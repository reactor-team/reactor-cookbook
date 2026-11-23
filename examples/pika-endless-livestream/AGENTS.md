# Workspace instructions

An endless live channel on `reactor/pika-h3-flex`: a showrunner plans episodes, a director writes each clip's
prompt, and the page keeps clips queued ahead of the playhead on either Reactor API.

## Map

- `app/Livestream.tsx`: the page and both engines. The streaming engine is `send()` and `clipParams()`. The
  generation engine is `sendGeneration()`, `createGeneration()`, `watchGeneration()`, `checkGeneration()` and
  `playNextGeneration()`. The writer loop (`prepare()`, `takeBeat()`, `direct()`, `writeStory()`) is shared.
- `app/lib/story.ts`: the showrunner's prompt and schema (cast, props, places, beats) and `fillLabels()`.
- `app/lib/director.ts`: the director's prompt, in Pika's official T2VA, FL2VA and Ref2VA formats.
- `app/lib/check.ts`: the shot check's prompt and verdict schema.
- `app/lib/server/`: key lookup (`keys.ts`), the Claude and OpenAI writers (`writer.ts`), and the Reactor
  proxy (`reactor.ts`). Server only.
- `app/lib/examples.ts` and `public/examples/`: the starting shows and their Reactor-made pictures and voices.

## Rules

- Keys: `REACTOR_API_KEY`, `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` live in `.env.local` and are read only by
  route handlers under `app/api/`. A key pasted in the page travels in the `x-*-api-key` headers to those same
  handlers and overrides the environment. Never call Reactor, Anthropic or OpenAI from the browser with a key,
  and never commit `.env.local`.
- Session JWTs are minted in `app/api/token/route.ts` for `reactor/pika-h3-flex` only, one per page load (or
  per Reactor key). Keep that the only streaming auth path. `app/api/generations` always sends that model,
  whatever the page asks.
- Streaming: autoplay on and `set_flush_on_clip_end` off at connect. A refused `enqueue` resolves `undefined`
  and its reason arrives on the SDK's `error` event; never wait for a `command_error` message, and never show
  an error with `operation: "sendCommand"` as a connection failure.
- Generations: clips play strictly in queued order; a later clip never jumps an earlier one. A retake takes
  its first take's place in that order. The page plays downloaded blobs, never the presigned URL directly.
- Cast: the director names cast members and props only as `@label`; `fillLabels()` writes their looks in on
  the server. Do not ask the director to describe a cast member's look.
- Prompts follow Pika's prompt guidance (github.com/Pika-Labs/Pika-H3-Flex). The director's continuity rules
  are adapted from DreamVerse Harness (MIT, see `NOTICE`); keep that credit when editing them.

## Verify

```sh
pnpm install && pnpm build   # typecheck runs in the build
```

Don't run `pnpm build` in a folder where `pnpm dev` is serving: both write `.next`, and the dev server then
404s its own chunks until it restarts on a clean `.next`.
