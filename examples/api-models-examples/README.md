# API model examples

One runnable Next.js app per model Reactor serves on the API. Each folder is a
self-contained project: clone it, add an API key, and it runs. Each also carries a
`skill/SKILL.md` — an agent skill that captures the design decisions, the gotchas,
and the patterns for growing the example into a product.

These moved here from `reactor-team/js-sdk`, which is being wound down. They are
the templates `npx create-reactor-app` scaffolds from, and the CLI will be
repointed at this folder.

## The examples

| Example                                 | Typed SDK                                                                                          | What it demonstrates                                                                                                                                                                                                                                                                                |
| --------------------------------------- | -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`happy-oyster/`](./happy-oyster)       | [`@reactor-models/happy-oyster`](https://www.npmjs.com/package/@reactor-models/happy-oyster)       | Interactive world model. Build a world from a prompt (or attach a permanent one), then travel it live: **Adventure** worlds you drive with WASD, **Directing** worlds you steer with text `instruct` plus pause/rewind. Mode-fixed sessions, authoritative `world_state` snapshot.                  |
| [`helios/`](./helios)                   | [`@reactor-models/helios`](https://www.npmjs.com/package/@reactor-models/helios)                   | Continuous prompt-driven video. Curated text and image scenes, mid-stream prompt hot-swap, atomic `setConditioning({ prompt, image })` for image-to-video, clip capture, design tokens from `@reactor-team/ui`.                                                                                     |
| [`lingbot/`](./lingbot)                 | [`@reactor-models/lingbot`](https://www.npmjs.com/package/@reactor-models/lingbot)                 | Interactive world model. Pick a starting image, drive the scene with WASD, layer curated dynamic events (rain, fog, …) as live prompt swaps, clip capture.                                                                                                                                          |
| [`lingbot-world-2/`](./lingbot-world-2) | [`@reactor-models/lingbot-world-2`](https://www.npmjs.com/package/@reactor-models/lingbot-world-2) | Interactive world model driven like a game. Two-axis WASD, per-latent `set_camera_pose` motion (mouse-look, roll, orbit, jump arcs, crouch dips), hold-key world events, a layered prompt workbench, attention-window and KV-cache knobs.                                                           |
| [`longlive-v2/`](./longlive-v2)         | [`@reactor-models/longlive-v2`](https://www.npmjs.com/package/@reactor-models/longlive-v2)         | Multi-shot **director's storyboard**. Compose shots and cuts on a chunk timeline, schedule beats ahead of time, then direct live. Surfaces the per-scene chunk budget and how cuts extend length.                                                                                                   |
| [`ltx2/`](./ltx2)                       | [`@reactor-models/ltx2`](https://www.npmjs.com/package/@reactor-models/ltx2)                       | Streaming **talking-head avatar**. Upload a face and a script; the model generates voice and lip-synced video together and streams both. The take in flight is frozen while the session stays editable — mid-run edits queue for the next take. Server-authoritative `valid_commands`, TTFF timing. |
| [`sana-streaming/`](./sana-streaming)   | [`@reactor-models/sana-streaming`](https://www.npmjs.com/package/@reactor-models/sana-streaming)   | Streaming **video-to-video editor**. Live webcam transform via a manual `camera` publish, file-clip editing with side-by-side compare, mid-stream re-prompting, seed control.                                                                                                                       |
| [`x2/`](./x2)                           | [`@reactor-models/x2`](https://www.npmjs.com/package/@reactor-models/x2)                           | Streaming **video-to-video editor** on XMAX X2. Webcam, file-clip or still-image sources on one `source` track, side-by-side compare, reference-image conditioning, drag-to-steer pointer on the output, keep-backlog toggle.                                                                       |

## Running one

Each folder is a standalone pnpm project and does **not** join a workspace, so
copying it out works exactly the way the scaffolding CLI does:

```bash
cd examples/api-models-examples/helios
cp .env.example .env.local
# add REACTOR_API_KEY=rk_...

pnpm install
pnpm dev
```

API keys come from [reactor.inc/account/api-keys](https://www.reactor.inc/account/api-keys).

## How auth works in every example

The same shape everywhere, and the only shape these examples document:

- The `rk_` **API key stays server-side**. It is read by
  `app/api/reactor/token/route.ts` and never sent to the browser.
- That route mints a **short-lived, session-scoped JWT** via Reactor's `/tokens`
  endpoint, pinned to the example's model through `authorization_details` with a
  bounded session budget. The JWT is the only credential the browser holds, and it
  can only operate sessions it created itself.
- The client hands a **resolver** to `<ModelProvider jwtToken={fetchToken}>`. The
  SDK calls it before every authenticated request, so no hop 401s on an aged-out
  token.
- The resolver **memoizes the token** in module scope until shortly before expiry
  and fetches `no-store`. This is not an optimization: a session-scoped token may
  only operate the sessions it created, so every hop of one session must present
  the same JWT, and a browser HTTP cache cannot promise that.

Each `skill/SKILL.md` explains the failure mode if you break that last rule.

## What changed with `@reactor-team/js-sdk` 3.x

Every example targets 3.x. One behavioural change matters more than the rest, and
no part of getting it wrong is a compile error:

**A command's result belongs on the awaited call, not on a subscription.** When a
model's handler answers with a message, that answer is **addressed**: the runtime
sends it to the one connection whose command earned it, correlated by request id.
On that connection it arrives twice over — it resolves the awaited call, and the
same frame also raises the `message` event — so a typed hook for an answer does
fire. It just fires for *any* call of that command on this connection, with no way
to say which, and it never fires on a second client in the session at all.

```tsx
// ⚠️ fires, but for any setImage on this connection — and never on a second
// client watching the same session.
useHeliosImageAccepted((msg) => setDimensions(msg));

// ✅ tied to this call, and it tells you the handler finished
const accepted = await setImage({ image: ref });
if (accepted) setDimensions(accepted);
```

The corollary for multi-client sessions: anything every client has to agree on
must **broadcast**, which is what a model's `state` snapshot is for. Never build
shared UI state out of answers.

Two more corollaries:

- **Awaiting a command that answers with nothing is still a barrier.** The runtime
  acknowledges every correlated command once its handler has run, so a resolved
  `await` means the handler finished. Delete any sleep that existed to "give the
  model time".
- **Commands never reject.** `try/catch` is not how you detect failure. Where the
  reason turns up is per-model, and the three shapes are genuinely different —
  see below.

Which commands answer, which broadcast, and which answer with nothing is
per-model, so each `skill/SKILL.md` carries the table for its own model. Check that
table before porting a pattern between folders — how one model reports a refusal is
not evidence for another.

**How a refusal reaches you, by model.** Two shapes, and they need different
client code:

| Models | A refused command… |
| --- | --- |
| `helios`, `lingbot`, `lingbot-world-2`, `longlive-v2`, `ltx2`, `sana-streaming` | resolves the call `undefined` and broadcasts `command_error` to every connection. Surface that hook. |
| `x2` | declares no `command_error` at all from its 1.0.0 release, so there is no hook. The refusal is the command's own error reply, which the SDK records on `lastError` and raises on the `error` event. |

A refusal is never the awaited value, so `undefined` — not a message with an
error-ish shape — is what a call site tests for.

## Conventions

- Standalone Next.js 15 + React 19 + Tailwind v4 + TypeScript.
- `@reactor-team/js-sdk` `^3.0.0`, plus one `@reactor-models/*` typed SDK per
  folder, generated from the model's published schema.
- One model per folder. The folder name is the model identifier the scaffolding
  CLI takes as `--model <name>`.
- Read a folder's `skill/SKILL.md` before changing it. It is where the reasoning
  behind the code lives.
