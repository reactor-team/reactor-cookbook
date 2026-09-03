# Agent instructions for reactor-cookbook

This is the public Reactor Cookbook (Apache-2.0): runnable examples,
deployable models, and robotics integrations for the Reactor platform.
Everything here is public and is copied verbatim by developers and coding
agents — treat every file as product surface.

**Before touching anything under `models/`, read [GUIDELINES.md](./GUIDELINES.md).**
It is the binding standard for model code: authoring shape, file naming,
typed contracts, moderation marks, ghost-surface rules, manifest layout, and
how to verify a change. A model edit that violates it is wrong even if it
works.

## Contribution policy

- Never commit, push, or open PRs without explicit permission from a human
  maintainer in the current conversation.
- Keep each example self-contained; a change to one model must not reach
  into another model's folder.

## Layout

- `models/` — deployable models; each folder is a `reactor` CLI workspace
  governed by GUIDELINES.md.
- `examples/` — complete applications built on hosted Reactor models.
- `robotics/` — Python SDK integrations that drive already-served models.

## `examples/`

Complete applications built on models Reactor already serves. The per-model
reference frontends that `npx create-reactor-app` scaffolds from are **not**
here — they live beside that CLI in
[reactor-team/create-reactor-app](https://github.com/reactor-team/create-reactor-app),
because a folder name there is the public `--model` identifier. A new
per-model reference frontend belongs in that repository, under the
[`scaffold-model-example`](https://github.com/reactor-team/ai-skills/blob/main/workflow/scaffold-model-example.md)
standard. A complete application or demo belongs here.

Two rules when you touch a frontend here, because both are things a reader
copies verbatim:

- **Auth is an API key server-side, exchanged for a session-scoped JWT.** That is
  the only auth model these examples document. Do not add a third-party identity
  provider, and do not name one: readers copy verbatim, and a session-scoped
  Reactor token has the opposite lifetime rule from a refreshed identity token.
  The resolver must return a *stable* token for a session's whole life.
- **A command's result belongs on the awaited call, not on a subscription.**
  `@reactor-team/js-sdk` 3.x delivers a handler's answer addressed to the
  connection that asked, correlated by request id. There it both resolves the
  awaited call and raises that connection's `message` event — so an acceptance
  listener does fire, but for any call of that command and never on a second
  client in the session. Read answers off the await; keep subscriptions for what
  the model broadcasts. Which commands answer, and where a refusal surfaces, is
  per-model — each skill carries its own table.

Verify a change the way a reader would:

```sh
cd examples/<name>
pnpm install && pnpm build   # `tsc --noEmit` runs as part of next build
```

## Verifying model changes

```sh
# From a model folder: render the client-facing contract without weights.
python -m reactor_runtime.schema --path . --out /tmp/schema.json

# Run the model's tests.
PYTHONPATH=. python -m pytest tests/ -q
```

Schema output is the client contract: diff it before and after a change and
confirm only the intended surface moved.
