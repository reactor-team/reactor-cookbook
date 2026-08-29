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
  - `examples/api-models-examples/` — the per-model reference frontends, one
    folder per model Reactor serves on the API. See below before editing one.
- `robotics/` — Python SDK integrations that drive already-served models.

## `examples/api-models-examples/`

One standalone Next.js app per API model, each carrying a `skill/SKILL.md` that
holds the reasoning behind its code. These are the templates
`npx create-reactor-app` scaffolds from, and the CLI is being repointed at this
folder — so a folder name here is a public identifier (`--model <name>`), not an
internal label. Renaming one breaks the CLI.

Three rules when you touch one:

- **The folder's `skill/SKILL.md` is part of the deliverable.** A change to how
  the example works that leaves the skill describing the old behaviour is
  incomplete — the skill is what agents read, so a stale one actively teaches the
  wrong thing.
- **Auth is an API key server-side, exchanged for a session-scoped JWT.** That is
  the only auth model these examples document. Do not add a third-party identity
  provider, and do not name one: readers copy verbatim, and a session-scoped
  Reactor token has the opposite lifetime rule from a refreshed identity token.
  The resolver must return a *stable* token for a session's whole life.
- **A command's result arrives on the awaited call, not on a subscription.**
  `@reactor-team/js-sdk` 3.x delivers a handler's answer correlated to the command
  that earned it, so it reaches only the connection that asked and never the
  message event. A listener waiting for an acceptance message compiles and never
  fires. Which commands answer is per-model; each skill carries its own table.

Verify a change the way a reader would:

```sh
cd examples/api-models-examples/<model>
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
