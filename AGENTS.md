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

## Two authoring shapes under `models/`

The runtime's current authoring surface is a `ReactorApp` written as two
halves: an application half the runtime drives one step at a time
(`process_input()`, `generate()`, `process_output()`) and a model half that
owns the weights and imports nothing from the runtime, meeting on a step input
and a step result. GUIDELINES.md states the rules; the reference for them is
the runtime's
[Application and Model](https://docs.reactor.inc/deploy/development/reactor-app/application-and-model)
page, its
[`application-model-isolation`](https://github.com/reactor-team/reactor-runtime/blob/main/skills/application-model-isolation/SKILL.md)
skill, and its
[`examples/waypoint`](https://github.com/reactor-team/reactor-runtime/tree/main/examples/waypoint).
In this repo, `models/lingbot-world-v1-fast/` is the model written that way:
`lingbot_world_v1.py` is the application half, `lingbot_world_v1_model.py`
the model half, and its `tests/` drive both without a GPU.

Most other folders are still the previous shape, a `ReactorPipeline` whose
`inference()` generator mixes the client checks, the model call, and the
messages in one loop. They run unchanged on the current runtime and are moved
one at a time. Three rules follow, for an agent editing or reviewing here:

- **New model code is a `ReactorApp` in two halves.** Do not write a new
  `inference()` loop, and do not copy one from a neighbouring folder.
- **A routine change to a previous-shape model is not a port.** A version
  bump, a dependency roll, a prompt or description change: make it on the
  shape the folder has. A change to how the folder's loop decides a step,
  what the model receives, or what a step emits is the moment to port it,
  as its own PR.
- **A port is a split, not a translation, and it moves no client surface.**
  The runtime's
  [`porting-to-reactor-app`](https://github.com/reactor-team/reactor-runtime/blob/main/skills/porting-to-reactor-app/SKILL.md)
  skill is the method; this repo carries no copy of it. Render
  `python -m reactor_runtime.schema` before and after and diff: commands,
  messages, tracks, and descriptions must be byte-identical. Review a port
  against the split: nothing from `reactor_runtime` in the model half; the
  application reads the model only through the result; `ApplicationError`
  only in `process_input()`, the model's own exceptions out of `generate()`;
  every message an explicit `await self.send()` in `process_output()`;
  `reset()` without arguments; the step time read from `outcome.elapsed`,
  not measured again.

## Every `reactor.yaml` is on the latest spec format

Readers copy a manifest from here as their starting point, so every model
manifest declares the current `reactor` CLI spec format, `$schema: reactor/v2`.
An older header (`$schema: reactor/v1`, the `apiVersion`/`kind` header, or no
header at all) still loads, but the CLI prints a deprecation warning, and a
later CLI will refuse it. When the CLI moves to a newer format, all models here
move with it in one change.

- `recording:` nests under `runtime:`. A top-level `recording:` block is the
  legacy placement, and under `reactor/v2` the CLI rejects it.
- A deployment plan nests `instances:` under `deployment:`.
- `reactor validate` in the model folder prints `✓ reactor.yaml` with no
  warning. Review asks for the switch on any PR that adds or edits a
  manifest on an older format.

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
