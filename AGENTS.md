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

## Verifying model changes

```sh
# From a model folder: render the client-facing contract without weights.
python -m reactor_runtime.schema --path . --out /tmp/schema.json

# Run the model's tests.
PYTHONPATH=. python -m pytest tests/ -q
```

Schema output is the client contract: diff it before and after a change and
confirm only the intended surface moved.
