# Model authoring guidelines

The folders under [`models/`](./models) are the reference examples for
building models on the [Reactor Runtime](https://github.com/reactor-team/reactor-runtime).
People — and coding agents — copy their patterns verbatim, so every model
follows every rule below. A change that violates one of these rules in one
model will be replicated into the next ten models written against it; keep
them clean.

## Authoring shape

A model is a `ReactorApp` written as two halves, on Reactor Runtime 3.5.0 or
later. The runtime drives the class one **step** at a time, and each step is
three calls the author writes: `process_input()`, `generate()`,
`process_output()`. The rules are the runtime's
[Application and Model](https://docs.reactor.inc/deploy/development/reactor-app/application-and-model)
page and its
[`application-model-isolation`](https://github.com/reactor-team/reactor-runtime/blob/main/skills/application-model-isolation/SKILL.md)
skill; `models/lingbot-world-v1-fast/` is the example in this repo.

- **The application half** is the `ReactorApp` subclass in `<model>.py`. It
  declares `state: <Model>State` (an `InputState` subclass), owns every
  `@event` command and lifecycle hook, and writes the three step hooks.
  `process_input()` reads `self.state` and the inbound tracks, refuses a
  step with `ApplicationError("reason")` when the client is not ready, and
  returns the step input. `generate()` is one line forwarding that input to
  the model half. `process_output()` receives the `StepOutcome`, maps the
  result onto the `Output` tracks, sends every message with an explicit
  `await self.send()`, and either recovers from a model error it expects or
  re-raises it. Handlers write state, decode uploads, flush playout, and
  call the model half's `reset()`; they never run inference.
- **The model half** is a plain class in `<model>_model.py` with `load()`,
  `generate(input)`, and `reset()`. It imports nothing from
  `reactor_runtime` and knows nothing about clients, tracks, or commands. It
  owns the weights, the caches, and its own chunk count, and it raises its
  own exception types when it cannot step from the state it holds. `reset()`
  takes no arguments.
- **Its own process, or one per GPU**, is the runtime's `DistributedRunner`.
  The application wraps the model half in it in `load()`
  (`self._engine = DistributedRunner(<Model>Model, world_size=N, load_kwargs={...})`),
  and `generate()` stays one line. The runner owns the worker processes, the
  transport, and their teardown; the model half spawns nothing and takes
  only picklable arguments in `load()`. Only rank 0's result reaches the
  application. `models/lingbot-world-v1-fast/` shows it.
- **The inner contract** is two frozen dataclasses the model half owns,
  `<Model>Input` and `<Model>Result`, made of plain values and CPU
  `numpy` arrays. They are the only things that cross between the halves.
  The application reads the model only through the result, never through an
  attribute; a fresh world or a new reference travels as an id on the
  input, with the image only until the result reports that id applied.
- Commands are `@event`-decorated methods; session and connection lifecycle
  uses `@session_started`, `@session_ended`, `@connected`, `@disconnected`,
  and `@file_uploaded`. Nothing else is part of the command surface.
- `load()` constructs the model half and loads it once; `@session_started`
  resets per-session state; `@session_ended` calls the model half's
  `reset()`. The runtime never resets a model itself.
- The step time is `outcome.elapsed`, measured by the runtime around
  `generate()`; do not time it again. Playout follows it unless the class
  pins `fps`.

Folders still written as a `ReactorPipeline` with an `inference()` generator
are the previous shape. They keep running (the runtime imports the old names
as deprecated aliases) and are moved onto the shape above as they change; do
not add a new `inference()` loop, and do not copy one into a new model.

## Files and naming

A model folder is a `reactor` CLI workspace named after the model. Its Python
modules use the model name as a prefix and split along fixed seams:

| File | Owns |
| --- | --- |
| `<model>.py` | The application half: the `ReactorApp` subclass, commands, lifecycle, the three step hooks |
| `<model>_model.py` | The model half: `load()` / `generate()` / `reset()`, `<Model>Input`, `<Model>Result`, its own errors; no runtime import |
| `<model>_types.py` | `InputState`, `Output` tracks, and every `ModelMessage` |
| `<model>_assets.py` | Config parsing plus source/checkpoint download and validation |
| `<model>_backend.py` | The in-process upstream model wrapper (GPU code) the model half calls |
| `<model>_camera.py`, `<model>_images.py` | Optional camera-planning and image helpers, used by the application half |

Models whose upstream cannot share the runtime's Python environment
(conflicting dependencies) run it in a separate interpreter with
`upstream_backend.py` + `worker.py`. A model that only needs its own process
or several GPUs uses `DistributedRunner` and keeps `<model>_backend.py`.
Either kind may carry `download_snapshot.py`, a documented `*.patch`, and
`<model>_config.py` in place of `<model>_assets.py`. Everything else about
them follows the same rules.

Also uniform across models: a `.dockerignore`, an `example_images/` folder
when the README shows sample inputs, `tests/test_<model>.py` when tests
exist, and **no** `__init__.py` — model modules are imported flat by
`reactor.yaml`'s `runtime.import`.

## Typed contracts

- Every `@event` handler declares a concrete `ModelMessage` return type (or
  `None`) and returns exactly that type. A command that changes shared state
  returns its specific result message and broadcasts a full `state_update`.
- `process_input()` is annotated to return `<Model>Input`, `generate()` to
  take it and return `<Model>Result`, and `process_output()` to take a
  `StepOutcome` and return `<Model>Output | None`. Never import private
  runtime names (anything underscore-prefixed) to type a signature.
- Do not annotate `output:` on the model class — `self.output` is the
  runtime's `OutputStream`; the `Output` subclass only declares tracks.

## Moderation marks

Every field that carries free-form client content sets `moderate=True` on
its `InputField`: free-text strings (prompts) and every `UploadedFile`
parameter. Enum-constrained (`choices=`) and bounded numeric fields never
carry the mark — it does nothing for them.

## No ghost surface

- No undecorated command-shaped methods. If a command is not exposed with
  `@event`, its handler, its message types, and its state do not exist.
  There is no "keep it for later" — git history is the archive.
- Defining a `ModelMessage` subclass publishes it in the schema as a model
  message. A message class no live code sends is a schema lie; delete it.
- No write-only attributes: every `self._x` assigned must be read somewhere.
- Description strings, docstrings, and READMEs mention only commands a
  client can send and messages the model actually emits, by wire name in
  backticks. Never describe internals (caches, latents, config keys) the
  client cannot observe.

## Code style

- Comments and docstrings describe the end state — no iteration narration
  ("previously", "no longer", "now we"), no narrating what the code visibly
  does.
- No `hasattr`/`getattr` guards on the model's own attributes; initialize
  them in `__init__` and trust them. Guards on upstream objects that vary by
  version are fine when the comment says why.
- `self.state` is `None` only between sessions. Guard it only on paths that
  can actually run between sessions, and use the state consistently within
  one method — never check then use unguarded.
- Free functions over unused flexibility: no parameters, config fields, or
  constants that nothing reads.

## Manifest and dependencies

- `reactor.yaml` opens with `$schema: reactor/v2`, the current spec format
  (see AGENTS.md), and orders `model:`, `runtime:` (with `recording:` nested
  under it), then `build:`. `model.version` is semver with a `v` prefix and bumps
  with every shipped change, sized to the schema impact — any command,
  message, or field change is at least a minor bump.
- `build.runtime_version` pins the current Reactor Runtime release, 3.5.0
  or later for a `ReactorApp` and 3.6.0 or later for one that uses
  `DistributedRunner`. Models still on the previous shape pin the
  release they were verified on until they move; a move bumps the pin and
  `model.version` together.
- `requirements.txt` starts with the shared two-line header explaining that
  `build.runtime_version` owns the Runtime release, then lists only
  dependencies the model's own code imports.

## Verifying a change

From the model folder:

```sh
python -m reactor_runtime.schema --path . --out /tmp/schema.json  # contract renders
PYTHONPATH=. python -m pytest tests/ -q                           # tests pass
```

Diff the rendered schema before and after your change: only the surface you
intended to change may move.

Tests drive the application half through its three hooks with a fake model
half, and the model half with the GPU work faked, so they run without a GPU,
the weights, or the upstream dependencies. At least: a refused step never
reaches the model; `generate()` reads only its input; a fresh world carries
its anchor once; a model error out of `generate()` reaches `process_output()`
and does not count as a completed step.
