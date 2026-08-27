# Model authoring guidelines

The folders under [`models/`](./models) are the reference examples for
building models on the [Reactor Runtime](https://github.com/reactor-team/reactor-runtime).
People — and coding agents — copy their patterns verbatim, so every model
follows every rule below. A change that violates one of these rules in one
model will be replicated into the next ten models written against it; keep
them clean.

## Authoring shape

- Every model subclasses `ReactorPipeline`, declares `state: <Model>State`
  (an `InputState` subclass), and implements `inference()` as a generator.
- Commands are `@event`-decorated methods; session and connection lifecycle
  uses `@session_started`, `@session_ended`, `@connected`, `@disconnected`,
  and `@file_uploaded`. Nothing else is part of the command surface.
- `load()` does one-time startup (assets, weights, warmup); `@session_started`
  resets per-session state; `@session_ended` releases what the session held.

## Files and naming

A model folder is a `reactor` CLI workspace named after the model. Its Python
modules use the model name as a prefix and split along fixed seams:

| File | Owns |
| --- | --- |
| `<model>.py` | The `ReactorPipeline` subclass: commands, lifecycle, `inference()` |
| `<model>_types.py` | `InputState`, `Output` tracks, and every `ModelMessage` |
| `<model>_assets.py` | Config parsing plus source/checkpoint download and validation |
| `<model>_backend.py` | The in-process upstream model wrapper (GPU code) |
| `<model>_camera.py`, `<model>_images.py` | Optional camera-planning and image helpers |

Models that must isolate the upstream model in a subprocess (conflicting
dependencies, patched source) use `upstream_backend.py` + `worker.py` +
`download_snapshot.py`, a documented `*.patch`, and `<model>_config.py`
instead of `<model>_backend.py` / `<model>_assets.py`. Everything else about
them follows the same rules.

Also uniform across models: a `.dockerignore`, an `example_images/` folder
when the README shows sample inputs, `tests/test_<model>.py` when tests
exist, and **no** `__init__.py` — model modules are imported flat by
`reactor.yaml`'s `runtime.import`.

## Typed contracts

- Every `@event` handler declares a concrete `ModelMessage` return type (or
  `None`) and returns exactly that type. A command that changes shared state
  returns its specific result message and broadcasts a full `state_update`.
- `inference()` is annotated `AsyncGenerator[<Model>Output | None, None]`
  (sync generators: `Iterator[<Model>Output | None]`). Yield `None` to skip
  a turn. Never import private runtime names (anything underscore-prefixed)
  to type a signature.
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

- `reactor.yaml` orders `model:`, `runtime:` (with `recording:` nested under
  it), then `build:`. `model.version` is semver with a `v` prefix and bumps
  with every shipped change, sized to the schema impact — any command,
  message, or field change is at least a minor bump.
- `build.runtime_version` pins the current Reactor Runtime release; all
  models in this repo pin the same one.
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
