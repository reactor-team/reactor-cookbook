# libero-sim

Drives a real [LIBERO](https://libero-project.github.io/) (robosuite/MuJoCo)
environment closed-loop, lock-step with a Reactor-served `lingbot-va` over
the [reactor-sdk](https://pypi.org/project/reactor-sdk/) transport.

## Layout

```
libero_sim/
  contract.py   wire schema: the executed-action echo + chunk decoding
  env.py        LIBERO/robosuite env wrapper (reset, step, frames)
  loop.py       RolloutState (thread-safe hand-off) + SimDriver (steps the env)
  tracks.py     CameraTrack: publishes one video frame per env render
  bridge.py     Bridge / BridgeThread: the reactor-sdk integration
  main.py       entrypoint: wires everything together and runs the rollout
check_wiring.py network-free smoke test: feeds a synthetic chunk through
                RolloutState and checks the echo it produces
```

## Setup

Requires Python 3.10 exactly (robosuite/bddl/`torch<2.6` pin the ceiling; see
`pyproject.toml`). With [uv](https://docs.astral.sh/uv/), which fetches that
interpreter itself rather than assuming one is already on PATH:

```
cd libero
uv sync --python 3.10
```

LIBERO isn't on PyPI, so clone and install it from source:

```
git clone https://github.com/Lifelong-Robot-Learning/LIBERO vendor/LIBERO
touch vendor/LIBERO/libero/__init__.py   # missing in the upstream tree
uv pip install --no-deps -e vendor/LIBERO
```

LIBERO prompts interactively for a benchmark config path on first import if
one isn't already set. Pre-seed it to avoid that:

```
mkdir -p .libero
cat > .libero/config.yaml <<EOF
assets: $PWD/vendor/LIBERO/libero/libero/./assets
bddl_files: $PWD/vendor/LIBERO/libero/libero/./bddl_files
benchmark_root: $PWD/vendor/LIBERO/libero/libero
datasets: $PWD/vendor/LIBERO/libero/libero/../datasets
init_states: $PWD/vendor/LIBERO/libero/libero/./init_files
EOF
export LIBERO_CONFIG_PATH="$PWD/.libero"
```

## Run

```
export REACTOR_API_KEY=...   # create one at https://reactor.inc/account/api-keys
uv run python -m libero_sim.main --task-id 0
```

Ctrl-C to stop. Useful flags:

- `--suite` / `--task-id` / `--init-state-id`: which LIBERO benchmark, task,
  and initial state to run (default `libero_10` task 0, init state 0)
- `--no-flip`: publish raw (upside-down) frames, for diagnosing camera
  orientation only
- `--echo-delay`: settle window before echoing the executed actions (see the
  timing caveat in `bridge.py`)
- `--record out.mp4`: write the published camera views, side by side, to a
  video file, as proof the rollout is actually running rather than just
  connected

## Smoke test

```
uv run python check_wiring.py
```

Feeds a synthetic action chunk straight into `RolloutState` and checks that
the env executes it and produces a correctly shaped echo. Use it to confirm
the LIBERO install and env wrapper work before touching the network at all.

## Gotchas

- **Image orientation.** MuJoCo's offscreen renderer returns bottom-up
  arrays; `env.py`'s `get_frames()` applies a vertical flip (`img[::-1]`) to
  match the training frames. This is deliberately *not* the 180-degree
  rotation (`img[::-1, ::-1]`) some other LIBERO harnesses use; the two
  differ by a horizontal mirror. Getting it wrong doesn't raise, it just
  mirrors every observation the policy sees.
- **Lock-step, not open-loop.** The env only steps once the client has
  echoed back what it executed. See the comment at the top of `loop.py`.
- **Main-thread env.** The env is constructed and stepped from the main
  thread; the reactor bridge gets its own thread instead. On macOS, creating
  or using MuJoCo's offscreen GL context off the thread that created it
  segfaults the process rather than raising.
