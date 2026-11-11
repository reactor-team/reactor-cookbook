# Robotics first-action examples

Run the same examples shown in the official robotics quickstarts. They send
synthetic camera observations and measured-state-shaped inputs, then check the
returned actions. No local GPU, simulator, or model weights are required.
These checks establish connectivity and response shape, not task success.

## Setup

```bash
git clone https://github.com/reactor-team/reactor-cookbook.git
cd reactor-cookbook/robotics/sim/notebooks
uv sync --frozen --python 3.12
export REACTOR_API_KEY='your-api-key'
```

This installs the tested Python SDK from the lockfile and the
[`reactor_robotics` helper directory in Reactor's public cookbook](./reactor_robotics).
The helpers are separate from the SDK; their README explains installation and
return values. On Linux, the SDK wheel requires glibc 2.34 or newer.

## Choose a model

| Model | Endpoint | Run from this directory | Documentation |
| --- | --- | --- | --- |
| Cosmos Nano Policy DROID | `reactor/cosmos-nano-policy-droid` | `uv run --frozen python first_cosmos_actions.py` | [Quickstart](https://docs.reactor.inc/robotics/cosmos/nano-policy-droid/quickstart) |
| DreamZero DROID | `reactor/dreamzero` | `uv run --frozen python first_dreamzero_droid_actions.py` | [Quickstart](https://docs.reactor.inc/robotics/dreamzero/droid/quickstart) |
| FLUX Action DROID | `reactor/flux3-action-droid` | `uv run --frozen python first_flux_actions.py` | [Quickstart](https://docs.reactor.inc/robotics/flux-action/droid/quickstart) |
| Praxis-1 DROID | `reactor/praxis-1-droid` | `uv run --frozen python first_praxis_droid_actions.py` | [Quickstart](https://docs.reactor.inc/robotics/praxis/droid/quickstart) |
| Praxis-1 YAM (bimanual) | `reactor/praxis-1-yam` | `uv run --frozen python first_praxis_yam_actions.py` | [Quickstart](https://docs.reactor.inc/robotics/praxis/yam/quickstart) |

Source: [Cosmos](./first_cosmos_actions.py),
[DreamZero](./first_dreamzero_droid_actions.py),
[FLUX Action](./first_flux_actions.py),
[Praxis-1 DROID](./first_praxis_droid_actions.py),
[Praxis-1 YAM](./first_praxis_yam_actions.py).

The Praxis-1 examples request the first chunk (`step: 0`, 21 rows), echo
`set_executed_step`, and check the next chunk (`step: 1`, 24 rows).

The models use different state fields and inference protocols. Do not change
only the model slug in another policy's example. The model reference describes
camera names, action units, prediction horizon, and execution rules.

For recorded-observation replay, see the [Cosmos guide](./cosmos_droid_quickstart.md),
[DreamZero guide](./dreamzero_quickstart.md), and
[Praxis-1 guide](./praxis_quickstart.md) (DROID and YAM). The
[FLUX Action client](../../flux3-action-droid) supports synthetic or user-supplied
NPZ inputs and checkpoint selection.

## Session lifecycle

`ReactorSession` registers handlers before connecting, waits for readiness, and
publishes camera tracks. It repeats the last supplied image until replaced.
Shared capture timestamps do not guarantee synchronized model observations;
use fresh measurements and the model's simulator adapter for a control loop.

The SDK owns transport keepalive. Operation failures raise exceptions, while
model rejections can arrive as `command_error` messages. See the
[helper reference](./reactor_robotics#messages-and-return-values).
Always close a session in `finally`, including after a failed connection.

Create a key at [Reactor API keys](https://reactor.inc/account/api-keys) and keep
it in your shell. `REACTOR_API_URL` defaults to `https://api.reactor.inc`.
A successful installation does not guarantee available model capacity: HTTP
429 on session creation means no worker is free. Retry later or contact Reactor.

## Simulator integration

- Cosmos: [RoboLab gateway](../cosmos-droid).
- DreamZero: [RoboLab gateway](../dreamzero).
- FLUX Action: no closed-loop simulator adapter is included; use its
  [integration guide](https://docs.reactor.inc/robotics/flux-action/droid/integration)
  to connect your own controller.
- Praxis-1: no simulator adapter or physical-robot client is included; use the
  [DROID](https://docs.reactor.inc/robotics/praxis/droid/integration) or
  [YAM](https://docs.reactor.inc/robotics/praxis/yam/integration) integration
  guide to connect your own controller.

The RoboLab integrations require a separate simulator installation and an
RTX-capable GPU. Their wiring checks can run without the simulator and do not
measure task performance.

## Maintenance

The official documentation follows cookbook `main`. After changing a helper,
example, or guide, run:

```bash
uv run --frozen python -m unittest discover -s tests -v
```

Validate affected first-action examples against the hosted model before
merging. Keep dependency manifests and lockfiles aligned. Avoid sharing logs
containing credentials, session IDs, internal hostnames, or ICE candidate IPs.
