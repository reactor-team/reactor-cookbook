# DROID policy examples

Run hosted policies through Reactor's Python SDK. Start with a first-action
script before installing a simulator.

## Quickstarts

The [quickstart directory](./notebooks) contains the runnable examples shown in
the official documentation. They use synthetic observations and require an API
key, Python, and [uv](https://docs.astral.sh/uv/). They do not execute robot motion.

| Model | Hosted endpoint | First-action script | Simulator adapter |
| --- | --- | --- | --- |
| Cosmos Nano Policy DROID | `reactor/cosmos-nano-policy-droid` | [Cosmos](./notebooks/first_cosmos_actions.py) | [RoboLab gateway](./cosmos-droid) |
| DreamZero DROID | `reactor/dreamzero` | [DreamZero](./notebooks/first_dreamzero_droid_actions.py) | [RoboLab gateway](./dreamzero) |
| FLUX Action DROID | `reactor/flux3-action-droid` | [FLUX Action](./notebooks/first_flux_actions.py) | No closed-loop adapter included |
| Praxis-1 DROID | `reactor/praxis-1-droid` | [Praxis-1 DROID](./notebooks/first_praxis_droid_actions.py) | No closed-loop adapter included |
| Praxis-1 YAM | `reactor/praxis-1-yam` | [Praxis-1 YAM](./notebooks/first_praxis_yam_actions.py) | No closed-loop adapter included |

Except for Praxis-1 YAM, these policies share the DROID embodiment, but their
input fields and execution protocols differ. Follow the [model-specific documentation](https://docs.reactor.inc/robotics/overview).
The [`reactor_robotics` helper directory in Reactor's public cookbook](./notebooks/reactor_robotics)
documents the shared session wrapper; it is separate from the Python SDK.

## Simulator integrations

The Cosmos and DreamZero gateways connect a locally running RoboLab/Isaac Sim
environment to the hosted model. Each gateway README includes its simulator
setup, run command, and offline wiring checks. RoboLab needs an RTX-capable GPU;
installing the gateway alone does not install the simulator.

FLUX Action has a [client for synthetic or NPZ inputs](../flux3-action-droid).
Connecting it to a simulator requires your own adapter; receiving an action
chunk does not establish closed-loop task success. The same applies to
Praxis-1: use its [DROID](https://docs.reactor.inc/robotics/praxis/droid/integration)
or [YAM](https://docs.reactor.inc/robotics/praxis/yam/integration) integration guide.

## Setup

Each directory has its own dependency manifest and lockfile. Follow the selected
example's setup commands rather than combining simulator environments.

Create a key at [Reactor API keys](https://reactor.inc/account/api-keys), then set
`REACTOR_API_KEY` in your shell. Never commit it. Hosted inference requires
available capacity; HTTP 429 during session creation means no worker is
available. Close sessions when finished.
