# RLDX-1 with RTC

This recipe deploys RLDX-1 on Reactor. It contains the
Reactor adapter and the vendored RLDX-1 inference source used by the model image.

RLDX-1 is a vision-language-action policy: camera views, robot proprioception,
and a language instruction produce robot action chunks. It does not generate
video. Reactor Runtime 3.5's `process_input()` aligns and snapshots observations,
`generate()` predicts one action chunk, and `process_output()` sends
`action_prediction`. The client retains ownership of the RTC execution cursor.

`rldx1.py` owns observation alignment and client messages; `rldx1_model.py`
owns policy weights and episode memory. Frozen inputs and results carry aligned
observations, predicted actions, and the acknowledged episode identity.

The default configuration enables guided Real-Time Chunking (RTC):

- action horizon: 16 control steps;
- RTC delay: 5 control steps; and
- execution horizon: 8 control steps.

The released RoboCasa checkpoint supports guided RTC. The `trained` RTC mode
requires a checkpoint trained with RTC.

## Deploy from a new laptop

You need [Git](https://git-scm.com/downloads), a running Docker engine (for
example, [Docker Desktop](https://docs.docker.com/desktop/)), and a Reactor
account with deployment capacity in your selected region. The GPU runs on
Reactor; your laptop does not need a local NVIDIA GPU.

The source is Apache-2.0. The checkpoint uses the separate
[RLWRLD Model License](https://huggingface.co/RLWRLD/RLDX-1-FT-ROBOCASA/blob/main/LICENSE.md),
which includes use restrictions. Review that license before downloading or
deploying the weights.

### 1. Clone the cookbook

```bash
git clone https://github.com/reactor-team/reactor-cookbook.git
cd reactor-cookbook
```

### 2. Install the Reactor CLI and sign in

Install the Reactor CLI:

```bash
curl -fsSL https://reactor.inc/install | bash
```

Then check the installation and sign in:

```bash
reactor version
reactor auth login
```

Complete the browser sign-in with the account that will own this deployment.

### 3. Pull the public model image

Start Docker, then download the image:

```bash
docker pull reactortechnologies/rldx-1:1.0.1
```

Use **`1.0.1`** for both pulling and publishing.

### 4. Open the model workspace

```bash
cd models/rldx-1
```

The checked-in `reactor.yaml` points directly to the checkpoint:

```yaml
runtime:
  weights_path: "hf://RLWRLD/RLDX-1-FT-ROBOCASA"
```

The CLI records a pinned Hugging Face reference, and Reactor fetches the
checkpoint when starting the model.

The example `reactor.yaml` requests one instance in `us-west` by default:

```yaml
deployment:
  instances:
    - region: us-west
      count: 1
```

Using the manifest as-is requires capacity in `us-west`. To deploy in another
region, change the `region` value above to one with available deployment capacity
before running `reactor model deploy`.

### 5. Publish, deploy, and check status

Run these as separate commands from `models/rldx-1`:

```bash
reactor model publish --source reactortechnologies/rldx-1:1.0.1
reactor model deploy
reactor model status
```

`publish --source` pushes the downloaded image into your account without
rebuilding it and associates the Hugging Face weights with the release.
`deploy` activates that release and applies the instance plan. Model name and
release version come from `reactor.yaml`; the Docker image tag is the source
image version, not the account's release version.

Wait for the deployment to be ready before starting the client. Initial startup
includes fetching the checkpoint. Keep the account-qualified model name printed
by the CLI, such as `your-account/rldx-1`, for the client command below.

## Run the test client

The matching client publishes one synthetic frame on each of the three camera
tracks every control tick. Each tick uses one capture timestamp for all frames
and the proprioceptive state.

First install [uv](https://docs.astral.sh/uv/getting-started/installation/) with
[Homebrew](https://brew.sh/) if it is not already installed:

```bash
brew install uv
uv --version
```

From `models/rldx-1`, enter the client directory, install its dependencies, and
activate its virtual environment:

```bash
cd client
uv sync --python 3.12 --upgrade-package reactor-sdk
source .venv/bin/activate
```

`uv sync` creates `.venv`, downloads Python 3.12 if needed, and installs the
client dependencies from `pyproject.toml`, including the Reactor SDK and NumPy.

Then run the client in the same terminal:

```bash
export REACTOR_API_KEY=rk_your_key_here
python main.py --model <account-slug>/rldx-1 --duration 60
```

Replace `<account-slug>` with the account slug printed by Reactor when the
model is published, and use an API key from the same Reactor account. CLI
sign-in does not set `REACTOR_API_KEY` for the Python client. In a new terminal,
return to `models/rldx-1/client` and run `source .venv/bin/activate` again before
using `python main.py`.

The summary reports RTC request-to-response latency, observation age, and
cross-view capture skew at p50 and p99. The returned actions are model outputs
for synthetic inputs and must not be sent to a robot.

The client also prints the current WebRTC RTT every 10 seconds using
`reactor-sdk >= 1.6.0`. Average RTT appears once in the final summary.
See the [client guide](client/README.md#webrtc-stats)
for metric meanings and `--stats-interval`.

## Source provenance

The `rldx/` directory is the inference subset of
[`RLWRLD/RLDX-1`](https://github.com/RLWRLD/RLDX-1) at commit
`ecbfaf80cd031dcc892186ed30465de3591047e6`. Reactor-specific RTC changes are
documented in `rldx/PROVENANCE.md`.
