# Robotics example helpers

`reactor_robotics` is the helper directory in Reactor’s public cookbook. It is
installed from this repository, separately from the published
[Python SDK](https://docs.reactor.inc/sdk-reference/python/reactor). It is not a
standalone published robotics SDK or a robot driver.

## Install

```bash
git clone https://github.com/reactor-team/reactor-cookbook.git
cd reactor-cookbook/robotics/sim/notebooks
uv sync --frozen --python 3.12
export REACTOR_API_KEY='your-api-key'
```

Run scripts with `uv run --frozen python <script>.py` from that directory.
`uv sync` installs the local project (distribution name `reactor-robotics-notebooks`,
import name `reactor_robotics`) and the Python SDK version recorded in `uv.lock`.
Installing `reactor-sdk` alone does not install these helpers.

## Session and camera observations

[`ReactorSession`](session.py) wraps `reactor_sdk.Reactor`. Call
`await session.connect(track_names, subscribe=(...))` to register message queues,
connect, and publish the selected tracks. Always call `await session.close()` in
`finally`, including after a failed connection. The SDK owns transport keepalive.

`session.set_frames({track_name: rgb_array, ...})` replaces the images on all
published tracks. Each image is an RGB `uint8` NumPy array shaped `(H, W, 3)`.
Unknown or missing track names raise an error. Supply the camera names and geometry
required by your model.

The helper repeats the last supplied images at its configured `fps`. All views
in one `set_frames` call share a capture timestamp; repeats retain that timestamp.
You can supply `capture_time_us` in the SDK's clock domain. Repetition does not
produce fresh measurements, and shared timestamps do not guarantee the model
consumed synchronized observations. For temporal-history policies, use the model's
simulator adapter and real samples rather than replaying held images.

## Messages and return values

| Call | Result |
| --- | --- |
| `await session.send(command, payload)` | The SDK's correlated message envelope `{"type": "...", "data": {...}}`, or `None` for an acknowledgement without a message. |
| `await session.next_message(message_type, timeout_s=...)` | The next queued message's inner `data` dictionary. |
| `session.drain(message_type)` | A list of already queued data dictionaries; does not wait. |

Subscribe to expected message types before sending commands so early replies are
retained. Replies to commands also reach the SDK message callback. The helper
queues only that callback path, once. Consume either the returned envelope or its
queued payload; do not execute both as separate actions. The first-action examples
consume queued messages, which also handles predictions sent independently of a
command reply.

SDK failures raise exceptions. Models can also emit `command_error` messages;
subscribe where the model declares that message. A bodyless acknowledgement does
not establish prediction completion or robot execution. `next_message` raises
`TimeoutError` when its deadline expires. Model-specific counters and message
fields are described in each [model guide](https://docs.reactor.inc/robotics/overview).

## Model-specific helpers

These clients add their model's protocol on top of `ReactorSession`; they are not
interchangeable by changing a slug. Read the matching implementation and guide.

| Helper | Guide |
| --- | --- |
| [`XwamClient`](xwam.py) | [X-WAM](../xwam_quickstart.md) |
| [`CosmosDroidClient`](cosmos_droid.py) | [Cosmos](../cosmos_droid_quickstart.md) |
| [`LingbotVaClient`](lingbot_va.py) | [LingBot-VA](../lingbot_va_quickstart.md) |
| [`DreamZeroClient`](dreamzero.py) | [DreamZero](../dreamzero_quickstart.md) |
| [`Xr1Robocasa365Client`](xr1_robocasa365.py) | [XR-1 RoboCasa365](../xr1_robocasa365_quickstart.md) |
| [`GrootN17Client`](groot_n17.py) | [GR00T example](../groot_n17_quickstart.md) |

For example, `XwamClient.predict(...)` returns an `XwamPrediction` containing
NumPy `actions` and `proprios`, the request's `step`, and timing information.
Other helpers have their own result types. Generic `ReactorSession` does not
normalize actions or execute them.

## Maintenance

The official docs follow cookbook `main`. Keep dependency manifests and lockfiles
updated together, run the robotics CI checks, and validate live changes against
the affected model before merging. Simulator and model-runtime versions are
separate dependencies; these hosted-inference helpers do not install a runtime.
