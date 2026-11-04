# Robotics

Quickstarts and simulator integrations for Reactor-hosted DROID policies.
Reactor receives camera observations, measured robot state, and a task, then
returns predicted actions. Your client owns the simulator or robot controller.

| Model family | Embodiment | First actions |
| --- | --- | --- |
| Cosmos Nano Policy | DROID | [Cosmos quickstart](https://docs.reactor.inc/robotics/cosmos/nano-policy-droid/quickstart) |
| DreamZero | DROID | [DreamZero quickstart](https://docs.reactor.inc/robotics/dreamzero/droid/quickstart) |
| FLUX Action | DROID | [FLUX Action quickstart](https://docs.reactor.inc/robotics/flux-action/droid/quickstart) |

Start with the [first-action scripts](./sim/notebooks): an API key is enough;
you do not need a local GPU, simulator, or model weights. Synthetic observations
verify connectivity and action shape, not task performance.

For a closed loop, see the [simulator integrations](./sim). The
[FLUX Action client](./flux3-action-droid) also supports checkpoint selection
and replaying your own NPZ observations. Use each model's guide for its camera
names, state layout, action units, and execution protocol.
