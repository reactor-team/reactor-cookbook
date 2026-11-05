# Robotics

Examples for hosted robot policies and action-conditioned simulation.

## Policy inference

These quickstarts cover DROID policies.
Reactor receives camera observations, measured robot state, and a task, then
returns predicted actions. Your client owns the simulator or robot controller.

| Model family | Embodiment | First actions |
| --- | --- | --- |
| Cosmos Nano | DROID | [Cosmos quickstart](https://docs.reactor.inc/robotics/cosmos/nano-policy-droid/quickstart) |
| DreamZero | DROID | [DreamZero quickstart](https://docs.reactor.inc/robotics/dreamzero/droid/quickstart) |
| FLUX Action | DROID | [FLUX Action quickstart](https://docs.reactor.inc/robotics/flux-action/droid/quickstart) |

Start with the [first-action scripts](./sim/notebooks): an API key is enough;
you do not need a local GPU, simulator, or model weights. Synthetic observations
verify connectivity and action shape, not task performance.

For a closed loop, see the [simulator integrations](./sim). The
[FLUX Action client](./flux3-action-droid) also supports checkpoint selection
and replaying your own NPZ observations. Use each model's guide for its camera
names, state layout, action units, and execution protocol.

## Action-conditioned simulation

[Cosmos3 Nano Sim Bimanual](./simulation/cosmos-bimanual) takes an AgiBot scene
image and controller poses/gripper values, and returns generated scene video.
Run the hosted rollout example to initialize, change controls, and reset a world.
This complements policy inference: you supply the actions rather than receive them.

[Cosmos3 Nano Sim Transfer](./simulation/cosmos-transfer) renders prepared
simulator controls as RGB observations. Try recorded-video rendering and reliable
incremental depth results, with synthetic input fixtures you can generate locally.
