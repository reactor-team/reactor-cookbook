# Robotics: embodiments

One folder per robot. Each folder describes the robot as a Reactor client sees
it (camera views, state layout, action layout, and how to drive the hardware)
and holds a client for each hosted policy for that robot.

- [`yam/`](./yam): the bimanual I2RT YAM, with a client for
  [`reactor/rho-yam-box`](./yam/rho-yam-box) (Microsoft Rho).

Policy quickstarts organized by simulator are in [`../sim/`](../sim).
