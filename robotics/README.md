# Robotics

Quickstarts and integrations for Reactor-hosted robot policies.

- [FLUX 0.3.0 quickstart](./flux3-action-droid): choose one of six checkpoints per session, send three camera views and measured state, and receive `(32, 8)` action predictions. Use synthetic inputs or replay an NPZ; no simulator is required.
- [`sim/`](./sim): policy quickstarts and closed-loop simulator integrations.

Each example documents its policy contract, setup, and validation steps. Simulator
integrations also explain how to connect the simulator. Hosted model access is
required even when the client code is public.
