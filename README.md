<div align="center">

<img src="assets/banner.png" alt="Reactor Cookbook" width="100%" />

**Practical, runnable examples for building with the Reactor platform.**

[🌐 Reactor](https://reactor.inc) · [🛰️ reactor-webrtc](https://github.com/reactor-team/reactor-webrtc) · [⚙️ reactor-runtime](https://github.com/reactor-team/reactor-runtime)

</div>

---

Short, self-contained code you copy, run, and adapt — not reference
documentation. For API references and guides, see the
[Reactor docs](https://docs.reactor.inc/overview).

## Structure

The three top-level folders cover distinct ways to build with Reactor.

- [`examples/`](./examples) holds complete applications and demos built on
  hosted Reactor models. Each example is a top-level, self-contained project
  you can copy, run, and adapt.
- [`models/`](./models) holds models you deploy. Each folder is a `reactor`
  workspace you build and run with the CLI, and everything that deployment
  needs lives inside it. [Build your own model](https://docs.reactor.inc/deploy/overview)
  covers the workspace they follow.
- [`robotics/`](./robotics) holds code that drives a model someone else is
  already serving: policy quickstarts and closed-loop simulator integrations
  built on the Python SDK.

Each example has a README explaining what it does and how to run it.

## Contributing

Adding a complete application or demo? Put it directly under
[`examples/`](./examples). Put model deployments under [`models/`](./models)
and robotics integrations under [`robotics/`](./robotics). Keep each project
self-contained, give the folder a name describing what it does rather than the
API it happens to call, and lead its README with the problem it solves.
Model deployments follow the rules in [GUIDELINES.md](./GUIDELINES.md).

## License

Apache-2.0 — see [LICENSE](./LICENSE).
