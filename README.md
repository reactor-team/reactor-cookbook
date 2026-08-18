<div align="center">

<img src="assets/banner.png" alt="Reactor Cookbook" width="100%" />

**Practical, runnable examples for building with the Reactor platform.**

[🌐 Reactor](https://reactor.inc) · [🛰️ reactor-webrtc](https://github.com/reactor-team/reactor-webrtc) · [⚙️ reactor-runtime](https://github.com/reactor-team/reactor-runtime)

</div>

---

Short, self-contained code you copy, run, and adapt — not reference
documentation. For API references and guides, see the
[Reactor docs](https://docs.reactor.inc).

## Structure

The two top-level folders sit on opposite sides of a session.

- [`models/`](./models) holds models you serve. Each folder is a `reactor`
  workspace you build and run with the CLI, and everything that example needs
  lives inside it — the adapter, its configuration, and any client shipped to
  demonstrate it. [Build your own model](https://docs.reactor.inc/deploy/overview)
  covers the workspace they follow.
- [`robotics/`](./robotics) holds code that drives a model someone else is
  already serving: policy quickstarts and closed-loop simulator integrations
  built on the Python SDK.

Each example has a README explaining what it does and how to run it.

## Contributing

Adding an example? Put it under [`models/`](./models) if it serves a model, and
under [`robotics/`](./robotics) if it drives one that is already served. Keep
each example self-contained, give the folder a name describing what it does
rather than the API it happens to call, and lead its README with the problem it
solves.

## License

Apache-2.0 — see [LICENSE](./LICENSE).
