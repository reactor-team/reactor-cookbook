# Models

Recipes and runnable examples organized by model. A self-hosted model folder is
a `reactor` workspace: a `reactor.yaml` naming the model and the GPU it wants,
an adapter built on the [Reactor Runtime](https://github.com/reactor-team/reactor-runtime),
and a Dockerfile the CLI builds into the image it runs. `reactor build` and
`reactor run` serve those recipes on your own machine, and [Build your own
model](https://docs.reactor.inc/deploy/overview) explains the shape they follow.

For a hosted Reactor model, use a folder named after the model and place each
complete client beneath it. For example, `lingbot-world-2/arcade` is a runnable
application powered by the hosted LingBot World 2 model rather than a runtime
adapter.

An example folder is self-contained. Whatever it needs stays inside it —
configuration, pinned upstream revisions, and any client written to demonstrate
it — so the model and the code that exercises it are read, changed, and copied
together.

Name a folder for the model it serves, and give it a README covering:

- What it does and when you'd reach for it
- Prerequisites (versions, credentials, GPU, environment)
- How to run it
- Notes on anything surprising in the code

Nothing lives directly in this folder besides this file — every model gets its
own subfolder. Model-specific product clients stay with their model; broader
robotics integrations belong in [`robotics/`](../robotics).
