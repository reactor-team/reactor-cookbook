# Models

Models you serve on Reactor. Every folder here is a `reactor` workspace: a
`reactor.yaml` naming the model and the GPU it wants, an adapter built on the
[Reactor Runtime](https://github.com/reactor-team/reactor-runtime), and a
Dockerfile the CLI builds into the image it runs. `reactor build` and
`reactor run` serve any of them on your own machine, and
[Build your own model](https://docs.reactor.inc/deploy/overview) explains the
shape they follow.

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
own subfolder. Code that drives a model someone else is already serving belongs
in [`robotics/`](../robotics) instead.
