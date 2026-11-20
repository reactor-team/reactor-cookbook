# Reactor Cookbook model gallery

A static site for the models under [`models/`](../models): one page per model
with its README as the model card, its files with the last commit that touched
each, and the shell command that runs it on a GPU node.

## What a model page shows

- **Model card**: the folder's `README.md`, with the hardware and runtime from
  its `reactor.yaml`, and whether it ships a frontend in `demo/`.
- **Files and versions**: the folder's tracked files, each with its last
  commit, and a viewer for text files and images.
- **Run it on a GPU node**: on a Linux node with Docker, the NVIDIA Container
  Toolkit and the [`reactor` CLI](https://docs.reactor.inc/deploy/platform/installation),
  a sparse clone of this repository that fetches only the model's folder,
  followed by `reactor run`. How the weights arrive decides its shape:

  | `weights` in `categories.yaml` | The page shows |
  | --- | --- |
  | `hf`: `runtime.weights_path` is an `hf://` reference | one command; `reactor run --yes` downloads and mounts the pinned snapshot |
  | `auto`: the model downloads its weights on first start | one command; the weights directory is created first, so Docker does not create it as root |
  | `manual`: the README downloads them by hand | two steps; the README's own download commands, verbatim, then the run command |
  | `readme`: no one-command start | the clone command, and a pointer to the README |

  `--gpus` comes from the GPU count in `reactor.yaml`: `device=0` for one GPU,
  `all` for more, with a note on pinning devices.
  When the image compiles something heavy (a `--no-build-isolation` install,
  `FLASH_ATTENTION_FORCE_BUILD`, `TORCH_CUDA_ARCH_LIST`, `nvcc`, `cmake --build`
  in the Dockerfile or build block), the page says so.
- **Serve it on Reactor**: deploy the folder under the reader's own account
  with `reactor model publish --remote`, `reactor weights show` and
  `reactor model deploy`, then check it with `reactor model status`,
  `reactor model instances` and `reactor logs`. The GPU type and count, the
  regions and the release come from `reactor.yaml`. The page marks a model
  "Needs changes to deploy", with the reasons, when `reactor.yaml` has no
  `deployment:` block (and there is no `deployment.yaml`), when the remote build
  context (the folder minus `.dockerignore`) is over a `reactor build --remote`
  limit, when a weights folder or `demo/node_modules` inside the folder is not
  ignored, or when `categories.yaml` records a `serve_blocker`.

## Families

[`categories.yaml`](./categories.yaml) places every model folder in a family
(video: autoregressive, clip by clip, code video models; robot policies) and
records how its weights arrive. The build fails if a folder under `models/` is
missing from it, if an entry names a folder that does not exist, if an `auto`
entry's evidence (the file and the download call in it) is gone, or if a
`manual` entry's README no longer has the download commands, if a model's
image compiles something heavy without a `first_build` note (or has the note
and compiles nothing), or if `reactor.yaml` has no `model.name`, `model.version`
or `model.resources.gpu`.

## Building it

Node.js 22 or later. From this directory:

```sh
npm ci
npm run dev      # http://localhost:4321
npm run build    # static site in dist/
```

The build reads the repository it sits in (`..`), including its git history
for the per-file commits, so build from a full clone rather than a shallow one.
`CATALOG_REPO_DIR` points it at another checkout, and `CATALOG_BASE` sets the
path the site is served from (`/` by default).

## Where it is published

The site is built and published from a machine, not from GitHub Actions; the
details of that machine and its schedule will be added here.
