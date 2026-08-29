# Generated model client

`client.ts` and `client.react.tsx` are generated from the model's own schema, so
every command, parameter, and message in this app is checked against what the
model actually serves. Do not edit them by hand.

`client.ts` holds the command parameters, the message types, and the track list.
`client.react.tsx` holds the React layer: `useAlayaWorld()` for the typed
commands, one hook per message, and `AlayaWorldMainVideoView` for the video.
Both import `Reactor` from `@reactor-team/js-sdk`, which the app depends on
directly.

The `AlayaWorld` in those names is the model's own name, pascal-cased:
`model.name` in the example's `reactor.yaml` is `alaya-world`, and the generator
splits it on the hyphen. Renaming the model renames this whole surface.

The app does not use the generated provider, because that provider fixes the
model name at generation time. It wraps the SDK's own `ReactorProvider` instead,
so the name can come from the environment.

## A command answers with a message

Every command this model declares returns one: `set_image` and `random_image`
answer with `image_selected`, `set_prompt` with `prompt_queued`, the six camera
axes with `camera_motion_changed`, `reset` with `rollout_reset_queued`. The
generated wrappers are typed from the schema, so the answer is the awaited
call's value:

```ts
const image = await model.setImage({ image: reference });
if (image) console.log(image.filename);
```

An answer is addressed to the connection that asked, so read it there rather
than through the matching message hook — the hook fires on this connection too,
but it cannot say which call it answers and it never fires for a second viewer.
Hooks are for what the model genuinely broadcasts: `state_update`, and
`rollout_reset_queued` when the rollout loop restarts on its own (which is why
that one message arrives both ways).

## Regenerating

The schema can be rendered from the model's source without weights or a GPU,
which is the quickest way to refresh these files after changing the Python
contract:

```sh
# from this model's root (the folder holding reactor.yaml)
uv run --with 'reactor-runtime==3.2.5' --with numpy --with pillow \
  python -m reactor_runtime.schema --path . --version v0.1.0 \
  --out /tmp/alayaworld-schema.json

npx @reactor-team/codegen \
  --schema /tmp/alayaworld-schema.json \
  --standalone --react \
  --output demo/lib/alayaworld/client.ts
```

Pin the runtime to whatever `build.runtime_version` in `reactor.yaml` names, and
pass `--version` the release tag from `model.version`, since that is what the
generator stamps into `MODEL_VERSION`. Reading the schema off a running
container works too — `curl -fsS localhost:8080/schema` — and is worth doing
when you want the contract exactly as deployed rather than as written.
