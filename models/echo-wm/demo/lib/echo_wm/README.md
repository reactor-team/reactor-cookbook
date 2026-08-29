# Generated model client

`client.ts` and `client.react.tsx` are generated from the model's own schema, so
every command, parameter, and message in this app is checked against what the
model actually serves. Do not edit them by hand.

`client.ts` holds the command parameters, the message types, and the track list.
`client.react.tsx` holds the React layer: `useEchoWmFlash()` for the typed
commands, one hook per message, and the audiovisual `EchoWmFlashMainVideoView`
component. Both import `Reactor` from `@reactor-team/js-sdk`, which the app
depends on directly.

The `EchoWmFlash` in those names is the model's own name, pascal-cased:
`model.name` in `reactor.yaml` is `echo-wm-flash`, and the generator splits it on
the hyphens. Renaming the model renames this whole surface.

The app wraps the SDK's `ReactorProvider` rather than the generated
model-specific provider, so `REACTOR_MODEL_NAME` can select the deployed model
without changing source code.

## A command answers with a message

Every command this model declares returns one: `set_image` and `random_image`
answer with `image_selected`, `set_prompt` with `prompt_queued`, the camera
commands with `camera_motion_changed`, `reset` with `rollout_reset_queued`. The
generated wrappers are typed from the schema, so the answer is the awaited
call's value:

```ts
const image = await model.setImage({ image: reference });
if (image) console.log(image.filename);
```

An answer is addressed to the connection that asked, so read it there rather
than through the matching message hook — the hook fires on this connection too,
but it cannot say which call it answers and it never fires for a second viewer.
Hooks are for what the model genuinely broadcasts: `state_update`,
`chunk_completed`, and `automatic_reset_queued`.

## Regenerating

The schema can be rendered from the model's source without weights or a GPU,
which is the quickest way to refresh these files after changing the Python
contract:

```sh
# from this model's root (the folder holding reactor.yaml)
uv run --with 'reactor-runtime==3.2.5' --with numpy --with pillow \
  python -m reactor_runtime.schema --path . --version v0.1.0 \
  --out /tmp/echo-wm-schema.json

npx @reactor-team/codegen \
  --schema /tmp/echo-wm-schema.json \
  --standalone --react \
  --output demo/lib/echo_wm/client.ts
```

Pin the runtime to whatever `build.runtime_version` in `reactor.yaml` names, and
pass `--version` the release tag from `model.version`, since that is what the
generator stamps into `MODEL_VERSION`. Reading the schema off a running
container works too — `curl -fsS localhost:8080/schema` — and is worth doing
when you want the contract exactly as deployed rather than as written.

Then verify the frontend:

```sh
pnpm typecheck
pnpm build
```
