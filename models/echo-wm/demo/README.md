# Echo-WM Flash demo frontend

A small [Next.js](https://nextjs.org) app for driving the Echo-WM Flash
example: choose a starting image, write a prompt, steer the camera, and receive
synchronized generated video and audio over WebRTC.

It follows the AlayaWorld demo structure and uses
[`@reactor-team/js-sdk`](https://www.npmjs.com/package/@reactor-team/js-sdk)
with a typed client derived from this model's schema. Connection settings come
from the environment described in [`.env.example`](./.env.example).

## Run it

Build the weights folder as the [model guide](../README.md#run-with-the-reactor-cli)
shows, then start the model from the example root:

```sh
cd models/echo-wm
reactor build
reactor run --gpus device=0
```

Then start the app:

```sh
cd demo
pnpm install
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000), select **Connect**, choose
a starting image, and drive the world. Generated sound plays through the same
video element.

The default model URL is `http://localhost:8080`. Set a different URL when the
model uses another port:

```sh
cp .env.example .env
# REACTOR_LOCAL_URL=http://localhost:18091
```

## Controls

Each axis is a velocity from -1 to 1. Echo-WM holds the value until another
camera command arrives and samples all four axes together at the next chunk
boundary.

| Keys      | Axis      | Effect                |
| --------- | --------- | --------------------- |
| `W` / `S` | `forward` | Move forward and back |
| `A` / `D` | `strafe`  | Move left and right   |
| `I` / `K` | `pitch`   | Look up and down      |
| `J` / `L` | `yaw`     | Turn left and right   |

The on-screen pad sends the same atomic camera command, so touch and keyboard
input share one state. The field-of-view control covers Echo-WM's supported
30–120 degree range.

## How it works

The app wraps the SDK's `ReactorProvider` and reads the model through the typed
hooks:

```tsx
const model = useEchoWmFlash();

await model.setCameraMotion({
  forward: 1,
  strafe: 0,
  pitch: 0,
  yaw: 0,
});
await model.setPrompt({ prompt: "a storm rolls in with distant thunder" });

const reference = await model.uploadFile(file);
await model.setImage({ image: reference });
```

An upload without a newly entered prompt applies the backend's image-neutral
default for the new scene. `ReactorView` combines the `main_video` and
`main_audio` receive tracks in one playback element.

The model broadcasts a complete state snapshot after each observable change.
The UI uses that snapshot as its source of truth for camera meters, prompt,
generation state, and chunk progress:

```tsx
useEchoWmFlashStateUpdate((state) => {
  state.forward;
  state.generating;
  state.next_chunk;
});
```

The schema-derived client lives in [`lib/echo_wm/`](./lib/echo_wm).

## Layout

| Path                          | What it holds                                     |
| ----------------------------- | ------------------------------------------------- |
| `app/page.tsx`                | Resolves environment-backed connection settings   |
| `app/api/token/`              | Mints a scoped token for a hosted deployment      |
| `components/App.tsx`          | Configures the SDK connection                     |
| `components/Workspace.tsx`    | Subscribes to state and lays out the demo         |
| `components/CameraBar.tsx`    | Four-axis camera and field-of-view controls       |
| `components/ScenePanel.tsx`   | Random scene, upload, and prompt controls         |
| `components/RolloutPanel.tsx` | Seed, reset, progress, and generation state       |
| `lib/controls.ts`             | Keyboard bindings and atomic velocity calculation |
| `lib/echo_wm/`                | Schema-derived typed client and React hooks       |
