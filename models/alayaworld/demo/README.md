# AlayaWorld demo frontend

A small [Next.js](https://nextjs.org) app for driving the AlayaWorld example:
pick a starting image, write a prompt, and steer the camera from the keyboard
while the video streams back over WebRTC.

It is built on [`@reactor-team/js-sdk`](https://www.npmjs.com/package/@reactor-team/js-sdk)
plus a typed client generated from the model's own schema, so every command and
message in the app is checked against what the model actually serves.

It talks to the container you start yourself, on your own machine. Everything it
needs to know about where the model is comes from the environment, listed in
[`.env.example`](./.env.example), so copying the folder out and pointing it
somewhere else takes no code change.

## Run it

Start the model first, from the example root:

```sh
cd models/alayaworld
reactor build
reactor run --gpus device=0 -e HF_TOKEN
```

Then start the app:

```sh
cd demo
pnpm install
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) and follow the three steps
the page lists: **Connect**, choose a starting image, then drive.

With no configuration the app connects to `http://localhost:8080`, which is where
`reactor run` serves. If you passed `reactor run --port`, say so:

```sh
cp .env.example .env
# REACTOR_LOCAL_URL=http://localhost:18080
```

## Controls

Each axis takes a velocity between -1 and 1 that the model holds until it is
replaced, so a key press and a key release are one command each. The model reads
all six axes when the next chunk starts, which is why the panel names the chunk
your change will land on.

| Keys      | Axis       | Effect                     |
| --------- | ---------- | -------------------------- |
| `W` / `S` | `forward`  | Move forward and back      |
| `A` / `D` | `strafe`   | Move left and right        |
| `Space` / `C` | `vertical` | Rise and descend       |
| `I` / `K` | `pitch`    | Look up and down           |
| `J` / `L` | `yaw`      | Turn left and right        |
| `Q` / `E` | `roll`     | Roll counterclockwise and clockwise |

Looking around is on the keyboard rather than the mouse on purpose. The model
samples one velocity per chunk, so mouse deltas would be averaged into a single
number instead of felt as a movement.

The on-screen pad sends the same commands, so the demo works from a touchscreen
and shows which keys are down.

## How it works

The app wraps the SDK's `ReactorProvider` and reads the model through the
generated hooks:

```tsx
const model = useAlayaWorld();

// Commands are typed from the schema.
await model.setForward({ forward: 1 });
await model.setPrompt({ prompt: "a storm rolling in" });

// So are uploads.
const reference = await model.uploadFile(file);
await model.setImage({ image: reference });
```

The model broadcasts a complete snapshot of everything a client can observe
whenever anything changes, and sends one to each viewer as it joins. The app
treats that snapshot as its only source of truth, which is why the axis meters
and the chunk counter all agree with what the model will actually do next —
even when a second tab is driving:

```tsx
useAlayaWorldStateUpdate((state) => {
  state.forward; // the velocity the next chunk will use
  state.next_chunk; // the chunk a change queued now will land on
});
```

The generated client lives in [`lib/alayaworld/`](./lib/alayaworld) and is
regenerated from the model's schema — see the README there.

## Layout

| Path                 | What it holds                                                   |
| -------------------- | --------------------------------------------------------------- |
| `app/page.tsx`       | Reads the environment on the server and hands down the config    |
| `app/api/token/`     | Mints a session token, used only if you deploy the model yourself |
| `components/App.tsx` | Configures the connection                                       |
| `components/Workspace.tsx` | Subscribes to the model's snapshot and lays out the panels |
| `components/CameraBar.tsx` | The six axes, under the video                              |
| `lib/controls.ts`    | Key bindings and the press-to-velocity mapping                   |
| `lib/alayaworld/`    | Generated typed client — do not edit                             |
