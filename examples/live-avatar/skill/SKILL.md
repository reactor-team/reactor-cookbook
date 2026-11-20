---
name: live-avatar
description: Set up and run the Live Avatar example end to end (the LiveAvatar model on Reactor, the LiveKit voice agent on OpenAI Realtime, and the client page), asking the user for what it needs first; and explain how LiveAvatar streams and keeps the avatar's face in sync with the agent's voice. Use when asked to set up, run, configure, debug or stop this example, its model, its voice agent or its page, or to explain how the avatar model or its lip sync works.
---

# Live Avatar

Three pieces, each its own process, plus LiveKit and OpenAI:

| Piece | Folder | Started with |
| --- | --- | --- |
| The avatar model | `liveavatar-model/` | `reactor build` and `reactor run` on a B200 machine, or a deployment on Reactor |
| The voice agent | `voice-agent/` | `uv run python agent.py dev` |
| The page and its server routes | `client/` | `npm run dev` |

The page's server starts each call: it creates the LiveKit room, dispatches the agent, starts the model's session with
the portrait and has it join the room, then lets the browser in. The README at the example's root is the reference for
every step below; this skill is the order to do them in and what to check.

## Ground rules

- Never print a secret: LiveKit's API secret, the OpenAI key, a Reactor API key (`rk_...`), or a room token. Write them
  into the env files (`client/.env.local`, `voice-agent/.env`; both are git-ignored) with `umask 077`, and when you need
  to show a file, mask its values: `sed 's/=.*/=…/' voice-agent/.env`.
- Put the weights (about 50 GB) on a disk with room; ask where if unsure.
- A running model holds a GPU. Stop it when the user is done.

## 1. Ask what the user has

Ask these before running anything, together if your tools allow a form. Skip what the user already said.

1. **Where should the model run?** On a machine with an NVIDIA B200 they can use (which machine, which GPU index), or
   on Reactor (they need the deployed model's name and a Reactor API key).
2. **A LiveKit project?** Its `wss://` URL, an API key and its secret. Without one, they can create a free project at
   [cloud.livekit.io](https://cloud.livekit.io) (Settings, then Keys). The model must reach the URL, so a LiveKit server
   on `localhost` works only when the model runs on that same machine.
3. **An OpenAI API key** with access to the Realtime API.
4. **Where will they open the page?** On the same machine, or remotely. Remote needs a port forward
   (`ssh -L 3000:127.0.0.1:3000 <host>`): browsers allow the microphone only on `localhost` or HTTPS.
5. **A name for the voice agent** (`AVATAR_AGENT_NAME`), unique on their LiveKit project, e.g. `avatar-agent-<name>`.

Check the tools: `nvidia-smi` (a B200 with about 80 GB free) and `docker` with the NVIDIA runtime on the GPU machine,
`reactor --version`, `uv --version`, `node --version` (20 or later). Point at the README's install links for any that
is missing.

## 2. The model

Skip to step 3 if it runs on Reactor.

```sh
cd liveavatar-model
python prepare_weights.py --output <weights folder> --allow-download   # once; downloads about 50 GB
reactor build
reactor run --gpus device=<gpu> --weights <weights folder> --port 8080
```

Wait until `curl -s http://localhost:8080/health` reports `"state":"available"`: the first load compiles the model and
renders warm-up blocks, a few minutes. If it fails, read the `reactor run` output for the first error; a missing weight
file names itself, and an out-of-memory error means the GPU is shared with another job.

## 3. The voice agent

```sh
cd voice-agent
cp .env.example .env   # then write LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, OPENAI_API_KEY, AVATAR_AGENT_NAME
uv run python agent.py dev
```

Check the log for `registered worker {"agent_name": "<AVATAR_AGENT_NAME>", ...}`. Keep it running.

## 4. The page

```sh
cd client
npm install
cp .env.example .env.local   # the same LiveKit values and AVATAR_AGENT_NAME, and the model's location
npm run dev
```

For the model, set `REACTOR_RUNTIME_URL=http://<gpu machine>:8080` (or `http://127.0.0.1:8080` on the same machine), or
`REACTOR_API_KEY` and `REACTOR_MODEL` for Reactor, never both. Check `curl -s http://127.0.0.1:3000/api/status`: it
names the model, the LiveKit host and the agent.

## 5. Hand over

Tell the user to open `http://localhost:3000` (through the port forward if remote), pick a face and a voice, and press
**Start call**; the avatar greets them a few seconds later. Each call logs one line on the page's server
(`call started`, with its room and how long the start took) and one when it stops.

If something fails, the page shows the error the server got. Match it against the README's "When it does not work"
table, then read the log of the piece it names: the page's server for the start sequence, the agent for the
conversation, the model for the avatar.

## 6. Stop

End calls first (the page's **End call** stops the model's session and deletes the room), then stop the page, the
agent, and the model with Ctrl+C, and free the GPU.

## Explaining how it works

When the user asks how the avatar model works, how it keeps real time, or how the face stays in sync with the voice,
read [`how-it-works.md`](how-it-works.md) and answer from it, pointing at the files it names. Answer the question asked;
it has more than any one answer needs.
