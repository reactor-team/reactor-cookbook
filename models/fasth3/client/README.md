# fasth3 queue client

A reference client for the fasth3 clip queue, built on the Reactor Python SDK
(`reactor-sdk`). It walks the whole contract once — enqueue two prompts with
metadata, watch `queue_update` report them ready, play the first clip to the
end, confirm the stream holds on black, play the second by UUID, stop it
mid-play — and writes everything it received to disk: one `.mp4` per clip
(video with its synchronized audio), the raw message log, and a timing report.

Use it as a smoke test after serving the model, or as the starting point for
your own queue-driving application.

## Run

```sh
pip install -r requirements.txt   # reactor-sdk + numpy; ffmpeg must be on PATH

# Against a local `reactor run` on :8080 (the default):
python client.py

# Against a hosted session:
python client.py --api-key rk_...

# Shorter clips build faster; outputs land in ./fasth3_out by default:
python client.py --seconds 5.167 --out ./out
```

What to expect: `enqueue` answers immediately with the clip's UUID; the clip
turns `ready: true` on `queue_update` after one build (roughly the clip's own
duration up to a few multiples of it, depending on the deployment's kernel
profile and GPU count); `play` puts first frames on the tracks within a
fraction of a second, because the clip is already built. Nothing plays on its
own: after `clip_finished` the stream stays black until the next `play`.
