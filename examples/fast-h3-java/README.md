# Fast H3 with the Reactor Java SDK

A small desktop/command-line app that generates one clip with `reactor/fast-h3`,
plays the video in a window, and saves `clip.mp4`, `frame.png`, and a run summary.
It uses the **published `inc.reactor:reactor-sdk:1.0.0` from Maven Central**, including
its packaged native library. No SDK checkout, Rust build, web server, or GPU is needed.

## Run

Requires **JDK 22 or later** (tested with Temurin 25) and Maven 3.9+.
The Java SDK supports desktop/server JVMs; it does not run on Java 21 or Android.
See the SDK's [supported platforms](https://github.com/reactor-team/reactor-client-sdks/tree/main/sdks/java#requirements).

For a desktop client, have your backend issue a scoped JWT for `reactor/fast-h3`:

```bash
export REACTOR_JWT='your-backend-issued-scoped-JWT'
./run.sh
```

For a **trusted local developer smoke test**, the app can also exchange a
`REACTOR_API_KEY` environment variable for a JWT restricted to Fast H3 and one
session, with a five-minute session cap and ten-minute token expiry. The app keeps
the same JWT throughout the session. Never bundle that key into an
application distributed to users. `REACTOR_JWT` takes precedence when both are set.
See [Java authentication](https://docs.reactor.inc/sdk-reference/java/reactor#authentication)
for the `fetchJwt` overload used here to scope models, sessions, and token expiry.

```bash
export REACTOR_API_KEY='your-Reactor-API-key'
./run.sh
```

Or choose a prompt, duration, and a fresh output directory:

```bash
./run.sh 'A red toy sailboat floats across a sunlit pond, gentle ripples.' 6 result-2
```

This creates a paid hosted session. Each run queues exactly one clip and disconnects
afterward. It does not retry a generation. The demo accepts 6–14 seconds; the model
snaps the request to its supported frame count, which is printed in the output.
Existing output files are never intentionally overwritten. Use a new output directory
for each run, including after a partially completed run.

Set `REACTOR_SHOW=0` for a headless run. The saved MP4 is still available afterward.
The preview displays video only; play the MP4 to hear any recorded audio.
Closing the preview window does not cancel generation; Ctrl-C triggers session cleanup.
The key stays in the process environment. Do not commit a key or put it in client-side code.
If using 1Password, resolve the environment reference with `op run -- ./run.sh`.

`run.sh` packages the app and launches Java with `--enable-native-access=ALL-UNNAMED`.
For Windows, use `mvn package` followed by Java with the same native-access flag and a
semicolon-separated classpath: `target/classes;target/dependency/*`.

## What the app checks

1. Read the backend-issued JWT (or exchange a developer API key) and connect to `reactor/fast-h3`.
2. Attach to `main_video` after connect, when the Java SDK knows the track names.
3. Disable autoplay and send `enqueue` with the prompt and requested seconds.
4. Wait for `clip_generated`, send `play`, and match `clip_started`/`clip_finished`
   to the generated clip ID. Count frames during playback, excluding the idle build.
5. Request a recording window using the model's effective duration and disconnect
   to flush the final chunk. Download the recording through the Java SDK and save
   a copied video frame as PNG.
6. Always disconnect and close native resources on errors/timeouts, too.

Fast H3 stops producing media after playback. Waiting for its recording while
keeping the session open can therefore stall. SDK 1.0.0 also stops polling a
not-yet-ready recording when disconnected. The demo ends the session first and
retries only recording retrieval for up to 60 seconds while finalization completes.
It never reconnects or queues another clip during those retries.

The MP4 is a rolling session recording, not a frame-exact model export. Control-message
delay and recording segment boundaries can shift its edges. `frame.png` is a frame
received during playback; it is useful for inspecting the decoded pixels.

Generation/playback has a 180-second deadline, downloads a 60-second deadline, and
disconnect a 15-second deadline. A cleanup failure is reported so the session can be
checked in the Reactor dashboard. Force-killing the process cannot run its shutdown hook.

## Verify without a key or paid session

```bash
./verify.sh
```

This compiles with Java lint enabled, checks token scope, duration validation, both model event
envelopes, clip-ID matching, model errors, timeouts, concurrent cleanup, BGRA pixel conversion, and PNG
output. It also loads and closes the native SDK without connecting to a model.
These checks use Java SE only; there is no added test-framework dependency.

The application is in `src/main/java/inc/reactor/cookbook/FastH3Demo.java`.
`Preview.java` provides the optional Swing display. The SDK and its transitive
dependencies are the only application dependencies.

## Sources

Adapted from Reactor's [connect/receive and record-clip Java examples](https://github.com/reactor-team/reactor-client-sdks/tree/main/sdks/java/examples)
and checked against all four published Java SDK pages:
[Installation](https://docs.reactor.inc/sdk-reference/java/installation),
[Reactor](https://docs.reactor.inc/sdk-reference/java/reactor),
[Track](https://docs.reactor.inc/sdk-reference/java/track), and
[Types](https://docs.reactor.inc/sdk-reference/java/types).
Model commands come from the [Fast H3 contract](https://docs.reactor.inc/model-api-reference/fast-h3/schema).
The generic Java examples mostly use Helios; H3 uses `enqueue`/`play`, rather than
Helios's `set_prompt`/`start`.
