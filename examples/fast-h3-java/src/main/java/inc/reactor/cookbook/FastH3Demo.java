package inc.reactor.cookbook;

import inc.reactor.sdk.Clip;
import inc.reactor.sdk.CommandReply;
import inc.reactor.sdk.DownloadedClip;
import inc.reactor.sdk.JsonValue;
import inc.reactor.sdk.JsonValue.JsonObject;
import inc.reactor.sdk.Reactor;
import inc.reactor.sdk.ReactorOptions;
import inc.reactor.sdk.ReactorSdk;
import inc.reactor.sdk.RequestTimeoutException;
import inc.reactor.sdk.VideoFrame;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;

/** One text-to-video generation through the published Java SDK. */
public final class FastH3Demo {
    static final String MODEL = "reactor/fast-h3";
    static final Set<String> FAILURES = Set.of("command_error", "clip_failed", "clip_stopped", "sdk_error");
    static final Set<String> CLIP_EVENTS = Set.of("clip_generated", "clip_started", "clip_finished");

    private FastH3Demo() {}

    public static void main(String[] args) throws Exception {
        if (args.length == 1 && args[0].equals("--help")) {
            System.out.println("Usage: ./run.sh [prompt] [seconds: 6..14] [output-directory]\n"
                    + "Requires REACTOR_JWT (or REACTOR_API_KEY for local development); REACTOR_SHOW=0 disables the video window.\n"
                    + "Default: one six-second clip, saved under result/. Runs a paid hosted session.");
            return;
        }
        if (args.length > 3) throw new IllegalArgumentException("Expected at most three arguments; use --help");
        String prompt = args.length > 0 ? args[0] : "A red toy sailboat floats across a sunlit pond, gentle ripples, slow tracking shot.";
        if (prompt.isBlank()) throw new IllegalArgumentException("Prompt must not be blank");
        double seconds = args.length > 1 ? seconds(args[1]) : 6;
        Path output = Path.of(args.length > 2 ? args[2] : "result").toAbsolutePath();
        // Refuse to overwrite a previous run before creating a paid session.
        Files.createDirectories(output);
        for (String name : new String[] {"clip.mp4", "frame.png", "run.txt"}) {
            if (Files.exists(output.resolve(name))) throw new IllegalArgumentException("Output exists: " + output.resolve(name));
        }
        String apiUrl = "https://api.reactor.inc";
        System.out.println("Java SDK " + ReactorSdk.version() + "; model " + MODEL);
        String jwt = System.getenv("REACTOR_JWT");
        if (jwt == null || jwt.isBlank()) {
            // Developer-only convenience. Distributed desktop apps receive a scoped JWT
            // from their backend, as described in the Java SDK authentication documentation.
            String key = System.getenv("REACTOR_API_KEY");
            if (key == null || key.isBlank() || key.startsWith("op://")) {
                throw new IllegalArgumentException("Set REACTOR_JWT, or a resolved REACTOR_API_KEY for local development");
            }
            jwt = await(Reactor.fetchJwt(apiUrl, key, tokenScope(), false), 30);
        }
        try (Preview preview = Preview.open(!"0".equals(System.getenv("REACTOR_SHOW")));
                Reactor reactor = Reactor.open(ReactorOptions.builder(apiUrl, MODEL).jwt(jwt).build())) {
            SessionCleanup cleanup = new SessionCleanup(reactor::disconnect, reactor::close);
            Thread shutdown = new Thread(cleanup::close, "reactor-cleanup");
            Runtime.getRuntime().addShutdownHook(shutdown);
            try {
                // Resource cleanup preserves a generation failure and suppresses any cleanup failure.
                try (cleanup) {
                    run(reactor, cleanup, preview, prompt, seconds, output);
                }
            } finally {
                try {
                    Runtime.getRuntime().removeShutdownHook(shutdown);
                } catch (IllegalStateException shuttingDown) {
                    // The hook is already running and awaits the same disconnect future.
                }
            }
        }
    }

    static JsonObject tokenScope() {
        return (JsonObject) JsonValue.object()
                .putArray("models", List.of(JsonValue.of(MODEL)))
                .put("max_sessions", 1)
                .put("max_session_duration_seconds", 300)
                .put("expires_after", 600)
                .build();
    }

    static void run(Reactor reactor, SessionCleanup cleanup, Preview preview, String prompt, double seconds, Path output) throws Exception {
        BlockingQueue<JsonObject> events = new LinkedBlockingQueue<>();
        AtomicReference<String> target = new AtomicReference<>();
        AtomicBoolean playing = new AtomicBoolean();
        AtomicInteger frames = new AtomicInteger();
        long started = System.nanoTime();
        reactor.onStatus(status -> System.out.println("status: " + status));
        reactor.onError(error -> {
            if (!error.isRecoverable()) events.add((JsonObject) JsonValue.object()
                    .put("type", "sdk_error").put("reason", error.code()).build());
        });
        reactor.onMessage(value -> {
            if (!(value instanceof JsonObject event)) return;
            String type = event.getString("type").orElse("");
            if (type.equals("clip_started") && clipId(event).equals(target.get())) playing.set(true);
            if (type.equals("clip_finished") && clipId(event).equals(target.get())) playing.set(false);
            if (CLIP_EVENTS.contains(type) || FAILURES.contains(type)) {
                System.out.println("event: " + type + (clipId(event).isEmpty() ? "" : " " + clipId(event)));
                events.add(event);
            }
        });
        await(reactor.connect(), 60);
        String session = reactor.sessionId().orElseThrow();
        System.out.println("session: " + session);
        // Java discovers named tracks during connect; attach before sending any model commands.
        // A frame is only borrowed until this callback returns. Copy pixels before keeping them.
        reactor.track("main_video").onFrame((VideoFrame frame) -> {
            if (playing.get()) {
                int count = frames.incrementAndGet();
                if (count == 1) System.out.println("first playback frame: " + frame.width() + "x" + frame.height());
                preview.submit(frame.toByteArray(), frame.width(), frame.height());
            }
        });
        // Stop autoplay so generation, explicit play, and received media can be checked separately.
        command(reactor, "set_autoplay", JsonValue.object().put("enabled", false).build());
        Optional<CommandReply> queued = command(reactor, "enqueue", JsonValue.object()
                .put("prompt", prompt).put("seconds", seconds).build());
        String queuedId = queued.map(CommandReply::dataOrNull).filter(JsonObject.class::isInstance)
                .map(JsonObject.class::cast).map(FastH3Demo::clipId).orElse("");
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(180);
        JsonObject generated = receive(events, "clip_generated", queuedId, deadline);
        JsonObject clipInfo = clipInfo(generated);
        String clipId = clipId(generated);
        if (clipId.isBlank()) throw new IllegalStateException("clip_generated has no clip ID");
        double actualSeconds = clipInfo.getNumber("seconds").orElseThrow();
        if (!Double.isFinite(actualSeconds) || actualSeconds <= 0) throw new IllegalStateException("Invalid clip duration");
        target.set(clipId);
        System.out.println("generated: " + clipId + ", " + actualSeconds + " seconds");
        command(reactor, "play", JsonValue.object().put("clip_id", clipId).build());
        receive(events, "clip_started", clipId, deadline);
        receive(events, "clip_finished", clipId, deadline);
        if (frames.get() == 0) throw new IllegalStateException("Clip finished but no video frames arrived during playback");
        // Rolling recording window: use the model's snapped duration, not time spent generating.
        Clip recording = await(reactor.requestClip(actualSeconds), 20);
        preview.save(output.resolve("frame.png"));
        // H3 is idle now. Ending the session flushes its final recording chunk; keeping it
        // connected does not advance media time. Keep the native handle open for download.
        cleanup.disconnect();
        System.out.println("session disconnected; waiting for recording finalization");
        System.out.println("downloading recording...");
        var downloaded = downloadFinalized(
                () -> reactor.downloadClip(recording, output.resolve("clip.mp4"), 0, null), 60);
        if (downloaded.bytes() <= 0) throw new IllegalStateException("Empty recording download");
        String report = "sdk=" + ReactorSdk.version() + "\nmodel=" + MODEL + "\nsession=" + session
                + "\nclip=" + clipId + "\nrequested_seconds=" + seconds + "\nmodel_seconds=" + actualSeconds
                + "\nreceived_playback_frames=" + frames.get() + "\nmp4_bytes=" + downloaded.bytes()
                + "\nelapsed_seconds=" + TimeUnit.NANOSECONDS.toSeconds(System.nanoTime() - started) + "\n";
        Files.writeString(output.resolve("run.txt"), report);
        System.out.println("PASS: " + frames.get() + " playback frames; saved " + downloaded.path());
    }

    static Optional<CommandReply> command(Reactor reactor, String name, JsonValue args) throws Exception {
        Optional<CommandReply> reply = await(reactor.sendCommand(name, args), 30);
        if (reply.flatMap(CommandReply::type).filter(FAILURES::contains).isPresent()) {
            throw new IllegalStateException("Model rejected command: " + name + ": "
                    + diagnostic(reply.orElseThrow().dataOrNull()));
        }
        return reply;
    }

    static JsonObject payload(JsonObject event) {
        return event.get("data").filter(JsonObject.class::isInstance).map(JsonObject.class::cast).orElse(event);
    }

    static JsonObject clipInfo(JsonObject event) {
        return payload(event).get("clip").filter(JsonObject.class::isInstance).map(JsonObject.class::cast)
                .orElse((JsonObject) JsonValue.object().build());
    }

    static String clipId(JsonObject event) {
        return clipInfo(event).getString("clip_id").orElse("");
    }

    static JsonObject receive(BlockingQueue<JsonObject> events, String kind, String id, long deadline) throws Exception {
        while (true) {
            long remaining = deadline - System.nanoTime();
            if (remaining <= 0) throw new TimeoutException("Timed out waiting for " + kind);
            JsonObject event = events.poll(remaining, TimeUnit.NANOSECONDS);
            if (event == null) throw new TimeoutException("Timed out waiting for " + kind);
            String type = event.getString("type").orElse("");
            if (FAILURES.contains(type)) throw new IllegalStateException("Reactor reported " + type + ": " + diagnostic(event));
            if (type.equals(kind) && (id.isEmpty() || clipId(event).equals(id))) return event;
        }
    }

    static double seconds(String text) {
        double value = Double.parseDouble(text);
        if (!Double.isFinite(value) || value < 6 || value > 14) {
            throw new IllegalArgumentException("Use a duration between 6 and 14 seconds");
        }
        return value;
    }

    static <T> T await(CompletableFuture<T> future, long seconds) throws Exception {
        return future.get(seconds, TimeUnit.SECONDS);
    }

    static String diagnostic(JsonValue value) {
        if (!(value instanceof JsonObject object)) return "no diagnostic supplied";
        JsonObject data = payload(object);
        // Restrict output to diagnostic fields, rather than logging complete model/session payloads.
        String text = data.getString("reason").or(() -> data.getString("message"))
                .or(() -> data.getString("code")).orElse("no diagnostic supplied");
        return text.replaceAll("[\\r\\n\\p{Cntrl}]", " ").substring(0, Math.min(text.length(), 500));
    }

    static DownloadedClip downloadFinalized(Supplier<CompletableFuture<DownloadedClip>> retrieve, long timeoutSeconds) throws Exception {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(timeoutSeconds);
        RequestTimeoutException lastTimeout = null;
        while (true) {
            long remaining = deadline - System.nanoTime();
            if (remaining <= 0) {
                TimeoutException failure = new TimeoutException("Recording retrieval did not complete within " + timeoutSeconds + " seconds");
                if (lastTimeout != null) failure.initCause(lastTimeout);
                throw failure;
            }
            try {
                return retrieve.get().get(remaining, TimeUnit.NANOSECONDS);
            } catch (ExecutionException error) {
                // SDK 1.0.0 treats a 202 after disconnect as terminal. Finalization can still
                // be in flight, so retry retrieval only. This never reconnects or generates.
                if (!(error.getCause() instanceof RequestTimeoutException timeout)) throw error;
                if (lastTimeout == null) System.err.println("Recording not ready or retrieval timed out; retrying retrieval only ("
                        + timeout.code() + ", " + timeout.operation() + ")");
                lastTimeout = timeout;
                Thread.sleep(1000);
            }
        }
    }

    static final class SessionCleanup implements AutoCloseable {
        private final Supplier<CompletableFuture<Void>> leave;
        private final Runnable release;
        private CompletableFuture<Void> disconnected;

        SessionCleanup(Supplier<CompletableFuture<Void>> leave, Runnable release) {
            this.leave = leave;
            this.release = release;
        }

        private synchronized CompletableFuture<Void> leaving() {
            if (disconnected == null) disconnected = leave.get();
            return disconnected;
        }

        void disconnect() {
            try {
                await(leaving(), 15);
            } catch (Exception error) {
                throw new IllegalStateException("Disconnect failed; check the Reactor dashboard", error);
            }
        }

        @Override public void close() {
            try {
                disconnect();
            } finally {
                release.run();
            }
        }
    }
}
