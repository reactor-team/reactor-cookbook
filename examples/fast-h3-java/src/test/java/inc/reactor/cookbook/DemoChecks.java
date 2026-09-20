package inc.reactor.cookbook;

import inc.reactor.sdk.JsonValue;
import inc.reactor.sdk.JsonValue.JsonObject;
import inc.reactor.sdk.DownloadedClip;
import inc.reactor.sdk.Reactor;
import inc.reactor.sdk.ReactorException;
import inc.reactor.sdk.ReactorOptions;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicInteger;

/** Offline checks, including loading the native binary shipped on Maven Central. No API key used. */
public final class DemoChecks {
    private DemoChecks() {}

    public static void main(String[] args) throws Exception {
        JsonObject scope = FastH3Demo.tokenScope();
        check(scope.getNumber("max_sessions").orElseThrow() == 1, "one-session token");
        check(scope.getNumber("max_session_duration_seconds").orElseThrow() == 300, "five-minute session cap");
        check(scope.getNumber("expires_after").orElseThrow() == 600, "ten-minute token expiry");
        check(scope.get("models").orElseThrow().equals(JsonValue.array(java.util.List.of(JsonValue.of(FastH3Demo.MODEL)))),
                "Fast H3 model scope");
        check(FastH3Demo.seconds("6") == 6, "minimum demo duration");
        check(FastH3Demo.seconds("14") == 14, "maximum demo duration");
        for (String value : new String[] {"NaN", "Infinity", "5", "15", "bad"}) {
            rejects(IllegalArgumentException.class, () -> FastH3Demo.seconds(value));
        }
        JsonObject clip = (JsonObject) JsonValue.object().put("clip_id", "wanted").put("seconds", 6).build();
        JsonObject flat = (JsonObject) JsonValue.object().put("type", "clip_generated").put("clip", clip).build();
        JsonObject nested = (JsonObject) JsonValue.object().put("type", "clip_generated")
                .put("data", JsonValue.object().put("clip", clip).build()).build();
        check(FastH3Demo.clipId(flat).equals("wanted"), "flat model envelope");
        check(FastH3Demo.clipId(nested).equals("wanted"), "nested model envelope");
        var queue = new LinkedBlockingQueue<JsonObject>();
        queue.add((JsonObject) JsonValue.object().put("type", "clip_generated")
                .put("clip", JsonValue.object().put("clip_id", "other").build()).build());
        queue.add(nested);
        check(FastH3Demo.receive(queue, "clip_generated", "wanted", deadline()) == nested, "match own clip ID");
        rejects(TimeoutException.class, () -> FastH3Demo.receive(queue, "clip_finished", "wanted", System.nanoTime()));
        for (String type : FastH3Demo.FAILURES) {
            queue.add((JsonObject) JsonValue.object().put("type", type).build());
            rejects(IllegalStateException.class, () -> FastH3Demo.receive(queue, "clip_finished", "wanted", deadline()));
        }
        check(FastH3Demo.diagnostic(JsonValue.object().put("reason", "bad\nrequest").build())
                .equals("bad request"), "single-line diagnostics");
        checkCleanup();
        checkRecordingRetries();
        byte[] red = {0, 0, (byte) 255, (byte) 255};
        check(Preview.image(red, 1, 1).getRGB(0, 0) == 0xffff0000, "BGRA red stays red");
        rejects(IllegalArgumentException.class, () -> Preview.image(red, 2, 1));
        rejects(IllegalArgumentException.class, () -> Preview.image(red, 0, 1));
        Path directory = Files.createTempDirectory("fast-h3-check-");
        Path image = directory.resolve("test.png");
        try (Preview preview = Preview.open(false)) {
            rejects(IOException.class, () -> preview.save(image));
            preview.submit(red, 1, 1);
            preview.save(image);
            check(Files.size(image) > 0, "PNG saved in headless mode");
        } finally {
            Files.deleteIfExists(image);
            Files.delete(directory);
        }
        try (Reactor reactor = Reactor.open(ReactorOptions.builder("https://api.reactor.inc", FastH3Demo.MODEL).build())) {
            check(reactor.sessionId().isEmpty(), "native library loads without creating a session");
        }
        System.out.println("PASS: token scope, input validation, event envelopes, clip matching, failures, timeout, concurrent cleanup, recording retries, pixels, PNG, published native library");
    }

    static void checkRecordingRetries() throws Exception {
        ReactorException timeout = ReactorException.of("REQUEST_TIMEOUT", "not ready", null, "download_clip", null);
        var expected = new DownloadedClip(Path.of("unused.mp4"), 100, 1);
        var attempts = new AtomicInteger();
        var result = FastH3Demo.downloadFinalized(() -> attempts.incrementAndGet() == 1
                ? CompletableFuture.failedFuture(timeout) : CompletableFuture.completedFuture(expected), 3);
        check(result == expected && attempts.get() == 2, "retry a not-ready recording");
        try {
            FastH3Demo.downloadFinalized(() -> CompletableFuture.failedFuture(timeout), 1);
            throw new AssertionError("Expected recording timeout");
        } catch (TimeoutException error) {
            check(error.getCause() == timeout, "retain last SDK timeout");
        }
        var denied = new IOException("access denied");
        try {
            FastH3Demo.downloadFinalized(() -> CompletableFuture.failedFuture(denied), 1);
            throw new AssertionError("Expected non-retryable failure");
        } catch (java.util.concurrent.ExecutionException error) {
            check(error.getCause() == denied, "propagate non-timeout without retry");
        }
    }

    static void checkCleanup() throws Exception {
        var pending = new CompletableFuture<Void>();
        var started = new CountDownLatch(2);
        var calls = new AtomicInteger();
        var cleanup = new FastH3Demo.SessionCleanup(() -> {
            calls.incrementAndGet();
            return pending;
        }, () -> {});
        Runnable disconnect = () -> {
            started.countDown();
            cleanup.disconnect();
        };
        var mainCleanup = CompletableFuture.runAsync(disconnect);
        var shutdownCleanup = CompletableFuture.runAsync(disconnect);
        try {
            check(started.await(1, TimeUnit.SECONDS), "both cleanup callers started");
            rejects(TimeoutException.class, () -> mainCleanup.get(20, TimeUnit.MILLISECONDS));
            rejects(TimeoutException.class, () -> shutdownCleanup.get(20, TimeUnit.MILLISECONDS));
        } finally {
            pending.complete(null);
        }
        mainCleanup.get(1, TimeUnit.SECONDS);
        shutdownCleanup.get(1, TimeUnit.SECONDS);
        check(calls.get() == 1, "shutdown joins the existing disconnect");

        var broken = new FastH3Demo.SessionCleanup(
                () -> CompletableFuture.failedFuture(new IOException("disconnect failed")), () -> {});
        try (broken) {
            throw new IOException("generation failed");
        } catch (IOException original) {
            check(original.getMessage().equals("generation failed"), "keep original failure");
            check(original.getSuppressed().length == 1, "attach cleanup failure");
        }
    }

    static long deadline() { return System.nanoTime() + TimeUnit.SECONDS.toNanos(1); }

    static void check(boolean condition, String description) {
        if (!condition) throw new AssertionError(description);
    }

    interface Checked { void run() throws Exception; }

    static void rejects(Class<? extends Exception> expected, Checked action) throws Exception {
        try {
            action.run();
        } catch (Exception error) {
            if (expected.isInstance(error)) return;
            throw error;
        }
        throw new AssertionError("Expected " + expected.getSimpleName());
    }
}
