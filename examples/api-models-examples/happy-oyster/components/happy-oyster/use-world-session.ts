"use client";

// The session driver behind the fixed layout. It owns the user's pending
// intent, walks the model through connect → create (or attach) → auto-travel,
// and reduces every render's SDK snapshot to the single AppView the layout
// renders from.
//
// The walk is phase-driven, not a one-shot script: dev StrictMode's phantom
// mount/unmount disconnects the model mid-connect, so reacting to the phase
// means any aborted step simply gets retried on the next render.

import { useCallback, useEffect, useRef, useState } from "react";
import type { WorldIntent } from "@/lib/worlds";
import {
  deriveJourney,
  deriveView,
  type AppView,
  type JourneyStep,
} from "@/lib/view";
import { useHappyOysterClient, type HappyOysterClient } from "./ho-client";

// The intent lives above the provider (HappyOysterApp) because the mode it
// carries decides which model the session connects to. This hook receives it,
// walks it through the client, and drives run/exit back up.
export interface WorldSessionInput {
  intent: WorldIntent | null;
  /** Set a new intent (and its mode) — invoked by the browse surfaces. */
  onRun: (intent: WorldIntent) => void;
  /** Drop the intent, back to browsing. */
  onClearIntent: () => void;
}

export interface WorldSession {
  view: AppView;
  /** The loading steps of the API's machine, for the loading pane. */
  journey: JourneyStep[];
  client: HappyOysterClient;
  /** What the session is running, null while browsing. */
  intent: WorldIntent | null;
  /** The generated first frame that seeds the world, once the model reports it. */
  seedFrame: string | null;
  starting: boolean;
  /** Run a new intent through the session. */
  run: (intent: WorldIntent) => void;
  /** Retry the current intent from scratch. */
  retry: () => void;
  /** Drop the intent and disconnect: back to browsing. */
  exit: () => void;
  beginTravel: () => void;
}

export function useWorldSession({
  intent,
  onRun,
  onClearIntent,
}: WorldSessionInput): WorldSession {
  const client = useHappyOysterClient();
  const {
    phase,
    worldState,
    streaming,
    connect,
    createWorld,
    attachWorld,
    startTravel,
    disconnect,
  } = client;

  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const [tick, setTick] = useState(0);
  const busyRef = useRef(false);
  const connectTries = useRef(0);
  const lastConnectError = useRef<string | null>(null);
  const doneIntent = useRef<string | null>(null);
  const autoStarted = useRef<string | null>(null);

  const intentKey = intent
    ? `${attempt}:${
        intent.kind === "attach"
          ? intent.encryptedWorldId
          : JSON.stringify(intent.params)
      }`
    : null;

  useEffect(() => {
    if (!intent || !intentKey || error || busyRef.current) return;
    const step = (
      run: Promise<unknown>,
      onStepError: (cause: unknown) => void,
    ) => {
      busyRef.current = true;
      void run.catch(onStepError).finally(() => {
        busyRef.current = false;
        setTick((value) => value + 1);
      });
    };

    if (phase === "idle" || phase === "ended" || phase === "failed") {
      if (connectTries.current >= 3) {
        setError(lastConnectError.current ?? "Could not connect to the model.");
        return;
      }
      connectTries.current += 1;
      step(
        connect().then(() => {
          connectTries.current = 0;
        }),
        (cause) => {
          lastConnectError.current = String(cause);
        },
      );
      return;
    }
    if (phase === "connected" && doneIntent.current !== intentKey) {
      doneIntent.current = intentKey;
      step(
        intent.kind === "create"
          ? createWorld(intent.params)
          : attachWorld(intent.encryptedWorldId),
        (cause) => setError(String(cause)),
      );
    }
  }, [
    phase,
    tick,
    error,
    intentKey,
    intent,
    connect,
    createWorld,
    attachWorld,
  ]);

  const run = useCallback(
    (next: WorldIntent) => {
      connectTries.current = 0;
      lastConnectError.current = null;
      setError(null);
      setAttempt((value) => value + 1);
      onRun(next);
    },
    [onRun],
  );

  const retry = useCallback(() => {
    connectTries.current = 0;
    lastConnectError.current = null;
    setError(null);
    setAttempt((value) => value + 1);
  }, []);

  const exit = useCallback(() => {
    void disconnect().catch(() => {});
    setError(null);
    onClearIntent();
  }, [disconnect, onClearIntent]);

  const beginTravel = useCallback(() => {
    setError(null);
    setStarting(true);
    void startTravel()
      .catch((cause) => setError(String(cause)))
      .finally(() => setStarting(false));
  }, [startTravel]);

  // Auto-start once per intent: the first time its world reports ready, go
  // straight into the travel. Ending a travel lands back on the ready view
  // rather than looping into another.
  useEffect(() => {
    if (!intentKey || error || streaming || starting) return;
    if (phase !== "connected" || worldState?.phase !== "ready") return;
    if (autoStarted.current === intentKey) return;
    autoStarted.current = intentKey;
    beginTravel();
  }, [
    error,
    streaming,
    starting,
    phase,
    worldState?.phase,
    intentKey,
    beginTravel,
  ]);

  // Once this intent's world is up, a session drop exits to browse instead of
  // re-entering the connect loop — this example doesn't reconnect (SKILL.md).
  const established = useRef(false);
  useEffect(() => {
    established.current = false;
  }, [intentKey]);
  useEffect(() => {
    if (worldState?.phase === "ready" || worldState?.phase === "traveling")
      established.current = true;
    if (
      established.current &&
      intent &&
      (phase === "ended" || phase === "failed")
    )
      exit();
  }, [phase, worldState?.phase, intent, exit]);

  const autoStartPending =
    !!intentKey &&
    phase === "connected" &&
    worldState?.phase === "ready" &&
    autoStarted.current !== intentKey;

  const view = deriveView({
    intent,
    error,
    phase,
    worldState,
    streaming,
    starting,
    autoStartPending,
  });

  const journey = deriveJourney({
    intent,
    phase,
    worldState,
    streaming,
    starting,
    autoStartPending,
  });

  return {
    view,
    journey,
    client,
    intent,
    seedFrame: worldState?.first_frame ?? null,
    starting,
    run,
    retry,
    exit,
    beginTravel,
  };
}
