"use client";

import { useState } from "react";
import {
  useLingbotCommandError,
  useLingbotState,
} from "@reactor-models/lingbot";

// Surface command_error messages from the model. Lingbot emits these
// when a command fails its preconditions — for example, calling
// `start` before a prompt or an image has been set, or uploading a
// file that isn't a valid image. Without this component those
// failures are silent: the user clicks a button and nothing happens.
//
// We clear the error on the next `state` snapshot, since any state
// change implies the user has moved on from whatever triggered it.
export function CommandError() {
  const [error, setError] = useState<{
    command: string;
    reason: string;
  } | null>(null);

  useLingbotCommandError((msg) => {
    setError({ command: msg.command, reason: msg.reason });
  });

  useLingbotState(() => {
    setError(null);
  });

  if (!error) return null;

  return (
    <div className="rounded-lg border border-red-900/50 bg-red-950/20 p-3">
      <span className="text-[10px] uppercase tracking-wider text-red-500">
        {error.command} failed
      </span>
      <p className="mt-1 text-sm text-red-300">{error.reason}</p>
    </div>
  );
}
