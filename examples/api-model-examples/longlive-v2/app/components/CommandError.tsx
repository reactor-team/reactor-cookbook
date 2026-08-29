"use client";

import { useState } from "react";
import {
  useLongliveV2CommandError,
  useLongliveV2State,
} from "@reactor-models/longlive-v2";
import { cn, EYEBROW, PANEL } from "./ui";

// Surface command_error messages from the model. LongLive 2 emits these
// when a command fails its preconditions — for example, calling
// `start` before any prompt has been set. Without this component
// those failures are silent: the user clicks a button and nothing
// happens.
//
// We clear the error on the next `state` snapshot, since any state
// change implies the user has moved on from whatever triggered it.
export function CommandError() {
  const [error, setError] = useState<{
    command: string;
    reason: string;
  } | null>(null);

  useLongliveV2CommandError((msg) => {
    setError({ command: msg.command, reason: msg.reason });
  });

  useLongliveV2State(() => {
    setError(null);
  });

  if (!error) return null;

  return (
    <div className={cn(PANEL, "border-red-900/50 bg-red-950/20 p-3")}>
      <span className={cn(EYEBROW, "text-red-500")}>
        {error.command} failed
      </span>
      <p className="mt-1 text-sm text-red-300">{error.reason}</p>
    </div>
  );
}
