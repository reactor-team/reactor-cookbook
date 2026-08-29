"use client";

import { useEffect, useRef } from "react";

export interface NoticeValue {
  kind: "info" | "error";
  text: string;
}

// Dismissible banner for command rejections and one-shot confirmations.
//
// Every `command_error` lands here. Never swallow one: this model refuses all
// six `set_*` commands mid-run, so a rejection usually means the client's view
// of the session has drifted from the model's, and hiding it turns a visible
// bug into a mysterious one.
export function Notice({
  value,
  onDismiss,
}: {
  value: NoticeValue | null;
  onDismiss: () => void;
}) {
  // Auto-dismiss after 6s. `onDismiss` is read through a ref so a parent
  // re-render (this app re-renders on every window's state_update) can't
  // restart the timer and keep the banner up indefinitely.
  const dismissRef = useRef(onDismiss);
  dismissRef.current = onDismiss;

  useEffect(() => {
    if (!value) return;
    const t = setTimeout(() => dismissRef.current(), 6000);
    return () => clearTimeout(t);
  }, [value]);

  if (!value) return null;

  const error = value.kind === "error";
  return (
    <div
      role="status"
      className={`flex items-center gap-3 rounded-lg border px-3 py-2 ${
        error
          ? "border-red-500/40 bg-red-500/5 text-red-200"
          : "border-brand/40 bg-brand/5 text-zinc-200"
      }`}
    >
      <span className="text-sm">{value.text}</span>
      <button
        onClick={onDismiss}
        aria-label="Dismiss"
        className="ml-auto shrink-0 text-lg leading-none opacity-50 hover:opacity-100"
      >
        ×
      </button>
    </div>
  );
}
