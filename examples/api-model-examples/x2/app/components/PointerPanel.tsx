"use client";

import { useEffect, useState } from "react";
import { useX2, useX2StateUpdate } from "@reactor-models/x2";
import {
  DEFAULT_POINTER_READOUT,
  type X2PointerReadout,
} from "@/app/lib/types";
import { cn, Panel } from "./ui";

// Live readout of the model's drag pointer — the raw `set_pointer` fields as
// the model echoes them back in every `state_update` snapshot. Dragging on
// the edited output streams set_pointer(x, y, active); what renders here is
// the round trip, not the local gesture, so the numbers are exactly what the
// model is steering with. Available in every source mode.
//
// The pointer rides the same state_update as the rest of the session, but it
// changes every frame of a drag. Rather than reduce it into the shared
// X2UiState (which would re-render the whole workspace ~30 Hz), this panel
// subscribes to the stream itself and keeps the churn local — only these
// three fields re-render while a drag is in flight.
export function PointerPanel() {
  const { status } = useX2();
  const [readout, setReadout] = useState<X2PointerReadout>(
    DEFAULT_POINTER_READOUT,
  );

  // Track only the pointer fields; return the previous object when they are
  // unchanged so a non-pointer state_update doesn't re-render this panel.
  useX2StateUpdate((msg) => {
    setReadout((prev) =>
      prev.x === msg.pointer_x &&
      prev.y === msg.pointer_y &&
      prev.active === msg.pointer_active
        ? prev
        : { x: msg.pointer_x, y: msg.pointer_y, active: msg.pointer_active },
    );
  });

  // Snap back to the model's defaults on disconnect, matching the reset the
  // workspace applies to X2UiState (the model stops echoing once it's gone).
  useEffect(() => {
    if (status === "disconnected") setReadout(DEFAULT_POINTER_READOUT);
  }, [status]);

  const { x, y, active } = readout;

  return (
    <Panel label="Pointer">
      {/* Named after set_pointer's params (x, y, active); state_update
          echoes the same values as pointer_x / pointer_y / pointer_active. */}
      <dl className="grid grid-cols-3 gap-2 font-mono text-xs">
        <Field name="x" value={x.toFixed(3)} live={active} />
        <Field name="y" value={y.toFixed(3)} live={active} />
        <Field name="active" value={active ? "true" : "false"} live={active} />
      </dl>
      <p className="mt-2 text-xs text-zinc-600">
        Drag on the edited output to steer the subject. Sent as{" "}
        <code className="text-zinc-500">set_pointer</code>, echoed back here via{" "}
        <code className="text-zinc-500">state_update</code>.
      </p>
    </Panel>
  );
}

function Field({
  name,
  value,
  live,
}: {
  name: string;
  value: string;
  live: boolean;
}) {
  return (
    <div className="min-w-0 rounded-md border border-zinc-800 bg-zinc-950/60 px-2 py-1.5">
      <dt className="truncate text-[10px] tracking-wide text-zinc-600">
        {name}
      </dt>
      <dd
        className={cn(
          "tabular-nums transition-colors",
          live ? "text-brand" : "text-zinc-300",
        )}
      >
        {value}
      </dd>
    </div>
  );
}
