"use client";

import type { ReactorStatus } from "@reactor-team/js-sdk";
import {
  startBlockedReason,
  statusLine,
  validCommands,
  type TransportCommand,
} from "@/app/lib/machine";
import type { Ltx2UiState } from "@/app/lib/types";
import { Button, Icon, Panel } from "./ui";

// Transport: start a take, act on the one that is running, or wipe the
// session. Which buttons are live comes straight from the model, via
// `state_update.valid_commands` — see lib/machine.ts. `start` goes dead the
// moment a take begins and comes back when it ends; the conditions in the
// take panel below stay live throughout, because the model queues them.
//
// This lives in the sidebar with every other control rather than in a dock
// under the stage: the stage is for the take, and nothing competes with it.
export function Transport({
  status,
  ui,
  hasTake,
  ttffMs,
  presetPending,
  imagePending,
  onCommand,
}: {
  status: ReactorStatus;
  ui: Ltx2UiState;
  hasTake: boolean;
  ttffMs: number | null;
  presetPending: string | null;
  imagePending: boolean;
  onCommand: (command: TransportCommand) => Promise<void>;
}) {
  const valid = validCommands(status, ui);
  const fire = (command: TransportCommand) =>
    void onCommand(command).catch(() => {});

  // The readout belongs next to the buttons it describes, and it is the same
  // line the stage centers over an empty frame — so it appears here only when
  // the stage is NOT already showing it. Exactly one of the two, always.
  const stageIsEmpty = !hasTake;

  // The one thing the model's own `valid_commands` cannot tell you. It keeps
  // listing `start` while an avatar image is on its way, and starting then is
  // accepted — the take is simply generated from the face the model still has,
  // which is the old one. So holding Start across that window is the client's
  // job. A preset uploads an image of its own, so both flags are true during
  // its image step; the image is the more specific answer of the two.
  const held = imagePending || presetPending !== null;
  const holdReason = imagePending
    ? "Waiting for the model to confirm the avatar image…"
    : presetPending !== null
      ? "Applying the preset…"
      : null;

  // One line under the button, for whichever reason applies. A hold outranks
  // the readiness reason: it is what is stopping you right now, and unlike the
  // other it clears on its own. The readiness reason is worth spelling out at
  // all because `ready` is one flag covering two conditions, and the
  // pre-filled scene prompt makes it easy to believe the script has been
  // written when it has not.
  const startReason =
    holdReason ??
    (status === "ready" && !ui.generating ? startBlockedReason(ui) : null);

  return (
    <Panel label="Transport">
      <div className="flex flex-col gap-2">
        <Button
          variant="primary"
          size="md"
          className="w-full"
          disabled={!valid.has("start") || held}
          onClick={() => fire("start")}
          leadingIcon={<Icon name="play" />}
        >
          {held ? "Preparing…" : "Start take"}
        </Button>

        {startReason && (
          <p className="-mt-0.5 text-[11px] leading-snug text-zinc-500">
            {startReason}
          </p>
        )}

        <div className="grid grid-cols-3 gap-1.5">
          <Button
            variant="secondary"
            size="sm"
            disabled={!valid.has("pause")}
            onClick={() => fire("pause")}
            leadingIcon={<Icon name="pause" />}
          >
            Pause
          </Button>
          <Button
            variant="secondary"
            size="sm"
            disabled={!valid.has("resume")}
            onClick={() => fire("resume")}
            leadingIcon={<Icon name="play" />}
          >
            Resume
          </Button>
          <Button
            variant="secondary"
            size="sm"
            disabled={!valid.has("stop")}
            onClick={() => fire("stop")}
          >
            Stop
          </Button>
        </div>

        <Button
          variant="ghost"
          size="sm"
          className="w-full"
          disabled={!valid.has("reset")}
          onClick={() => fire("reset")}
          leadingIcon={<Icon name="reset" />}
        >
          Reset
        </Button>

        {!stageIsEmpty && (
          <div className="mt-0.5 flex flex-col gap-1 border-t border-edge pt-2 font-mono text-[11px]">
            <span className="text-zinc-400">{statusLine(status, ui)}</span>
            {ttffMs !== null && (
              <span
                className="text-zinc-600"
                title="From `start` going on the wire to the first frame composited after generation_started"
              >
                {(ttffMs / 1000).toFixed(1)}s to first frame
              </span>
            )}
          </div>
        )}
      </div>
    </Panel>
  );
}
