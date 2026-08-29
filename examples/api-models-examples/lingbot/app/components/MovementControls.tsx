"use client";

import { useCallback, useEffect, useState, type ReactNode } from "react";
import {
  useLingbot,
  useLingbotState,
  type LingbotStateMessage,
} from "@reactor-models/lingbot";

// Live-phase panel — Lingbot's signature capability.
//
// Lingbot is a real-time interactive world model: while it's
// generating, the client can stream movement and camera commands
// that the model picks up at chunk boundaries. The output video
// reflects those inputs a fraction of a second later, producing the
// feeling of "driving" the scene.
//
// Three commands flow from this component:
//   - set_movement         "idle" | "forward" | "back" | "strafe_left" | "strafe_right"
//   - set_look_horizontal  "idle" | "left" | "right"
//   - set_look_vertical    "idle" | "up" | "down"
//
// They are mapped to WASD + arrow keys both in the on-screen pad
// (good for touch / discoverability) AND in a global keyboard
// listener (good for the actual gameplay feel). Releasing a key
// returns the corresponding axis to "idle" automatically.
//
// We also expose a rotation-speed slider (`set_rotation_speed_deg`,
// 0–30 deg/latent-frame) — the only "knob" the live phase tweaks.
//
// IMPORTANT — UI highlighting reads LOCAL PRESS STATE, not the
// snapshot. The snapshot.movement / look_* fields lag user input by
// a chunk (they reflect what the model is currently using to
// generate, not what was just pressed) which makes the buttons
// flicker visibly behind the user's fingers. Local press state is
// instant and matches what the user just did. The slider, on the
// other hand, is a persistent value with no "release" — that still
// reads from the snapshot.

type Movement = "idle" | "forward" | "back" | "strafe_left" | "strafe_right";
type LookH = "idle" | "left" | "right";
type LookV = "idle" | "up" | "down";

// Keys → axis values. WASD for translation, arrow keys for look.
// Any additional key bindings (e.g. shift for sprint, gamepad
// sticks) would be added here.
const MOVEMENT_KEYS: Record<string, Movement> = {
  w: "forward",
  s: "back",
  a: "strafe_left",
  d: "strafe_right",
};
const LOOK_H_KEYS: Record<string, LookH> = {
  arrowleft: "left",
  arrowright: "right",
};
const LOOK_V_KEYS: Record<string, LookV> = {
  arrowup: "up",
  arrowdown: "down",
};

export function MovementControls() {
  const {
    status,
    setMovement,
    setLookHorizontal,
    setLookVertical,
    setRotationSpeedDeg,
  } = useLingbot();
  const [snapshot, setSnapshot] = useState<LingbotStateMessage | null>(null);

  // Local "what the user is pressing right now" state. Drives the
  // button highlights so they react instantly instead of waiting for
  // the next state snapshot to come back from the model.
  const [pressedMovement, setPressedMovement] = useState<Movement>("idle");
  const [pressedLookH, setPressedLookH] = useState<LookH>("idle");
  const [pressedLookV, setPressedLookV] = useState<LookV>("idle");

  useLingbotState((msg) => setSnapshot(msg));

  // Clear on disconnect. Also clear local press state — otherwise
  // a button could remain highlighted across a reconnect.
  useEffect(() => {
    if (status !== "ready") {
      setSnapshot(null);
      setPressedMovement("idle");
      setPressedLookH("idle");
      setPressedLookV("idle");
    }
  }, [status]);

  const ready = status === "ready" && snapshot?.started === true;

  // Send each axis as a typed event AND update local press state.
  // We don't try to debounce — Lingbot only consults the value at
  // the next chunk boundary, so sending more frequent updates is
  // harmless. Local state is the source of truth for the UI.
  const sendMovement = useCallback(
    (m: Movement) => {
      if (!ready) return;
      setPressedMovement(m);
      setMovement({ movement: m });
    },
    [ready, setMovement],
  );
  const sendLookH = useCallback(
    (l: LookH) => {
      if (!ready) return;
      setPressedLookH(l);
      setLookHorizontal({ look_horizontal: l });
    },
    [ready, setLookHorizontal],
  );
  const sendLookV = useCallback(
    (l: LookV) => {
      if (!ready) return;
      setPressedLookV(l);
      setLookVertical({ look_vertical: l });
    },
    [ready, setLookVertical],
  );

  // Global keyboard handling. We attach a single keydown/keyup pair
  // to the window so the pad responds even when the user hasn't
  // clicked into anything. Each axis is tracked independently so
  // holding W + A simultaneously produces "w+a" on the model.
  //
  // We deliberately don't filter out repeat events on keydown — the
  // first event sets the axis, subsequent repeats re-send the same
  // value, which the model treats as a no-op.
  useEffect(() => {
    if (!ready) return;

    const onKeyDown = (e: KeyboardEvent) => {
      // Don't hijack keys when the user is typing into an input.
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.isContentEditable)
      ) {
        return;
      }
      const k = e.key.toLowerCase();
      if (MOVEMENT_KEYS[k]) {
        e.preventDefault();
        sendMovement(MOVEMENT_KEYS[k]);
      } else if (LOOK_H_KEYS[k]) {
        e.preventDefault();
        sendLookH(LOOK_H_KEYS[k]);
      } else if (LOOK_V_KEYS[k]) {
        e.preventDefault();
        sendLookV(LOOK_V_KEYS[k]);
      }
    };

    const onKeyUp = (e: KeyboardEvent) => {
      const k = e.key.toLowerCase();
      if (MOVEMENT_KEYS[k]) {
        e.preventDefault();
        sendMovement("idle");
      } else if (LOOK_H_KEYS[k]) {
        e.preventDefault();
        sendLookH("idle");
      } else if (LOOK_V_KEYS[k]) {
        e.preventDefault();
        sendLookV("idle");
      }
    };

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [ready, sendMovement, sendLookH, sendLookV]);

  if (status !== "ready" || !snapshot?.started) return null;

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
      <label className="text-[10px] uppercase tracking-wider text-zinc-500">
        Drive the scene
      </label>

      <p className="mt-2 text-[11px] leading-snug text-zinc-500">
        WASD moves the subject. Arrow keys turn the camera. Hold to sustain —
        release to stop.
      </p>

      <div className="mt-3 grid grid-cols-2 gap-3">
        {/* Move pad — single axis. */}
        <PadFrame
          title="Move"
          legend="W A S D"
          top={
            <PadButton
              label="W"
              pressed={pressedMovement === "forward"}
              onPress={() => sendMovement("forward")}
              onRelease={() => sendMovement("idle")}
            />
          }
          left={
            <PadButton
              label="A"
              pressed={pressedMovement === "strafe_left"}
              onPress={() => sendMovement("strafe_left")}
              onRelease={() => sendMovement("idle")}
            />
          }
          right={
            <PadButton
              label="D"
              pressed={pressedMovement === "strafe_right"}
              onPress={() => sendMovement("strafe_right")}
              onRelease={() => sendMovement("idle")}
            />
          }
          bottom={
            <PadButton
              label="S"
              pressed={pressedMovement === "back"}
              onPress={() => sendMovement("back")}
              onRelease={() => sendMovement("idle")}
            />
          }
        />

        {/* Look pad — same 3x3 layout as Move, but each axis goes to a
            different setter: vertical arrows → set_look_vertical,
            horizontal arrows → set_look_horizontal. */}
        <PadFrame
          title="Look"
          legend="↑ ↓ ← →"
          top={
            <PadButton
              label="↑"
              pressed={pressedLookV === "up"}
              onPress={() => sendLookV("up")}
              onRelease={() => sendLookV("idle")}
            />
          }
          left={
            <PadButton
              label="←"
              pressed={pressedLookH === "left"}
              onPress={() => sendLookH("left")}
              onRelease={() => sendLookH("idle")}
            />
          }
          right={
            <PadButton
              label="→"
              pressed={pressedLookH === "right"}
              onPress={() => sendLookH("right")}
              onRelease={() => sendLookH("idle")}
            />
          }
          bottom={
            <PadButton
              label="↓"
              pressed={pressedLookV === "down"}
              onPress={() => sendLookV("down")}
              onRelease={() => sendLookV("idle")}
            />
          }
        />
      </div>

      <label className="mt-4 block text-[10px] uppercase tracking-wider text-zinc-500">
        Rotation speed · {snapshot.rotation_speed_deg.toFixed(1)}°/frame
      </label>
      <input
        type="range"
        min={0}
        max={30}
        step={0.5}
        value={snapshot.rotation_speed_deg}
        onChange={(e) =>
          setRotationSpeedDeg({ rotation_speed_deg: Number(e.target.value) })
        }
        className="mt-2 w-full accent-[color:var(--reactor-color-light-gold)]"
      />
    </div>
  );
}

// A framed 3x3 pad where the four corners are empty, the centre is
// a non-interactive neutral marker, and each directional slot is
// supplied as a named prop. Both Move and Look pads share this
// shape — Move binds all four slots to one axis, Look binds the
// vertical slots and the horizontal slots to two different axes.
function PadFrame({
  title,
  legend,
  top,
  left,
  right,
  bottom,
}: {
  title: string;
  legend: string;
  top: ReactNode;
  left: ReactNode;
  right: ReactNode;
  bottom: ReactNode;
}) {
  return (
    <div className="rounded-md border border-zinc-800 bg-zinc-950 p-2">
      <div className="flex items-baseline justify-between">
        <span className="text-[10px] uppercase tracking-wider text-zinc-400">
          {title}
        </span>
        <span className="font-mono text-[10px] text-zinc-600">{legend}</span>
      </div>
      <div className="mt-2 grid grid-cols-3 gap-1">
        <span />
        {top}
        <span />
        {left}
        <IdleCenter />
        {right}
        <span />
        {bottom}
        <span />
      </div>
    </div>
  );
}

function PadButton({
  label,
  pressed,
  onPress,
  onRelease,
}: {
  label: string;
  pressed: boolean;
  onPress: () => void;
  onRelease: () => void;
}) {
  return (
    <button
      onMouseDown={onPress}
      onMouseUp={onRelease}
      onMouseLeave={() => pressed && onRelease()}
      onTouchStart={onPress}
      onTouchEnd={onRelease}
      className={[
        "select-none rounded-sm border px-2 py-2 text-center text-sm font-medium transition-colors",
        pressed
          ? "border-brand bg-brand text-brand-fg"
          : "border-zinc-800 bg-zinc-950 text-zinc-300 hover:border-brand hover:text-brand",
      ].join(" ")}
    >
      {label}
    </button>
  );
}

// Non-interactive neutral marker at the centre of each pad. A
// small outlined circle just anchors the cross visually without
// inviting a click.
function IdleCenter() {
  return (
    <span className="grid place-items-center">
      <span className="h-2 w-2 rounded-full border border-zinc-700" />
    </span>
  );
}
