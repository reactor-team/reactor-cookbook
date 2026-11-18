import { useEffect, useRef, useState } from "react";

const controls = [
  ["KeyW", "W", "forward", "Move forward"],
  ["KeyA", "A", "left", "Move left"],
  ["KeyS", "S", "backward", "Move backward"],
  ["KeyD", "D", "right", "Move right"],
  ["ArrowLeft", "←", "yaw_left", "Turn left"],
  ["ArrowUp", "↑", "pitch_up", "Look up"],
  ["ArrowDown", "↓", "pitch_down", "Look down"],
  ["ArrowRight", "→", "yaw_right", "Turn right"],
  ["KeyQ", "Q", "down", "Descend"],
  ["KeyE", "E", "up", "Ascend"],
  ["Space", "Space", "idle", "Hold position"],
] as const;

export type CameraAction = (typeof controls)[number][2];
export const actionLabel = (action: CameraAction) => controls.find((entry) => entry[2] === action)?.[3];
const keyActions = new Map<string, CameraAction>(controls.map(([key, , action]) => [key, action]));

export function CameraControls({ enabled, move, release }: {
  enabled: boolean;
  move: (action: CameraAction, hold: boolean) => void;
  release: (discardPending: boolean) => void;
}) {
  const gesture = useRef<string | null>(null);
  const [pressed, setPressed] = useState<CameraAction | null>(null);

  function end(source?: string, discardPending = false) {
    const active = gesture.current;
    if (source && active !== source) return;
    if (!active) {
      if (discardPending) release(true);
      return;
    }
    gesture.current = null;
    setPressed(null);
    release(discardPending);
  }

  function press(action: CameraAction, source: string) {
    if (!enabled || gesture.current === source) return;
    gesture.current = source;
    setPressed(action);
    move(action, action !== "idle");
  }

  useEffect(() => {
    if (!enabled) return;
    function keydown(event: KeyboardEvent) {
      const action = keyActions.get(event.code);
      if (!action || event.repeat || event.ctrlKey || event.altKey || event.metaKey) return;
      const target = event.target as HTMLElement | null;
      if (target?.closest("input, textarea, select, [contenteditable]")) return;
      if (event.code === "Space" && target?.closest("button")) return;
      event.preventDefault();
      press(action, event.code);
    }
    const keyup = (event: KeyboardEvent) => end(event.code);
    const blur = () => end(undefined, true);
    const visibility = () => { if (document.hidden) blur(); };
    window.addEventListener("keydown", keydown);
    window.addEventListener("keyup", keyup);
    window.addEventListener("blur", blur);
    document.addEventListener("visibilitychange", visibility);
    return () => {
      window.removeEventListener("keydown", keydown);
      window.removeEventListener("keyup", keyup);
      window.removeEventListener("blur", blur);
      document.removeEventListener("visibilitychange", visibility);
      gesture.current = null;
    };
  }, [enabled, move, release]);

  return (
    <section className="camera-controls" aria-label="Camera controls">
      <h2>Camera controls</h2>
      <div className="camera-grid">
        {controls.map(([code, key, action, label]) => (
          <button key={code} type="button" title={label} aria-label={label}
            className={action === "idle" ? "hold-position" : undefined}
            aria-pressed={enabled && pressed === action} disabled={!enabled}
            onPointerDown={(event) => {
              if (event.button !== 0) return;
              event.currentTarget.setPointerCapture(event.pointerId);
              press(action, `pointer-${event.pointerId}`);
            }}
            onPointerUp={(event) => end(`pointer-${event.pointerId}`)}
            onPointerCancel={(event) => end(`pointer-${event.pointerId}`, true)}
            onLostPointerCapture={(event) => end(`pointer-${event.pointerId}`, true)}
            onClick={(event) => { if (event.detail === 0) move(action, false); }}>
            <kbd>{key}</kbd><span>{label}</span>
          </button>
        ))}
      </div>
      <p className="muted compact">Hold to move continuously. Release to keep the current view.</p>
    </section>
  );
}
