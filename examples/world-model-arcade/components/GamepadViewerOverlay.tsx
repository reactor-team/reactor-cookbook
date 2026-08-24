"use client";

import { MutableRefObject, useEffect, useRef, useState } from "react";
import type {
  ControllerAxes,
  InputMethod,
} from "@/hooks/useControllerInput";

type GamepadViewerOverlayProps = {
  axesRef: MutableRefObject<ControllerAxes>;
  buttonsRef: MutableRefObject<boolean[]>;
  connected: boolean;
  inputMethod: InputMethod;
  keysRef: MutableRefObject<Set<string>>;
  mode: "room" | "cabinet" | "booting" | "game";
  mouseLookActive: boolean;
};

const FACE_BUTTONS = [0, 1, 2, 3] as const;
const DPAD_BUTTONS = [12, 13, 14, 15] as const;

const MODE_ACTIONS = {
  room: [{ code: "Enter", key: "ENTER", label: "USE" }],
  cabinet: [
    { code: "Enter", key: "ENTER", label: "OPEN" },
    { code: "Digit3", key: "3", label: "PREVIEW" },
    { code: "Tab", key: "TAB", label: "BACK" },
  ],
  booting: [{ code: "Escape", key: "ESC", label: "CANCEL" }],
  game: [
    { code: "Digit1", key: "1", label: "ACTION" },
    { code: "Digit2", key: "2", label: "ACTION" },
    { code: "Digit3", key: "3", label: "ACTION" },
    { code: "Digit4", key: "4", label: "ACTION" },
    { code: "Tab", key: "TAB", label: "ARCADE" },
    { code: "KeyP", key: "P", label: "RESET" },
  ],
} as const;

export function GamepadViewerOverlay({
  axesRef,
  buttonsRef,
  connected,
  inputMethod,
  keysRef,
  mode,
  mouseLookActive,
}: GamepadViewerOverlayProps) {
  const buttonRefs = useRef<Array<HTMLSpanElement | null>>([]);
  const keyboardKeyRefs = useRef(new Map<string, HTMLElement>());
  const leftStickRef = useRef<HTMLSpanElement>(null);
  const rightStickRef = useRef<HTMLSpanElement>(null);
  const leftTriggerRef = useRef<HTMLSpanElement>(null);
  const rightTriggerRef = useRef<HTMLSpanElement>(null);
  const [controllerPromptDismissed, setControllerPromptDismissed] =
    useState(false);

  useEffect(() => {
    let frame = 0;
    const renderInput = () => {
      for (let index = 0; index < buttonRefs.current.length; index += 1) {
        buttonRefs.current[index]?.setAttribute(
          "data-pressed",
          buttonsRef.current[index] ? "true" : "false",
        );
      }

      const { lx, ly, rx, ry } = axesRef.current;
      if (leftStickRef.current) {
        leftStickRef.current.style.transform = `translate(${(lx * 14).toFixed(2)}px, ${(ly * 14).toFixed(2)}px)`;
      }
      if (rightStickRef.current) {
        rightStickRef.current.style.transform = `translate(${(rx * 14).toFixed(2)}px, ${(ry * 14).toFixed(2)}px)`;
      }
      if (leftTriggerRef.current) {
        leftTriggerRef.current.style.clipPath = buttonsRef.current[6]
          ? "inset(0 0 0 0)"
          : "inset(100% 0 0 0)";
      }
      if (rightTriggerRef.current) {
        rightTriggerRef.current.style.clipPath = buttonsRef.current[7]
          ? "inset(0 0 0 0)"
          : "inset(100% 0 0 0)";
      }
      keyboardKeyRefs.current.forEach((element, code) => {
        element.setAttribute(
          "data-pressed",
          keysRef.current.has(code) ? "true" : "false",
        );
      });
      frame = requestAnimationFrame(renderInput);
    };

    frame = requestAnimationFrame(renderInput);
    return () => cancelAnimationFrame(frame);
  }, [axesRef, buttonsRef, keysRef]);

  const bindButton = (index: number) => (element: HTMLSpanElement | null) => {
    buttonRefs.current[index] = element;
  };

  const bindKeyboardKey = (code: string) => (element: HTMLElement | null) => {
    if (element) keyboardKeyRefs.current.set(code, element);
    else keyboardKeyRefs.current.delete(code);
  };

  const actionKeys = MODE_ACTIONS[mode];
  const showControllerPrompt =
    mode === "room" && !connected && !controllerPromptDismissed;
  const mouseLookAvailable = mode === "room" || mode === "game";

  return (
    <aside
      className={`input-overlay input-overlay-${mode} input-overlay-${inputMethod}`}
      aria-label={
        inputMethod === "keyboard" ? "Live keyboard controls" : "Live controller input"
      }
    >
      {showControllerPrompt && (
        <div className="controller-connect-prompt">
          <span>
            <b>CONTROLLER OPTIONAL</b>
            Connect one anytime for analog controls.
          </span>
          <button
            type="button"
            onClick={() => setControllerPromptDismissed(true)}
            aria-label="Dismiss controller prompt"
          >
            HIDE
          </button>
        </div>
      )}
      <div className="input-overlay-caption">
        <span>INPUT VIEW</span>
        <span>{inputMethod === "gamepad" ? "XBOX PAD 01" : "KEYBOARD MAP"}</span>
      </div>
      <div className="input-overlay-viewport">
        {inputMethod === "keyboard" ? (
          <div className="keyboard-map" aria-hidden="true">
            <div className="keyboard-cluster">
              <div className="keyboard-key-grid keyboard-key-grid-wasd">
                <kbd ref={bindKeyboardKey("KeyW")}>W</kbd>
                <kbd ref={bindKeyboardKey("KeyA")}>A</kbd>
                <kbd ref={bindKeyboardKey("KeyS")}>S</kbd>
                <kbd ref={bindKeyboardKey("KeyD")}>D</kbd>
              </div>
              <span>{mode === "cabinet" ? "SELECT" : "MOVE"}</span>
            </div>
            <div className="keyboard-cluster">
              <kbd
                className="mouse-look-key"
                data-pressed={mouseLookAvailable && mouseLookActive ? "true" : "false"}
              >
                MOUSE
              </kbd>
              <span>
                {mouseLookAvailable
                  ? mouseLookActive
                    ? "LOOK ACTIVE"
                    : "CLICK TO LOOK"
                  : "SCREEN FIXED"}
              </span>
            </div>
            <div className="keyboard-actions">
              {actionKeys.map((action) => (
                <span className="keyboard-action" key={`${action.code}-${action.label}`}>
                  <kbd
                    ref={bindKeyboardKey(action.code)}
                    className={action.key.length > 1 ? "is-wide" : undefined}
                  >
                    {action.key}
                  </kbd>
                  <small>{action.label}</small>
                </span>
              ))}
            </div>
          </div>
        ) : (
          <div className="gpv-gamepad" aria-hidden="true">
            <div className="gpv-triggers">
              <span ref={leftTriggerRef} className="gpv-trigger gpv-left" />
              <span ref={rightTriggerRef} className="gpv-trigger gpv-right" />
            </div>
            <div className="gpv-bumpers">
              <span ref={bindButton(4)} className="gpv-bumper gpv-left" />
              <span ref={bindButton(5)} className="gpv-bumper gpv-right" />
            </div>
            <div className="gpv-arrows">
              <span ref={bindButton(8)} className="gpv-select" />
              <span ref={bindButton(9)} className="gpv-start" />
            </div>
            <div className="gpv-buttons">
              {FACE_BUTTONS.map((index) => (
                <span
                  key={index}
                  ref={bindButton(index)}
                  className={`gpv-button gpv-button-${["a", "b", "x", "y"][index]}`}
                />
              ))}
            </div>
            <div className="gpv-sticks">
              <span
                ref={(element) => {
                  leftStickRef.current = element;
                  buttonRefs.current[10] = element;
                }}
                className="gpv-stick gpv-left"
              />
              <span
                ref={(element) => {
                  rightStickRef.current = element;
                  buttonRefs.current[11] = element;
                }}
                className="gpv-stick gpv-right"
              />
            </div>
            <div className="gpv-dpad">
              {DPAD_BUTTONS.map((index) => (
                <span
                  key={index}
                  ref={bindButton(index)}
                  className={`gpv-face gpv-${["up", "down", "left", "right"][index - 12]}`}
                />
              ))}
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}
