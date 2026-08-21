"use client";

import { useEffect, useRef, useState } from "react";

export type ControllerAxes = {
  lx: number;
  ly: number;
  rx: number;
  ry: number;
};

type ControllerOptions = {
  onButtonDown: (button: number) => void;
  onButtonUp: (button: number) => void;
};

const DEAD_ZONE = 0.16;
const KEY_TO_BUTTON: Record<string, number> = {
  Enter: 0,
  Space: 0,
  Backspace: 1,
  KeyW: 12,
  KeyS: 13,
  Digit1: 0,
  Digit2: 1,
  Digit3: 2,
  Digit4: 3,
  KeyQ: 4,
  KeyE: 5,
  Tab: 8,
  Escape: 8,
  KeyP: 9,
};

function cleanAxis(value = 0) {
  return Math.abs(value) < DEAD_ZONE ? 0 : value;
}

function isTypingTarget(target: EventTarget | null) {
  if (!(target instanceof HTMLElement)) return false;
  return (
    target.tagName === "INPUT" ||
    target.tagName === "TEXTAREA" ||
    target.isContentEditable
  );
}

export function useControllerInput({ onButtonDown, onButtonUp }: ControllerOptions) {
  const axesRef = useRef<ControllerAxes>({ lx: 0, ly: 0, rx: 0, ry: 0 });
  const buttonsRef = useRef<boolean[]>(Array(18).fill(false));
  const keyboardRef = useRef(new Set<string>());
  const handlersRef = useRef({ onButtonDown, onButtonUp });
  const [connected, setConnected] = useState(false);
  const [lastDevice, setLastDevice] = useState<"gamepad" | "keyboard">(
    "keyboard",
  );

  handlersRef.current = { onButtonDown, onButtonUp };

  useEffect(() => {
    const keys = keyboardRef.current;

    const onKeyDown = (event: KeyboardEvent) => {
      if (isTypingTarget(event.target)) return;
      const isAxisKey = [
        "KeyW",
        "KeyA",
        "KeyS",
        "KeyD",
        "ArrowUp",
        "ArrowDown",
        "ArrowLeft",
        "ArrowRight",
      ].includes(event.code);
      const button = KEY_TO_BUTTON[event.code];
      if (!isAxisKey && button === undefined) return;
      event.preventDefault();
      setLastDevice("keyboard");
      if (isAxisKey) keys.add(event.code);
      if (button !== undefined && !event.repeat) {
        buttonsRef.current[button] = true;
        handlersRef.current.onButtonDown(button);
      }
    };

    const onKeyUp = (event: KeyboardEvent) => {
      const button = KEY_TO_BUTTON[event.code];
      keys.delete(event.code);
      if (button !== undefined) {
        buttonsRef.current[button] = false;
        handlersRef.current.onButtonUp(button);
      }
    };

    const clearKeys = () => keys.clear();
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", clearKeys);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", clearKeys);
    };
  }, []);

  useEffect(() => {
    let raf = 0;
    let wasConnected = false;
    const previousButtons = Array(18).fill(false) as boolean[];

    const poll = () => {
      const pads = navigator.getGamepads?.() ?? [];
      const pad = Array.from(pads).find(Boolean) ?? null;
      const isConnected = Boolean(pad);
      if (isConnected !== wasConnected) {
        wasConnected = isConnected;
        setConnected(isConnected);
      }

      const keys = keyboardRef.current;
      const keyboardAxes = {
        lx: (keys.has("KeyD") ? 1 : 0) - (keys.has("KeyA") ? 1 : 0),
        ly: (keys.has("KeyS") ? 1 : 0) - (keys.has("KeyW") ? 1 : 0),
        rx:
          (keys.has("ArrowRight") ? 1 : 0) -
          (keys.has("ArrowLeft") ? 1 : 0),
        ry:
          (keys.has("ArrowDown") ? 1 : 0) -
          (keys.has("ArrowUp") ? 1 : 0),
      };

      const gamepadAxes = pad
        ? {
            lx: cleanAxis(pad.axes[0]),
            ly: cleanAxis(pad.axes[1]),
            rx: cleanAxis(pad.axes[2]),
            ry: cleanAxis(pad.axes[3]),
          }
        : { lx: 0, ly: 0, rx: 0, ry: 0 };
      const gamepadActive = Object.values(gamepadAxes).some(
        (value) => Math.abs(value) > 0,
      );
      if (gamepadActive) setLastDevice("gamepad");
      axesRef.current = gamepadActive ? gamepadAxes : keyboardAxes;

      for (let index = 0; index < previousButtons.length; index += 1) {
        const pressed = Boolean(pad?.buttons[index]?.pressed);
        buttonsRef.current[index] = pressed;
        if (pressed !== previousButtons[index]) {
          previousButtons[index] = pressed;
          if (pressed) {
            setLastDevice("gamepad");
            handlersRef.current.onButtonDown(index);
          } else {
            handlersRef.current.onButtonUp(index);
          }
        }
      }
      raf = requestAnimationFrame(poll);
    };

    raf = requestAnimationFrame(poll);
    return () => cancelAnimationFrame(raf);
  }, []);

  return { axesRef, buttonsRef, connected, lastDevice };
}
