"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export type ControllerAxes = {
  lx: number;
  ly: number;
  rx: number;
  ry: number;
};

export type InputMethod = "gamepad" | "keyboard";

type ControllerOptions = {
  onButtonDown: (button: number) => void;
  onButtonUp: (button: number) => void;
};

const DEAD_ZONE = 0.16;
const MOUSE_LOOK_GAIN = 0.11;
const MOUSE_LOOK_DECAY = 0.84;
const MOUSE_LOOK_EPSILON = 0.008;
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
  const mouseLookRef = useRef({ rx: 0, ry: 0 });
  const handlersRef = useRef({ onButtonDown, onButtonUp });
  const [connected, setConnected] = useState(false);
  const [lastDevice, setLastDevice] = useState<InputMethod>("keyboard");
  const [pointerLocked, setPointerLocked] = useState(false);

  handlersRef.current = { onButtonDown, onButtonUp };

  const applyMouseLook = useCallback((movementX: number, movementY: number) => {
    if ((!movementX && !movementY) || !Number.isFinite(movementX + movementY)) {
      return;
    }
    const mouse = mouseLookRef.current;
    mouse.rx = Math.max(
      -1,
      Math.min(1, mouse.rx * 0.35 + movementX * MOUSE_LOOK_GAIN),
    );
    mouse.ry = Math.max(
      -1,
      Math.min(1, mouse.ry * 0.35 + movementY * MOUSE_LOOK_GAIN),
    );
    setLastDevice("keyboard");
  }, []);

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
    const onPointerLockChange = () => {
      const locked = Boolean(document.pointerLockElement);
      setPointerLocked(locked);
      if (locked) {
        setLastDevice("keyboard");
      } else {
        mouseLookRef.current.rx = 0;
        mouseLookRef.current.ry = 0;
      }
    };

    const onMouseMove = (event: MouseEvent) => {
      if (!document.pointerLockElement) return;
      applyMouseLook(event.movementX, event.movementY);
    };

    document.addEventListener("pointerlockchange", onPointerLockChange);
    document.addEventListener("mousemove", onMouseMove);
    return () => {
      document.removeEventListener("pointerlockchange", onPointerLockChange);
      document.removeEventListener("mousemove", onMouseMove);
    };
  }, [applyMouseLook]);

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
      const mouse = mouseLookRef.current;
      const arrowLook = {
        rx:
          (keys.has("ArrowRight") ? 1 : 0) -
          (keys.has("ArrowLeft") ? 1 : 0),
        ry:
          (keys.has("ArrowDown") ? 1 : 0) -
          (keys.has("ArrowUp") ? 1 : 0),
      };
      const keyboardAxes = {
        lx: (keys.has("KeyD") ? 1 : 0) - (keys.has("KeyA") ? 1 : 0),
        ly: (keys.has("KeyS") ? 1 : 0) - (keys.has("KeyW") ? 1 : 0),
        rx: arrowLook.rx || mouse.rx,
        ry: arrowLook.ry || mouse.ry,
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

      mouse.rx *= MOUSE_LOOK_DECAY;
      mouse.ry *= MOUSE_LOOK_DECAY;
      if (Math.abs(mouse.rx) < MOUSE_LOOK_EPSILON) mouse.rx = 0;
      if (Math.abs(mouse.ry) < MOUSE_LOOK_EPSILON) mouse.ry = 0;

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

  return {
    axesRef,
    buttonsRef,
    connected,
    keysRef: keyboardRef,
    lastDevice,
    pointerLocked,
    applyMouseLook,
  };
}
