"use client";

import { MutableRefObject, useEffect, useRef } from "react";
import type { ControllerAxes } from "@/hooks/useControllerInput";

type GamepadViewerOverlayProps = {
  axesRef: MutableRefObject<ControllerAxes>;
  buttonsRef: MutableRefObject<boolean[]>;
  connected: boolean;
  mode: "room" | "cabinet" | "booting" | "game";
};

const FACE_BUTTONS = [0, 1, 2, 3] as const;
const DPAD_BUTTONS = [12, 13, 14, 15] as const;

export function GamepadViewerOverlay({
  axesRef,
  buttonsRef,
  connected,
  mode,
}: GamepadViewerOverlayProps) {
  const buttonRefs = useRef<Array<HTMLSpanElement | null>>([]);
  const leftStickRef = useRef<HTMLSpanElement>(null);
  const rightStickRef = useRef<HTMLSpanElement>(null);
  const leftTriggerRef = useRef<HTMLSpanElement>(null);
  const rightTriggerRef = useRef<HTMLSpanElement>(null);

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
      frame = requestAnimationFrame(renderInput);
    };

    frame = requestAnimationFrame(renderInput);
    return () => cancelAnimationFrame(frame);
  }, [axesRef, buttonsRef]);

  const bindButton = (index: number) => (element: HTMLSpanElement | null) => {
    buttonRefs.current[index] = element;
  };

  return (
    <aside className={`input-overlay input-overlay-${mode}`} aria-label="Live controller input">
      <div className="input-overlay-caption">
        <span>INPUT VIEW</span>
        <span>{connected ? "XBOX PAD 01" : "KEYBOARD MAP"}</span>
      </div>
      <div className="input-overlay-viewport">
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
      </div>
    </aside>
  );
}
