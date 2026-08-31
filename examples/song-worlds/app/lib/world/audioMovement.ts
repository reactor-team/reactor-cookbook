"use client";

/**
 * User camera input for the Lingbot world.
 *
 * MOVEMENT IS USER-DRIVEN ONLY — the audio never moves or turns the
 * camera. Camera ANGLE is driven by the MOUSE (pointer-lock look); WASD
 * drives translation. The song's beats/downbeats drive activity in the
 * GENERATED WORLD (downbeat prompt pulses in worldSession) and in the
 * effects overlay — never the camera.
 *
 * The camera is driven via Lingbot's setCameraPose (see worldSession):
 * an all-zero pose LOCKS the viewpoint, so whenever look input resolves to
 * idle the camera holds perfectly still and cannot drift on its own — the
 * mouse only turns it while it's actively moving.
 *
 * `resolveUserCamera` turns the shared input state (written by the stage's
 * mouse/key handlers) into a CameraIntent each playback tick.
 */

import { USER_LOOK_ROTATION_DEG } from "./config";

export interface CameraIntent {
  /** Forward(+1)/back(-1)/idle(0) translation (WASD). */
  lon: -1 | 0 | 1;
  /** Strafe right(+1)/left(-1)/idle(0) (WASD). */
  lat: -1 | 0 | 1;
  /** Look right(+1)/left(-1)/idle(0) (mouse). */
  lookH: -1 | 0 | 1;
  /** Look up(+1)/down(-1)/idle(0) (mouse). */
  lookV: -1 | 0 | 1;
  /** Camera rotation rate, deg/frame (Lingbot range 0..30). */
  rotationSpeed: number;
}

/** Shared, mutable input state written by the stage's mouse/key handlers
 *  and read by the playback loop each tick. */
export interface CameraInputState {
  /** Look direction from mouse-X this move (right +1 / left -1). */
  lookH: -1 | 0 | 1;
  /** Look direction from mouse-Y this move (up +1 / down -1). */
  lookV: -1 | 0 | 1;
  /** Mouse-look expires shortly after the last real move — set by the
   *  stage's mousemove handler. Once it lapses, look resolves to idle and
   *  the camera pose locks. */
  lookActiveUntil: number;
  /** Translation from W/S (held while the key is down). */
  moveLon: -1 | 0 | 1;
  /** Strafe from A/D (held while the key is down). */
  moveLat: -1 | 0 | 1;
}

export function createCameraInputState(): CameraInputState {
  return { lookH: 0, lookV: 0, lookActiveUntil: 0, moveLon: 0, moveLat: 0 };
}

/**
 * The camera intent is PURELY the user's input — translation from WASD
 * (held), look from the MOUSE (active only within its brief post-move
 * window). No audio term: when the mouse isn't moving, look resolves to
 * idle → the camera pose locks and it cannot drift on its own.
 */
export function resolveUserCamera(
  input: CameraInputState,
  now: number,
): CameraIntent {
  const looking =
    now < input.lookActiveUntil && (input.lookH !== 0 || input.lookV !== 0);
  return {
    lon: input.moveLon,
    lat: input.moveLat,
    lookH: looking ? input.lookH : 0,
    lookV: looking ? input.lookV : 0,
    rotationSpeed: looking ? USER_LOOK_ROTATION_DEG : 0,
  };
}
