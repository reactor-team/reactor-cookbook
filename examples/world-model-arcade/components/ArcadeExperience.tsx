"use client";

import {
  LingbotWorld2MainVideoView,
  LingbotWorld2Provider,
  useLingbotWorld2,
  useLingbotWorld2Message,
} from "@reactor-models/lingbot-world-2";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { ArcadeScene } from "@/components/ArcadeScene";
import { GamepadViewerOverlay } from "@/components/GamepadViewerOverlay";
import {
  useControllerInput,
  type InputMethod,
} from "@/hooks/useControllerInput";
import {
  captureWorldFrame,
  compareFrameSignatures,
  signatureFromImageBlob,
  type FrameSignature,
} from "@/lib/consistency";
import {
  composeGamePrompt,
  GAMES,
  nextGameIndex,
  type ArcadeGame,
  type FaceButton,
  type MovementDirection,
} from "@/lib/games";

type Phase = "room" | "cabinet" | "booting" | "game";
type BootState =
  | "idle"
  | "connecting"
  | "uploading"
  | "conditioning"
  | "starting"
  | "error";

const API_URL =
  process.env.NEXT_PUBLIC_COORDINATOR_URL ?? "https://api.reactor.inc";
const FACE_BY_INDEX: Partial<Record<number, FaceButton>> = {
  0: "a",
  1: "b",
  2: "x",
  3: "y",
};
const CHUNK_LATENTS = 3;
const FRAME_AUDIT_EVERY = 3;
const WORLD_CAMERA_ROTATION_SPEED_DEG = 2.1;
const HUD_CAMERA_TURN_RATE = 0.042;
const HUD_SHIP_TURN_RATE = 0.034;

type AttentionWindow = "small" | "large";
type PromptStatus = "locked" | "queued" | "sending" | "accepted";

async function fetchToken(): Promise<string> {
  const response = await fetch("/api/reactor/token");
  const body = (await response.json().catch(() => ({}))) as {
    jwt?: string;
    error?: string;
  };
  if (!response.ok || !body.jwt) {
    throw new Error(body.error ?? `Token request failed (${response.status})`);
  }
  return body.jwt;
}

function waitFor(
  predicate: () => boolean,
  timeoutMs: number,
  timeoutMessage: string,
) {
  return new Promise<void>((resolve, reject) => {
    const started = Date.now();
    const timer = window.setInterval(() => {
      if (predicate()) {
        window.clearInterval(timer);
        resolve();
      } else if (Date.now() - started > timeoutMs) {
        window.clearInterval(timer);
        reject(new Error(timeoutMessage));
      }
    }, 120);
  });
}

function XboxGlyph({ button }: { button: FaceButton | "lb" | "rb" | "view" | "menu" | "ls" | "rs" }) {
  return <span className={`xbox-glyph xbox-${button}`}>{button.toUpperCase()}</span>;
}

function KeyboardGlyph({ label }: { label: string }) {
  return (
    <span className={`keyboard-glyph ${label.length > 2 ? "is-wide" : ""}`}>
      {label}
    </span>
  );
}

function RoomOverlay({
  inputMethod,
  near,
  onInteract,
}: {
  inputMethod: InputMethod;
  near: boolean;
  onInteract: () => void;
}) {
  return (
    <div className="room-overlay">
      <section className="room-intro">
        <p className="system-label">WORLD MODEL ARCADE</p>
        <h1>Pick a world.</h1>
        <p>Walk up to the cabinet. Your controls carry across every game.</p>
      </section>
      <div className={`focus-reticle ${near ? "is-near" : ""}`} aria-hidden="true">
        <span />
      </div>
      <button
        className="cabinet-interaction-hotspot"
        onClick={onInteract}
        disabled={!near}
        aria-label="Use arcade cabinet"
      >
        Use arcade cabinet
      </button>
      <div className="room-controls">
        {inputMethod === "keyboard" ? (
          <>
            <span><KeyboardGlyph label="WASD" /> MOVE</span>
            <span><KeyboardGlyph label="MOUSE" /> LOOK</span>
          </>
        ) : (
          <>
            <span><XboxGlyph button="ls" /> MOVE</span>
            <span><XboxGlyph button="rs" /> LOOK</span>
          </>
        )}
      </div>
    </div>
  );
}

function CabinetOverlay({
  selectedIndex,
  onSelect,
  onLaunch,
  onPreview,
  onBack,
}: {
  selectedIndex: number;
  onSelect: (index: number) => void;
  onLaunch: () => void;
  onPreview: () => void;
  onBack: () => void;
}) {
  return (
    <div className="cabinet-overlay">
      <section className="cabinet-screen-controls" aria-label="Arcade game selector">
        <button className="screen-hit screen-hit-launch" onClick={onLaunch}>
          Enter world
        </button>
        <button className="screen-hit screen-hit-preview" onClick={onPreview}>
          Preview HUD
        </button>
        <nav
          className="screen-game-hit-list"
          aria-label="Games"
          style={{ gridTemplateRows: `repeat(${GAMES.length}, 1fr)` }}
        >
          {GAMES.map((item, index) => (
            <button
              key={item.id}
              className={`screen-hit ${index === selectedIndex ? "is-selected" : ""}`}
              onClick={() => onSelect(index)}
              aria-current={index === selectedIndex ? "true" : undefined}
            >
              {item.title}
            </button>
          ))}
        </nav>
        <button className="screen-hit screen-hit-back" onClick={onBack}>
          Return to room
        </button>
      </section>
    </div>
  );
}

function BootOverlay({
  game,
  inputMethod,
  state,
  onCancel,
  onRetry,
  onPreview,
}: {
  game: ArcadeGame;
  inputMethod: InputMethod;
  state: BootState;
  onCancel: () => void;
  onRetry: () => void;
  onPreview: () => void;
}) {
  const copy: Record<Exclude<BootState, "idle" | "error">, string> = {
    connecting: "WAKING THE MACHINE",
    uploading: "LOADING YOUR WORLD",
    conditioning: "SETTING THE SCENE",
    starting: "ALMOST THERE",
  };
  return (
    <div className="boot-overlay">
      <section className={`boot-module ${state === "error" ? "has-error" : ""}`}>
        <img src={game.image} alt={`${game.title} reference frame`} />
        <div className="boot-copy">
          <span className="system-label">{game.title}</span>
          {state === "error" ? (
            <>
              <h2>THE WORLD DIDN&apos;T OPEN</h2>
              <p>Give it another try, or preview the controls while you wait.</p>
              <div className="boot-actions">
                <button onClick={onRetry}>
                  {inputMethod === "keyboard" ? <KeyboardGlyph label="ENTER" /> : <XboxGlyph button="a" />}
                  RETRY
                </button>
                <button onClick={onPreview}>
                  {inputMethod === "keyboard" ? <KeyboardGlyph label="3" /> : <XboxGlyph button="x" />}
                  LOCAL PREVIEW
                </button>
              </div>
            </>
          ) : (
            <>
              <h2>{state === "idle" ? "GETTING READY" : copy[state]}</h2>
              <p>{game.title} will be ready in a moment.</p>
              <div className="condition-line"><span /></div>
            </>
          )}
        </div>
      </section>
      <button className="cancel-boot" onClick={onCancel}>
        {inputMethod === "keyboard" ? <KeyboardGlyph label="ESC" /> : <XboxGlyph button="view" />}
        CANCEL
      </button>
    </div>
  );
}

function QuickActions({
  game,
  active,
  inputMethod,
  onDown,
  onUp,
}: {
  game: ArcadeGame;
  active: FaceButton[];
  inputMethod: InputMethod;
  onDown: (button: FaceButton) => void;
  onUp: (button: FaceButton) => void;
}) {
  return (
    <div className="quick-actions" aria-label="Quick actions">
      {(["y", "x", "b", "a"] as FaceButton[]).map((button) => (
        <button
          key={button}
          className={`quick-action action-${button} ${active.includes(button) ? "is-active" : ""}`}
          onPointerDown={() => onDown(button)}
          onPointerUp={() => onUp(button)}
          onPointerLeave={() => onUp(button)}
        >
          {inputMethod === "keyboard" ? (
            <KeyboardGlyph label={String(({ a: 1, b: 2, x: 3, y: 4 } as const)[button])} />
          ) : (
            <XboxGlyph button={button} />
          )}
          <span>{game.actions[button].label}</span>
        </button>
      ))}
    </div>
  );
}

function GameSpecificHud({ game }: { game: ArcadeGame }) {
  if (game.id === "good-dog-sf") {
    return (
      <>
        <div className="scent-compass"><span>SCENT</span><b>COFFEE + SEA AIR</b><small data-scent-distance>24 M · STRONGER</small><i /></div>
        <div className="scent-marker"><span /><b data-scent-state>STRONGER</b></div>
        <div className="dog-energy"><small>ZOOM</small><strong data-zoom>READY</strong></div>
      </>
    );
  }
  if (game.id === "windward") {
    return (
      <>
        <div className="sail-nav"><span>TRUE WIND</span><b data-wind>WNW · 18 KT</b><small data-passage>PASSAGE 2.4 NM</small></div>
        <div className="sail-helm"><i /><b data-course>284°</b><span>COURSE</span></div>
        <div className="hud-readout readout-left"><small>HEEL</small><strong data-heel>4°</strong><span data-sail-state>SAILS DRAWING</span></div>
        <div className="hud-readout readout-right"><small>SPEED</small><strong data-speed>5.6 KT</strong><span data-keel-depth>DEPTH 26 M</span></div>
      </>
    );
  }
  if (game.id === "dustline") {
    return (
      <>
        <div className="rally-roadbook"><span>RIDGE MARKER</span><b data-rally-distance>2.8 KM</b><small data-rally-line>FAST LINE: LEFT WASH</small></div>
        <div className="rally-gate"><i /><b data-checkpoint>CHECKPOINT</b></div>
        <div className="hud-readout readout-left"><small>GEAR</small><strong data-gear>2</strong><span data-rpm>2,800 RPM</span></div>
        <div className="hud-readout readout-right"><small>SPEED</small><strong data-speed>18 MPH</strong><span>SUSPENSION OK</span></div>
      </>
    );
  }
  if (game.id === "deep-signal") {
    return (
      <>
        <div className="dive-contact"><span>CONTACT</span><b data-contact>RELAY CORE</b><small data-bearing>BEARING 018</small></div>
        <div className="sonar-scope" aria-hidden="true"><i /><b /><span /></div>
        <div className="depth-ladder"><i /><i /><i /><i /><b data-depth>4,280 M</b></div>
        <div className="hud-readout readout-right"><small>DEPTH</small><strong data-depth>4,280 M</strong><span data-hull>HULL 92%</span></div>
      </>
    );
  }
  if (game.id === "blue-mesa") {
    return (
      <>
        <div className="kayak-route"><span>ARCH ISLAND</span><b data-kayak-distance>0.90 NM</b><small data-kayak-channel>OPEN CHANNEL</small></div>
        <div className="island-chart" aria-hidden="true"><i /><i /><i /><i /><i /><b /><span /></div>
        <div className="kayak-heading"><span>HEADING</span><b data-kayak-heading>072°</b><small data-kayak-state>EASY DRIFT</small></div>
        <div className="hud-readout readout-left"><small>STROKE</small><strong data-cadence>18 SPM</strong><span data-stroke-state>RHYTHM EVEN</span></div>
        <div className="hud-readout readout-right"><small>SPEED</small><strong data-speed>2.4 KT</strong><span data-water-depth>DEPTH 11 M</span></div>
      </>
    );
  }
  if (game.id === "free-range") {
    return (
      <>
        <div className="field-route"><span>RED GATE</span><b data-gate-distance>96 M</b><small data-field-line>MEADOW TRACK</small></div>
        <div className="field-gate" aria-hidden="true"><i /><b data-field-state>ON COURSE</b></div>
        <div className="hud-readout readout-left"><small>GAIT</small><strong data-chicken-speed>0.0 KPH</strong><span data-gait-state>WATCHFUL</span></div>
        <div className="hud-readout readout-right"><small>EGGS</small><strong data-egg-count>00</strong><span data-egg-state>NONE LAID</span></div>
      </>
    );
  }
  return (
    <>
      <div className="arch-route"><span>NEXT GATE</span><b>ARCH 03</b><small data-arch-distance>0.8 NM · CLEAR</small></div>
      <div className="arch-reticle"><i /><b>03</b><span data-gate-state>NEXT ARCH</span></div>
      <div className="flight-bank"><span>BANK</span><b data-bank>0°</b><small data-bank-state>WINGS LEVEL</small></div>
      <div className="hud-readout readout-left"><small>ALTITUDE</small><strong data-altitude>620 FT</strong><span data-floor-clearance>CANYON FLOOR +310</span></div>
      <div className="hud-readout readout-right"><small>AIRSPEED</small><strong data-speed>72 MPH</strong><span data-engine>ENGINE 68%</span></div>
    </>
  );
}

function GameHud({
  game,
  chunkIndex,
  runChunk,
  score,
  coherence,
  attentionWindow,
  promptStatus,
  lastActiveAction,
  debugVisible,
  paused,
  localPreview,
  activeButtons,
  inputMethod,
  axesRef,
  buttonsRef,
  resetToken,
  onActionDown,
  onActionUp,
  onExit,
  onReset,
  onToggleDebug,
}: {
  game: ArcadeGame;
  chunkIndex: number;
  runChunk: number;
  score: number;
  coherence: number;
  attentionWindow: AttentionWindow;
  promptStatus: PromptStatus;
  lastActiveAction: string;
  debugVisible: boolean;
  paused: boolean;
  localPreview: boolean;
  activeButtons: FaceButton[];
  inputMethod: InputMethod;
  axesRef: ReturnType<typeof useControllerInput>["axesRef"];
  buttonsRef: ReturnType<typeof useControllerInput>["buttonsRef"];
  resetToken: number;
  onActionDown: (button: FaceButton) => void;
  onActionUp: (button: FaceButton) => void;
  onExit: () => void;
  onReset: () => void;
  onToggleDebug: () => void;
}) {
  const hudRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    let raf = 0;
    let cameraHeading = 0;
    let shipHeading = 284;
    let previousFrame = performance.now();
    let telemetrySpeed =
      game.id === "windward"
        ? 5.6
        : game.id === "dustline"
          ? 18
          : game.id === "arch-runner"
            ? 72
            : game.id === "blue-mesa"
              ? 2.4
              : game.id === "free-range"
                ? 0
              : 0;
    let passageDistance = 2.4;
    let rallyDistance = 2.8;
    let scentDistance = 24;
    let depth = 4280;
    let altitude = 620;
    let archDistance = 0.8;
    let kayakHeading = 72;
    let islandDistance = 0.9;
    let zoomCharge = 1;
    let gateDistance = 96;
    let eggCount = 0;
    let eggWasPressed = false;

    const clamp = (value: number, min: number, max: number) =>
      Math.min(max, Math.max(min, value));
    const setText = (root: HTMLDivElement, selector: string, value: string) => {
      const node = root.querySelector<HTMLElement>(selector);
      if (node) node.textContent = value;
    };

    const tick = () => {
      const root = hudRef.current;
      if (root) {
        const now = performance.now();
        const elapsed = Math.min(40, now - previousFrame);
        const dt = elapsed / 1000;
        previousFrame = now;
        const { lx, ly, rx, ry } = axesRef.current;
        const forward = clamp(-ly, -1, 1);
        const aPressed = Boolean(buttonsRef.current[0]);
        const bPressed = Boolean(buttonsRef.current[1]);
        const xPressed = Boolean(buttonsRef.current[2]);
        const yPressed = Boolean(buttonsRef.current[3]);

        root.style.setProperty("--stick-x", lx.toFixed(3));
        root.style.setProperty("--stick-y", ly.toFixed(3));
        root.style.setProperty("--aim-x", `${(lx * 34 + rx * 18).toFixed(2)}px`);
        root.style.setProperty("--aim-y", `${(ly * 22 + ry * 12).toFixed(2)}px`);
        root.style.setProperty("--helm-turn", `${((game.id === "windward" ? rx : lx) * 32).toFixed(2)}deg`);
        root.style.setProperty("--rally-roll", `${(lx * 7).toFixed(2)}deg`);

        cameraHeading =
          (cameraHeading + rx * elapsed * HUD_CAMERA_TURN_RATE + 360) % 360;
        root.style.setProperty("--compass-shift", `${(-cameraHeading * 2.2).toFixed(2)}px`);
        setText(root, "[data-camera-heading]", String(Math.round(cameraHeading)).padStart(3, "0"));
        const cardinalNode = root.querySelector<HTMLElement>("[data-camera-cardinal]");
        if (cardinalNode) {
          cardinalNode.textContent = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][
            Math.round(cameraHeading / 45) % 8
          ];
        }

        if (game.id === "good-dog-sf") {
          scentDistance = clamp(
            scentDistance - Math.max(0, forward) * dt * 2.4 + Math.max(0, -forward) * dt * 0.7,
            2.5,
            36,
          );
          const scentStrength = clamp(1 - scentDistance / 38, 0.16, 1);
          zoomCharge = clamp(zoomCharge + (yPressed ? -dt * 1.6 : dt * 0.24), 0, 1);
          root.style.setProperty("--scent-x", `${(lx * 38 + rx * 24).toFixed(2)}px`);
          root.style.setProperty("--scent-y", `${(ly * 18 + ry * 12).toFixed(2)}px`);
          root.style.setProperty("--scent-arrow-x", `${((lx * 38 + rx * 24) * 0.24).toFixed(2)}px`);
          root.style.setProperty("--scent-scale", (0.82 + scentStrength * 0.38).toFixed(3));
          root.style.setProperty("--scent-alpha", (0.42 + scentStrength * 0.58).toFixed(3));
          root.style.setProperty("--scent-glow", `${(8 + scentStrength * 18).toFixed(1)}px`);
          root.style.setProperty("--zoom-charge", `${(zoomCharge * 100).toFixed(1)}%`);
          setText(root, "[data-scent-distance]", `${Math.round(scentDistance)} M · ${scentStrength > 0.8 ? "HOT" : scentStrength > 0.48 ? "STRONGER" : "FAINT"}`);
          setText(root, "[data-scent-state]", aPressed ? "AIRBORNE" : scentStrength > 0.8 ? "VERY CLOSE" : "STRONGER");
          setText(root, "[data-zoom]", yPressed ? "ACTIVE" : zoomCharge > 0.82 ? "READY" : `${Math.round(zoomCharge * 100)}%`);
        } else if (game.id === "windward") {
          const targetSpeed = clamp(5.2 + forward * 4.2 + (aPressed ? 2.6 : 0) - (xPressed ? 5 : 0), 0.4, 12.8);
          telemetrySpeed += (targetSpeed - telemetrySpeed) * Math.min(1, dt * 1.8);
          shipHeading = (shipHeading + rx * elapsed * HUD_SHIP_TURN_RATE + 360) % 360;
          passageDistance = Math.max(0.1, passageDistance - telemetrySpeed * dt / 520);
          const heel = Math.round(Math.abs(lx) * 20 + telemetrySpeed * 0.45);
          root.style.setProperty("--sail-brightness", `${(0.82 + clamp(telemetrySpeed / 12.8, 0, 1) * 0.42).toFixed(3)}`);
          setText(root, "[data-course]", `${String(Math.round(shipHeading)).padStart(3, "0")}°`);
          setText(root, "[data-heel]", `${heel}°`);
          setText(root, "[data-speed]", `${telemetrySpeed.toFixed(1)} KT`);
          setText(root, "[data-passage]", `PASSAGE ${passageDistance.toFixed(2)} NM`);
          setText(root, "[data-wind]", `${lx < -0.35 ? "W" : lx > 0.35 ? "NW" : "WNW"} · ${Math.round(16 + Math.abs(rx) * 8)} KT`);
          setText(root, "[data-sail-state]", xPressed ? "ANCHOR RUNNING" : aPressed ? "FULL DRAW" : Math.abs(lx) > 0.55 ? "TURNING" : "SAILS DRAWING");
          setText(root, "[data-keel-depth]", `DEPTH ${Math.round(26 - ry * 3)} M`);
        } else if (game.id === "dustline") {
          const targetSpeed = clamp(18 + Math.max(0, forward) * 74 - Math.max(0, -forward) * 15 + (aPressed ? 30 : 0) - (xPressed ? 20 : 0), 3, 124);
          telemetrySpeed += (targetSpeed - telemetrySpeed) * Math.min(1, dt * (bPressed ? 4.5 : 2.4));
          rallyDistance = Math.max(0.1, rallyDistance - telemetrySpeed * dt / 700);
          const gear = clamp(Math.floor(telemetrySpeed / 22) + 1, 1, 6);
          const rpm = Math.round(1900 + (telemetrySpeed % 22) * 245 + (aPressed ? 900 : 0));
          root.style.setProperty("--rally-alpha", `${(0.66 + clamp(telemetrySpeed / 124, 0, 1) * 0.34).toFixed(3)}`);
          setText(root, "[data-speed]", `${Math.round(telemetrySpeed)} MPH`);
          setText(root, "[data-gear]", String(gear));
          setText(root, "[data-rpm]", `${rpm.toLocaleString()} RPM`);
          setText(root, "[data-rally-distance]", `${rallyDistance.toFixed(2)} KM`);
          setText(root, "[data-rally-line]", `FAST LINE: ${lx < -0.28 ? "LEFT WASH" : lx > 0.28 ? "RIGHT RIDGE" : "CENTER CUT"}`);
          setText(root, "[data-checkpoint]", bPressed ? "DRIFTING" : aPressed ? "BOOST" : "CHECKPOINT");
        } else if (game.id === "deep-signal") {
          const ballastRate = bPressed ? 34 : 0;
          depth = clamp(depth + (ry * 18 + ballastRate + forward * 2.5) * dt, 4100, 4680);
          const bearing = (18 + cameraHeading + lx * 24 + 360) % 360;
          root.style.setProperty("--depth-shift", `${((depth - 4280) * 0.42).toFixed(2)}px`);
          root.style.setProperty("--sonar-x", `${(rx * 24 + lx * 11).toFixed(2)}px`);
          root.style.setProperty("--sonar-y", `${(ry * 22 + ly * 8).toFixed(2)}px`);
          root.style.setProperty("--sonar-echo-x", `${((rx * 24 + lx * 11) * -0.4).toFixed(2)}px`);
          root.style.setProperty("--sonar-echo-y", `${((ry * 22 + ly * 8) * -0.35).toFixed(2)}px`);
          root.style.setProperty("--sonar-duration", aPressed ? "0.62s" : "2.8s");
          root.style.setProperty("--sonar-glow-size", aPressed ? "34px" : "19px");
          root.style.setProperty("--sonar-glow-alpha", aPressed ? "0.82" : "0.3");
          root.style.setProperty("--sonar-brightness", yPressed ? "1.26" : "0.94");
          root.querySelectorAll<HTMLElement>("[data-depth]").forEach((node) => {
            node.textContent = `${Math.round(depth).toLocaleString()} M`;
          });
          setText(root, "[data-bearing]", `BEARING ${String(Math.round(bearing)).padStart(3, "0")}`);
          setText(root, "[data-contact]", aPressed ? "PING RETURN" : xPressed ? "ARM EXTENDED" : "RELAY CORE");
          setText(root, "[data-hull]", `HULL ${Math.round(92 - Math.max(0, depth - 4380) * 0.015)}%`);
        } else if (game.id === "blue-mesa") {
          const targetSpeed = clamp(
            1.8 + Math.max(0, forward) * 3.2 + (aPressed ? 1.7 : 0),
            0.25,
            6.4,
          );
          telemetrySpeed +=
            (targetSpeed - telemetrySpeed) * Math.min(1, dt * 1.9);
          kayakHeading =
            (kayakHeading + lx * elapsed * 0.052 + 360) % 360;
          islandDistance = Math.max(
            0.03,
            islandDistance - Math.max(0, telemetrySpeed) * dt / 420,
          );
          const paddlingInput = Math.abs(forward) > 0.12 || Math.abs(lx) > 0.12;
          const cadence = paddlingInput || aPressed
            ? Math.round(14 + Math.abs(forward) * 24 + Math.abs(lx) * 8 + (aPressed ? 18 : 0))
            : 0;
          const waterDepth = Math.round(
            11 + Math.sin(now * (xPressed ? 0.0024 : 0.00022)) * (xPressed ? 4 : 2) + ry * 2,
          );
          root.style.setProperty("--kayak-heading", `${kayakHeading.toFixed(2)}deg`);
          root.style.setProperty("--kayak-route-x", `${(lx * 23 + rx * 9).toFixed(2)}px`);
          root.style.setProperty("--kayak-route-y", `${(ly * 11 + ry * 5).toFixed(2)}px`);
          root.style.setProperty("--island-pulse-scale", yPressed ? "1.24" : "1");
          root.style.setProperty("--island-pulse-alpha", yPressed ? "0.94" : "0.48");
          root.style.setProperty(
            "--kayak-chart-brightness",
            (0.88 + clamp(telemetrySpeed / 6.4, 0, 1) * 0.28).toFixed(3),
          );
          setText(root, "[data-kayak-heading]", `${String(Math.round(kayakHeading)).padStart(3, "0")}°`);
          setText(root, "[data-kayak-distance]", `${islandDistance.toFixed(2)} NM`);
          setText(
            root,
            "[data-kayak-channel]",
            yPressed
              ? "DOLPHINS NEARBY"
              : bPressed
                ? "STORM FRONT"
                : xPressed
                  ? "CHOPPY WATER"
              : lx < -0.28
                ? "WEST ISLE CHANNEL"
                : lx > 0.28
                  ? "EAST REED CHANNEL"
                  : "OPEN CHANNEL",
          );
          setText(root, "[data-speed]", `${telemetrySpeed.toFixed(1)} KT`);
          setText(root, "[data-cadence]", `${cadence} SPM`);
          setText(root, "[data-water-depth]", `DEPTH ${waterDepth} M`);
          setText(
            root,
            "[data-stroke-state]",
            aPressed
                  ? "POWER RHYTHM"
                  : paddlingInput
                    ? "RHYTHM EVEN"
                    : "PADDLE RESTING",
          );
          setText(
            root,
            "[data-kayak-state]",
            yPressed
              ? "FINS SURFACING"
              : bPressed
                ? "HEAVY WEATHER"
                : xPressed
                  ? "SHORT CHOP"
                  : telemetrySpeed > 4.8
                    ? "FAST GLIDE"
                    : "EASY DRIFT",
          );
        } else if (game.id === "free-range") {
          const runningInput = Math.abs(forward) > 0.12 || Math.abs(lx) > 0.12;
          const targetSpeed = bPressed
            ? 0
            : runningInput || yPressed
              ? clamp(Math.abs(forward) * 13 + Math.abs(lx) * 4 + (yPressed ? 7 : 0), 2.5, 23)
              : 0;
          telemetrySpeed +=
            (targetSpeed - telemetrySpeed) * Math.min(1, dt * (targetSpeed > telemetrySpeed ? 5.2 : 7.4));
          gateDistance = Math.max(
            1,
            gateDistance - Math.max(0, forward) * (yPressed ? 3.2 : 2.1) * dt,
          );
          if (bPressed && !eggWasPressed) eggCount += 1;
          eggWasPressed = bPressed;
          root.style.setProperty("--field-gate-x", `${(lx * 42 + rx * 16).toFixed(2)}px`);
          root.style.setProperty("--field-gate-y", `${(ly * 10 + ry * 8).toFixed(2)}px`);
          root.style.setProperty("--field-gate-scale", aPressed ? "1.14" : yPressed ? "1.08" : "1");
          setText(root, "[data-gate-distance]", `${Math.round(gateDistance)} M`);
          setText(
            root,
            "[data-field-line]",
            bPressed
              ? "PAUSED TO LAY"
              : xPressed
                ? "SEEDS SPOTTED"
                : lx < -0.28
                  ? "LEFT GRASS LANE"
                  : lx > 0.28
                    ? "RIGHT GRASS LANE"
                    : "MEADOW TRACK",
          );
          setText(root, "[data-chicken-speed]", `${telemetrySpeed.toFixed(1)} KPH`);
          setText(
            root,
            "[data-gait-state]",
            aPressed
              ? "AIRBORNE"
              : bPressed
                ? "NESTING"
                : xPressed
                  ? "PECKING"
                  : yPressed
                    ? "WING SPRINT"
                    : runningInput
                      ? "RUNNING"
                      : "WATCHFUL",
          );
          setText(root, "[data-egg-count]", String(eggCount).padStart(2, "0"));
          setText(
            root,
            "[data-egg-state]",
            bPressed ? "EGG LAID" : eggCount > 0 ? `${eggCount} IN FIELD` : "NONE LAID",
          );
          setText(
            root,
            "[data-field-state]",
            aPressed
              ? "JUMP"
              : bPressed
                ? "LAYING EGG"
                : xPressed
                  ? "FORAGING"
                  : yPressed
                    ? "SPRINT"
                    : gateDistance < 18
                      ? "GATE CLOSE"
                      : "ON COURSE",
          );
        } else if (game.id === "arch-runner") {
          const targetSpeed = clamp(72 + forward * 46 + (aPressed ? 42 : 0), 38, 154) * (xPressed ? 0.48 : 1);
          telemetrySpeed += (targetSpeed - telemetrySpeed) * Math.min(1, dt * (xPressed ? 5 : 2.8));
          altitude = clamp(altitude + (ly * 92 + ry * 24) * dt, 260, 1180);
          archDistance = Math.max(0.05, archDistance - telemetrySpeed * dt / 880);
          const barrelRoll = bPressed ? Math.sin(now * 0.012) * 168 : 0;
          const bank = lx * 31 + barrelRoll;
          const clearance = Math.round(altitude - 310);
          root.style.setProperty("--flight-bank", `${bank.toFixed(2)}deg`);
          root.style.setProperty("--engine-brightness", `${(0.82 + clamp((telemetrySpeed - 38) / 116, 0, 1) * 0.44).toFixed(3)}`);
          setText(root, "[data-bank]", `${Math.round(bank)}°`);
          setText(root, "[data-bank-state]", bPressed ? "ROLLING" : Math.abs(bank) < 7 ? "WINGS LEVEL" : bank < 0 ? "BANK LEFT" : "BANK RIGHT");
          setText(root, "[data-altitude]", `${Math.round(altitude)} FT`);
          setText(root, "[data-floor-clearance]", `CANYON FLOOR +${clearance}`);
          setText(root, "[data-speed]", `${Math.round(telemetrySpeed)} MPH`);
          setText(root, "[data-engine]", xPressed ? "AIR BRAKE" : `ENGINE ${Math.round(48 + clamp(telemetrySpeed / 154, 0, 1) * 48)}%`);
          setText(root, "[data-arch-distance]", `${archDistance.toFixed(2)} NM · ${Math.abs(bank) > 34 ? "CORRECT" : "CLEAR"}`);
          setText(root, "[data-gate-state]", yPressed ? "ROUTE MARKED" : Math.abs(bank) > 34 ? "LEVEL WINGS" : "NEXT ARCH");
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [axesRef, buttonsRef, game.id, resetToken]);

  return (
    <div ref={hudRef} className={`game-hud hud-${game.id} ${paused ? "is-paused" : ""}`}>
      <div className="game-title-block">
        <span>{game.genre}</span>
        <h2>{game.title}</h2>
      </div>
      <div className="objective-block">
        <span>CURRENT OBJECTIVE</span>
        <strong>{game.objective}</strong>
      </div>
      <div className="world-telemetry">
        <span>{localPreview ? "REFERENCE FRAME" : `LIVE SESSION · ${String(runChunk).padStart(3, "0")}`}</span>
        <span>{localPreview ? "NO LIVE SESSION" : `${score.toLocaleString()} PTS`}</span>
      </div>
      <button className="coherence-chip" onClick={onToggleDebug}>
        <span>COHERENCE</span><strong>{coherence}%</strong><i style={{ width: `${coherence}%` }} />
      </button>
      <div className="run-progress is-live" aria-hidden="true"><i /></div>
      <div className="camera-compass" aria-label="Camera bearing">
        <span>CAMERA</span>
        <div className="camera-compass-scale"><i /><b data-camera-cardinal>N</b></div>
        <strong data-camera-heading>000</strong>
      </div>
      <GameSpecificHud game={game} />
      <div className="stick-help">
        <span>
          {inputMethod === "keyboard" ? <KeyboardGlyph label="WASD" /> : <XboxGlyph button="ls" />}
          MOVE
        </span>
        <span>
          {inputMethod === "keyboard" ? <KeyboardGlyph label="MOUSE" /> : <XboxGlyph button="rs" />}
          LOOK
        </span>
      </div>
      <QuickActions
        game={game}
        active={activeButtons}
        inputMethod={inputMethod}
        onDown={onActionDown}
        onUp={onActionUp}
      />
      <div className="game-system-controls">
        <button onClick={onToggleDebug}>
          {inputMethod === "keyboard" ? <KeyboardGlyph label="Q" /> : <XboxGlyph button="lb" />}
          DEBUG
        </button>
        <button onClick={onReset}>
          {inputMethod === "keyboard" ? <KeyboardGlyph label="P" /> : <XboxGlyph button="menu" />}
          RESTART
        </button>
        <button onClick={onExit}>
          {inputMethod === "keyboard" ? <KeyboardGlyph label="TAB" /> : <XboxGlyph button="view" />}
          ARCADE
        </button>
      </div>
      {debugVisible && (
        <aside className="consistency-debug">
          <span>CONSISTENCY DIRECTOR</span>
          <dl>
            <div><dt>SEED</dt><dd>{game.seed}</dd></div>
            <div><dt>MODEL CHUNK</dt><dd>{chunkIndex}</dd></div>
            <div><dt>SESSION CHUNK</dt><dd>{runChunk}</dd></div>
            <div><dt>FRAME SCORE</dt><dd>{coherence}%</dd></div>
            <div><dt>ATTENTION</dt><dd>{attentionWindow.toUpperCase()}</dd></div>
            <div><dt>PROMPT</dt><dd>{promptStatus.toUpperCase()}</dd></div>
            <div><dt>NATIVE ACTION</dt><dd>{lastActiveAction || "STILL"}</dd></div>
          </dl>
          <small>WORLD CONTRACT LOCKED · CONTINUOUS PLAY</small>
        </aside>
      )}
    </div>
  );
}

function ReactorArcade({ configured }: { configured: boolean }) {
  const appRef = useRef<HTMLElement>(null);
  const lw2 = useLingbotWorld2();
  const { status, sendCommand, uploadFile } = lw2;
  const [phase, setPhase] = useState<Phase>("room");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [nearCabinet, setNearCabinet] = useState(false);
  const [bootState, setBootState] = useState<BootState>("idle");
  const [bootError, setBootError] = useState<string | null>(null);
  const [activeButtons, setActiveButtons] = useState<FaceButton[]>([]);
  const [chunkIndex, setChunkIndex] = useState(0);
  const [hudResetToken, setHudResetToken] = useState(0);
  const [runChunk, setRunChunk] = useState(0);
  const [score, setScore] = useState(0);
  const [coherence, setCoherence] = useState(100);
  const [attentionWindow, setAttentionWindow] = useState<AttentionWindow>("small");
  const [promptStatus, setPromptStatus] = useState<PromptStatus>("locked");
  const [lastActiveAction, setLastActiveAction] = useState("still");
  const [debugVisible, setDebugVisible] = useState(false);
  const [paused, setPaused] = useState(false);
  const [localPreview, setLocalPreview] = useState(false);
  const [softMouseLook, setSoftMouseLook] = useState(false);

  const phaseRef = useRef<Phase>(phase);
  const statusRef = useRef(status);
  const pausedRef = useRef(false);
  const selectedIndexRef = useRef(selectedIndex);
  const currentGameRef = useRef<ArcadeGame>(GAMES[0]);
  const activeButtonsRef = useRef<FaceButton[]>([]);
  const movingRef = useRef(false);
  const localPreviewRef = useRef(false);
  const imageReadyRef = useRef<(() => void) | null>(null);
  const launchPendingRef = useRef(false);
  const resetPendingRef = useRef(false);
  const lastPromptRef = useRef("");
  const promptQueuedRef = useRef<string | null>(null);
  const promptSlotChunkRef = useRef(-1);
  const promptInFlightRef = useRef(false);
  const runChunkRef = useRef(0);
  const scoreRef = useRef(0);
  const coherenceRef = useRef(100);
  const canonicalSignatureRef = useRef<FrameSignature | null>(null);
  const auditPendingRef = useRef(false);
  const attentionWindowRef = useRef<AttentionWindow>("small");
  const auditFrameRef = useRef<() => void>(() => {});
  const lastMoveLongRef = useRef<"idle" | "forward" | "back">("idle");
  const lastMoveLatRef = useRef<"idle" | "strafe_left" | "strafe_right">("idle");
  const lastLookHorizontalRef = useRef<"idle" | "left" | "right">("idle");
  const lastLookVerticalRef = useRef<"idle" | "up" | "down">("idle");
  const poseSentRef = useRef(false);
  const sendPoseRef = useRef<() => void>(() => {});

  phaseRef.current = phase;
  statusRef.current = status;
  pausedRef.current = paused;
  selectedIndexRef.current = selectedIndex;
  localPreviewRef.current = localPreview;
  runChunkRef.current = runChunk;
  scoreRef.current = score;
  coherenceRef.current = coherence;

  const composeCurrentPrompt = useCallback(
    () => {
      const movementDirections: MovementDirection[] = [];
      if (lastMoveLongRef.current !== "idle") {
        movementDirections.push(lastMoveLongRef.current);
      }
      if (lastMoveLatRef.current !== "idle") {
        movementDirections.push(lastMoveLatRef.current);
      }
      return composeGamePrompt(
        currentGameRef.current,
        movingRef.current,
        activeButtonsRef.current,
        movementDirections,
      );
    },
    [],
  );

  const dispatchPromptForCurrentChunk = useCallback(
    (force = false) => {
      if (statusRef.current !== "ready" || localPreviewRef.current) return;
      const next = promptQueuedRef.current ?? composeCurrentPrompt();
      if (!force && next === lastPromptRef.current) {
        promptQueuedRef.current = null;
        setPromptStatus("locked");
        return;
      }
      if (
        !force &&
        (promptInFlightRef.current ||
          promptSlotChunkRef.current === runChunkRef.current)
      ) {
        promptQueuedRef.current = next;
        setPromptStatus("queued");
        return;
      }
      promptQueuedRef.current = null;
      promptSlotChunkRef.current = runChunkRef.current;
      promptInFlightRef.current = true;
      lastPromptRef.current = next;
      setPromptStatus("sending");
      lw2
        .setPrompt({ prompt: next })
        .then(() => setPromptStatus("accepted"))
        .catch((error) => {
          promptQueuedRef.current = next;
          setPromptStatus("queued");
          setBootError(error instanceof Error ? error.message : "Prompt update failed");
        })
        .finally(() => {
          promptInFlightRef.current = false;
        });
    },
    [composeCurrentPrompt, sendCommand],
  );

  const queueComposedPrompt = useCallback(
    (priority = false) => {
      promptQueuedRef.current = composeCurrentPrompt();
      setPromptStatus("queued");
      if (priority) dispatchPromptForCurrentChunk();
    },
    [composeCurrentPrompt, dispatchPromptForCurrentChunk],
  );

  const setAttentionMode = useCallback(
    (next: AttentionWindow, force = false) => {
      if (!force && attentionWindowRef.current === next) return;
      attentionWindowRef.current = next;
      setAttentionWindow(next);
      if (statusRef.current === "ready" && !localPreviewRef.current) {
        lw2.setAttnWindow({ attn_window: next }).catch(() => undefined);
      }
    },
    [sendCommand],
  );

  const auditCurrentFrame = useCallback(async () => {
    if (
      auditPendingRef.current ||
      localPreviewRef.current ||
      resetPendingRef.current ||
      !canonicalSignatureRef.current
    ) {
      return;
    }
    auditPendingRef.current = true;
    try {
      const capture = await captureWorldFrame();
      if (!capture) return;
      const canonicalScore = compareFrameSignatures(
        canonicalSignatureRef.current,
        capture.signature,
      );
      // The frame comparison is diagnostic only. A live world should be free to
      // move away from its seed without the app treating progress as corruption.
      const frameScore = Math.round(canonicalScore);
      coherenceRef.current = frameScore;
      setCoherence(frameScore);
    } finally {
      auditPendingRef.current = false;
    }
  }, []);
  auditFrameRef.current = () => {
    void auditCurrentFrame();
  };

  const sendPose = useCallback(() => {
    if (
      statusRef.current !== "ready" ||
      phaseRef.current !== "game" ||
      localPreviewRef.current
    ) {
      return;
    }
    const actionPose = activeButtonsRef.current
      .map((button) => currentGameRef.current.actions[button].pose)
      .find(Boolean);
    if (actionPose?.profile === "jump") {
      // LingBot's native pose layer is a per-frame velocity profile with y-down.
      // A symmetric up / hold / down arc reinforces the vertical prompt without
      // leaving the generated subject or camera suspended after the chunk.
      lw2
        .setCameraPose({
          camera_pose: [
            0, 0, 0, 0, -1, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0,
          ],
        })
        .catch(() => undefined);
      poseSentRef.current = true;
      return;
    }
    const rx = actionPose?.rx ?? 0;
    const ry = actionPose?.ry ?? 0;
    const rz = actionPose?.rz ?? 0;
    const ty = actionPose?.ty ?? 0;
    const active = Math.abs(rx) + Math.abs(ry) + Math.abs(rz) + Math.abs(ty) > 0.001;
    if (!active) {
      if (poseSentRef.current) {
        lw2.setCameraPose({ camera_pose: [] }).catch(() => undefined);
        poseSentRef.current = false;
      }
      return;
    }
    const cameraPose: number[] = [];
    for (let step = 0; step < CHUNK_LATENTS; step += 1) {
      cameraPose.push(rx, ry, rz, 0, ty, 0);
    }
    lw2.setCameraPose({ camera_pose: cameraPose }).catch(() => undefined);
    poseSentRef.current = true;
  }, [sendCommand]);
  sendPoseRef.current = sendPose;

  useLingbotWorld2Message((message) => {
    switch (message.type) {
      case "image_accepted":
        imageReadyRef.current?.();
        imageReadyRef.current = null;
        break;
      case "generation_started":
        setPhase("game");
        setBootState("idle");
        setLocalPreview(false);
        setPaused(false);
        setPromptStatus("accepted");
        break;
      case "chunk_complete":
        setChunkIndex(message.chunk_index);
        setLastActiveAction(message.active_action);
        promptSlotChunkRef.current = -1;
        sendPoseRef.current();
        {
          const nextRunChunk = runChunkRef.current + 1;
          runChunkRef.current = nextRunChunk;
          setRunChunk(nextRunChunk);
          const actionBonus = Object.values(currentGameRef.current.actions).some(
            (action) => message.active_prompt.includes(action.prompt),
          )
            ? 180
            : 0;
          const motionBonus = message.active_action === "still" ? 0 : 35;
          const coherenceBonus = Math.round(coherenceRef.current * 0.65);
          const nextScore = scoreRef.current + 100 + actionBonus + motionBonus + coherenceBonus;
          scoreRef.current = nextScore;
          setScore(nextScore);

          if (nextRunChunk % FRAME_AUDIT_EVERY === 0) {
            window.setTimeout(() => auditFrameRef.current(), 140);
          }

          queueComposedPrompt();
          dispatchPromptForCurrentChunk();
        }
        break;
      case "generation_paused":
        setPaused(true);
        break;
      case "generation_resumed":
        setPaused(false);
        break;
      case "generation_reset":
        setChunkIndex(0);
        break;
      case "prompt_accepted":
        if (message.prompt === lastPromptRef.current) setPromptStatus("accepted");
        break;
      case "command_error":
        setBootError(`${message.command}: ${message.reason}`);
        if (phaseRef.current === "booting") setBootState("error");
        break;
    }
  });

  const moveSelection = useCallback((direction: 1 | -1) => {
    setSelectedIndex((current) => nextGameIndex(current, direction));
  }, []);

  const clearActions = useCallback(() => {
    activeButtonsRef.current = [];
    setActiveButtons([]);
  }, []);

  const resetRunState = useCallback(() => {
    runChunkRef.current = 0;
    scoreRef.current = 0;
    coherenceRef.current = 100;
    promptQueuedRef.current = null;
    promptSlotChunkRef.current = -1;
    promptInFlightRef.current = false;
    setRunChunk(0);
    setScore(0);
    setCoherence(100);
    setLastActiveAction("still");
    setPromptStatus("locked");
  }, []);

  const stopMovement = useCallback(() => {
    if (statusRef.current !== "ready" || localPreviewRef.current) return;
    lastMoveLongRef.current = "idle";
    lastMoveLatRef.current = "idle";
    lastLookHorizontalRef.current = "idle";
    lastLookVerticalRef.current = "idle";
    movingRef.current = false;
    lw2.setMoveLongitudinal({ move_longitudinal: "idle" }).catch(() => undefined);
    lw2.setMoveLateral({ move_lateral: "idle" }).catch(() => undefined);
    lw2.setLookHorizontal({ look_horizontal: "idle" }).catch(() => undefined);
    lw2.setLookVertical({ look_vertical: "idle" }).catch(() => undefined);
    if (poseSentRef.current) {
      lw2.setCameraPose({ camera_pose: [] }).catch(() => undefined);
      poseSentRef.current = false;
    }
  }, [sendCommand]);

  const uploadReferenceAndStart = useCallback(
    async (game: ArcadeGame, imageBlob: Blob, prompt: string) => {
      const imageFile = new File([imageBlob], `${game.id}-anchor.png`, {
        type: imageBlob.type || "image/png",
      });
      const imageReady = new Promise<void>((resolve) => {
        imageReadyRef.current = resolve;
      });
      const fileRef = await uploadFile(imageFile);
      await lw2.setImage({ image: fileRef });
      await Promise.race([
        imageReady,
        waitFor(
          () => imageReadyRef.current === null,
          60_000,
          "The reference frame was not accepted in time.",
        ),
      ]);
      await lw2.setSeed({ seed: game.seed });
      await lw2.setAttnWindow({ attn_window: "small" });
      attentionWindowRef.current = "small";
      setAttentionWindow("small");
      setPromptStatus("sending");
      await lw2.setPrompt({ prompt });
      lastPromptRef.current = prompt;
      promptQueuedRef.current = null;
      promptSlotChunkRef.current = runChunkRef.current;
      setPromptStatus("accepted");
      await lw2.setRotationSpeedDeg({
        rotation_speed_deg: WORLD_CAMERA_ROTATION_SPEED_DEG,
      });
      await lw2.start();
    },
    [sendCommand, uploadFile],
  );

  const exitToCabinet = useCallback(() => {
    clearActions();
    stopMovement();
    setPaused(false);
    setLocalPreview(false);
    setPhase("cabinet");
    if (statusRef.current === "ready") lw2.reset().catch(() => undefined);
  }, [clearActions, stopMovement, sendCommand]);

  const cancelBoot = useCallback(() => {
    launchPendingRef.current = false;
    setBootError(null);
    setBootState("idle");
    setPhase("cabinet");
    if (statusRef.current !== "disconnected") lw2.disconnect().catch(() => undefined);
  }, [sendCommand]);

  const launchGame = useCallback(async () => {
    if (launchPendingRef.current) return;
    launchPendingRef.current = true;
    const game = GAMES[selectedIndexRef.current];
    currentGameRef.current = game;
    clearActions();
    resetRunState();
    setBootError(null);
    setLocalPreview(false);
    setPhase("booting");
    try {
      if (!configured) throw new Error("Live session credentials are not configured.");
      setBootState("connecting");
      if (statusRef.current === "disconnected") await lw2.connect();
      await waitFor(
        () => statusRef.current === "ready",
        120_000,
        "The world session did not become ready in time.",
      );

      setBootState("uploading");
      const imageResponse = await fetch(game.image);
      if (!imageResponse.ok) throw new Error("The reference frame could not be loaded.");
      const imageBlob = await imageResponse.blob();
      canonicalSignatureRef.current = await signatureFromImageBlob(imageBlob);
      const imageFile = new File([imageBlob], `${game.id}-seed.png`, {
        type: imageBlob.type || "image/png",
      });
      const imageReady = new Promise<void>((resolve) => {
        imageReadyRef.current = resolve;
      });
      const fileRef = await uploadFile(imageFile);
      await lw2.setImage({ image: fileRef });
      await Promise.race([
        imageReady,
        waitFor(
          () => imageReadyRef.current === null,
          60_000,
          "The reference frame was not accepted in time.",
        ),
      ]);

      setBootState("conditioning");
      const prompt = composeGamePrompt(game, false, []);
      lastPromptRef.current = prompt;
      await lw2.setSeed({ seed: game.seed });
      await lw2.setAttnWindow({ attn_window: "small" });
      attentionWindowRef.current = "small";
      setAttentionWindow("small");
      await lw2.setPrompt({ prompt });
      promptSlotChunkRef.current = 0;
      setPromptStatus("accepted");
      await lw2.setRotationSpeedDeg({
        rotation_speed_deg: WORLD_CAMERA_ROTATION_SPEED_DEG,
      });
      await new Promise((resolve) => window.setTimeout(resolve, 650));
      setBootState("starting");
      await lw2.start();
      await waitFor(
        () => phaseRef.current === "game",
        120_000,
        "The world renderer did not return its first frame in time.",
      );
    } catch (error) {
      setBootError(error instanceof Error ? error.message : "The world session could not start.");
      setBootState("error");
    } finally {
      launchPendingRef.current = false;
    }
  }, [clearActions, configured, resetRunState, sendCommand, uploadFile]);

  const useLocalPreview = useCallback(() => {
    currentGameRef.current = GAMES[selectedIndexRef.current];
    resetRunState();
    setLocalPreview(true);
    setPaused(false);
    setBootState("idle");
    setPhase("game");
  }, [resetRunState]);

  const actionDown = useCallback(
    (button: FaceButton) => {
      if (phaseRef.current !== "game") return;
      if (activeButtonsRef.current.includes(button)) return;
      activeButtonsRef.current = [...activeButtonsRef.current, button];
      setActiveButtons(activeButtonsRef.current);
      setAttentionMode("large");
      queueComposedPrompt(true);
      sendPoseRef.current();
    },
    [queueComposedPrompt, setAttentionMode],
  );

  const actionUp = useCallback(
    (button: FaceButton) => {
      if (!activeButtonsRef.current.includes(button)) return;
      activeButtonsRef.current = activeButtonsRef.current.filter(
        (item) => item !== button,
      );
      setActiveButtons(activeButtonsRef.current);
      setAttentionMode(
        movingRef.current || activeButtonsRef.current.length > 0 ? "large" : "small",
      );
      queueComposedPrompt(true);
      sendPoseRef.current();
    },
    [queueComposedPrompt, setAttentionMode],
  );

  const resetToFirstFrame = useCallback(async () => {
    if (phaseRef.current !== "game" || resetPendingRef.current) return;
    const game = currentGameRef.current;
    const wasLocalPreview = localPreviewRef.current;
    resetPendingRef.current = true;
    clearActions();
    stopMovement();
    resetRunState();
    setPaused(false);
    setChunkIndex(0);
    setHudResetToken((value) => value + 1);
    setBootError(null);
    setLocalPreview(true);
    const prompt = composeGamePrompt(game, false, []);
    lastPromptRef.current = prompt;

    if (wasLocalPreview || statusRef.current !== "ready") {
      resetPendingRef.current = false;
      return;
    }

    try {
      await lw2.reset();
      const imageResponse = await fetch(game.image);
      if (!imageResponse.ok) throw new Error("The reference frame could not be reloaded.");
      const imageBlob = await imageResponse.blob();
      canonicalSignatureRef.current = await signatureFromImageBlob(imageBlob);
      await uploadReferenceAndStart(game, imageBlob, prompt);
    } catch (error) {
      imageReadyRef.current = null;
      setBootError(error instanceof Error ? error.message : "The world could not restart.");
      setLocalPreview(true);
    } finally {
      resetPendingRef.current = false;
    }
  }, [clearActions, resetRunState, stopMovement, uploadReferenceAndStart]);

  const toggleDebug = useCallback(() => {
    setDebugVisible((value) => !value);
  }, []);

  const onButtonDown = useCallback(
    (button: number) => {
      const currentPhase = phaseRef.current;
      if (currentPhase === "room") {
        if (button === 0 && nearCabinet) setPhase("cabinet");
        return;
      }
      if (currentPhase === "cabinet") {
        if (button === 0) launchGame();
        else if (button === 2) useLocalPreview();
        else if (button === 1 || button === 8) setPhase("room");
        else if (button === 4 || button === 12 || button === 14) moveSelection(-1);
        else if (button === 5 || button === 13 || button === 15) moveSelection(1);
        return;
      }
      if (currentPhase === "booting") {
        if (button === 8 || button === 1) cancelBoot();
        else if (button === 0 && bootState === "error") launchGame();
        else if (button === 2 && bootState === "error") useLocalPreview();
        return;
      }
      if (button === 4) {
        toggleDebug();
        return;
      }
      if (button === 8) exitToCabinet();
      else if (button === 9) resetToFirstFrame();
      else {
        const face = FACE_BY_INDEX[button];
        if (face) actionDown(face);
      }
    },
    [actionDown, bootState, cancelBoot, exitToCabinet, launchGame, moveSelection, nearCabinet, resetToFirstFrame, toggleDebug, useLocalPreview],
  );

  const onButtonUp = useCallback(
    (button: number) => {
      const face = FACE_BY_INDEX[button];
      if (face) actionUp(face);
    },
    [actionUp],
  );

  const controller = useControllerInput({ onButtonDown, onButtonUp });

  const captureMouseLook = useCallback(
    (event: ReactPointerEvent<HTMLElement>) => {
      if (
        event.button !== 0 ||
        (phaseRef.current !== "room" && phaseRef.current !== "game") ||
        document.pointerLockElement
      ) {
        return;
      }
      const target = event.target;
      if (
        target instanceof HTMLElement &&
        target.closest("button, a, input, textarea, select, [role='button']")
      ) {
        return;
      }
      if (typeof event.currentTarget.requestPointerLock !== "function") {
        setSoftMouseLook(true);
        return;
      }
      try {
        const request = event.currentTarget.requestPointerLock();
        request?.catch(() => setSoftMouseLook(true));
      } catch {
        setSoftMouseLook(true);
      }
    },
    [],
  );

  const updateSoftMouseLook = useCallback(
    (event: ReactPointerEvent<HTMLElement>) => {
      if (
        !softMouseLook ||
        (phaseRef.current !== "room" && phaseRef.current !== "game")
      ) {
        return;
      }
      controller.applyMouseLook(event.movementX, event.movementY);
    },
    [controller.applyMouseLook, softMouseLook],
  );

  useEffect(() => {
    if (phase !== "room" && phase !== "game") {
      setSoftMouseLook(false);
      if (document.pointerLockElement === appRef.current) {
        document.exitPointerLock();
      }
    }
  }, [phase]);

  useEffect(() => {
    if (!softMouseLook) return;
    const releaseSoftMouseLook = (event: KeyboardEvent) => {
      if (event.code !== "Escape") return;
      event.preventDefault();
      event.stopImmediatePropagation();
      setSoftMouseLook(false);
    };
    window.addEventListener("keydown", releaseSoftMouseLook, true);
    return () => window.removeEventListener("keydown", releaseSoftMouseLook, true);
  }, [softMouseLook]);

  useEffect(() => {
    let navigationLatch = 0;
    const timer = window.setInterval(() => {
      if (phaseRef.current !== "cabinet") {
        navigationLatch = 0;
        return;
      }
      const y = controller.axesRef.current.ly;
      if (y > 0.72 && navigationLatch !== 1) {
        navigationLatch = 1;
        moveSelection(1);
      } else if (y < -0.72 && navigationLatch !== -1) {
        navigationLatch = -1;
        moveSelection(-1);
      } else if (Math.abs(y) < 0.35) {
        navigationLatch = 0;
      }
    }, 90);
    return () => window.clearInterval(timer);
  }, [controller.axesRef, moveSelection]);

  useEffect(() => {
    const timer = window.setInterval(() => {
      if (
        phaseRef.current !== "game" ||
        statusRef.current !== "ready" ||
        localPreviewRef.current ||
        pausedRef.current
      ) {
        return;
      }
      const { lx, ly, rx, ry } = controller.axesRef.current;
      const moveLong: "idle" | "forward" | "back" =
        lastMoveLongRef.current === "forward"
          ? ly > -0.2
            ? "idle"
            : "forward"
          : lastMoveLongRef.current === "back"
            ? ly < 0.2
              ? "idle"
              : "back"
            : ly < -0.36
              ? "forward"
              : ly > 0.36
                ? "back"
                : "idle";
      const moveLat: "idle" | "strafe_left" | "strafe_right" =
        lastMoveLatRef.current === "strafe_left"
          ? lx > -0.2
            ? "idle"
            : "strafe_left"
          : lastMoveLatRef.current === "strafe_right"
            ? lx < 0.2
              ? "idle"
              : "strafe_right"
            : lx < -0.36
              ? "strafe_left"
              : lx > 0.36
                ? "strafe_right"
                : "idle";
      const lookHorizontal: "idle" | "left" | "right" =
        rx < -0.24 ? "left" : rx > 0.24 ? "right" : "idle";
      const lookVertical: "idle" | "up" | "down" =
        ry < -0.24 ? "up" : ry > 0.24 ? "down" : "idle";
      let movementChanged = false;
      if (moveLong !== lastMoveLongRef.current) {
        lastMoveLongRef.current = moveLong;
        movementChanged = true;
        lw2.setMoveLongitudinal({ move_longitudinal: moveLong }).catch(() => undefined);
      }
      if (moveLat !== lastMoveLatRef.current) {
        lastMoveLatRef.current = moveLat;
        movementChanged = true;
        lw2.setMoveLateral({ move_lateral: moveLat }).catch(() => undefined);
      }
      if (lookHorizontal !== lastLookHorizontalRef.current) {
        lastLookHorizontalRef.current = lookHorizontal;
        lw2.setLookHorizontal({ look_horizontal: lookHorizontal }).catch(() => undefined);
      }
      if (lookVertical !== lastLookVerticalRef.current) {
        lastLookVerticalRef.current = lookVertical;
        lw2.setLookVertical({ look_vertical: lookVertical }).catch(() => undefined);
      }
      const moving = moveLong !== "idle" || moveLat !== "idle";
      if (moving !== movingRef.current) {
        movingRef.current = moving;
        movementChanged = true;
      }
      setAttentionMode(
        moving || activeButtonsRef.current.length > 0 ? "large" : "small",
      );
      if (movementChanged) {
        queueComposedPrompt(true);
      }
    }, 120);
    return () => window.clearInterval(timer);
  }, [controller.axesRef, queueComposedPrompt, sendCommand, setAttentionMode]);

  useEffect(() => {
    if (status === "disconnected") {
      setPaused(false);
      poseSentRef.current = false;
      lastMoveLongRef.current = "idle";
      lastMoveLatRef.current = "idle";
      lastLookHorizontalRef.current = "idle";
      lastLookVerticalRef.current = "idle";
    }
  }, [status]);

  const selectedGame = GAMES[selectedIndex];
  const activeGame = currentGameRef.current;
  const inputMethod = controller.lastDevice;
  const scenePhase: "room" | "cabinet" | "booting" =
    phase === "room" ? "room" : phase === "booting" ? "booting" : "cabinet";

  return (
    <main
      ref={appRef}
      className="arcade-app"
      data-pointer-locked={controller.pointerLocked ? "true" : "false"}
      data-mouse-look-active={
        controller.pointerLocked || softMouseLook ? "true" : "false"
      }
      onPointerDown={captureMouseLook}
      onPointerMove={updateSoftMouseLook}
      onPointerLeave={() => setSoftMouseLook(false)}
    >
      {phase !== "game" ? (
        <div className="scene-shell">
          <ArcadeScene
            phase={scenePhase}
            game={selectedGame}
            inputMethod={inputMethod}
            axesRef={controller.axesRef}
            onNearChange={setNearCabinet}
          />
          {phase === "room" && (
            <RoomOverlay
              inputMethod={inputMethod}
              near={nearCabinet}
              onInteract={() => {
                if (nearCabinet) setPhase("cabinet");
              }}
            />
          )}
          {phase === "cabinet" && (
            <CabinetOverlay
              selectedIndex={selectedIndex}
              onSelect={setSelectedIndex}
              onLaunch={launchGame}
              onPreview={useLocalPreview}
              onBack={() => setPhase("room")}
            />
          )}
          {phase === "booting" && (
            <BootOverlay
              game={selectedGame}
              inputMethod={inputMethod}
              state={bootState}
              onCancel={cancelBoot}
              onRetry={launchGame}
              onPreview={useLocalPreview}
            />
          )}
        </div>
      ) : (
        <div className="game-shell">
          <img className="seed-frame" src={activeGame.image} alt="" />
          {!localPreview && (
            <LingbotWorld2MainVideoView
              videoObjectFit="cover"
              className="world-video"
              style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
            />
          )}
          <div className="media-scrim" />
          <GameHud
            game={activeGame}
            chunkIndex={chunkIndex}
            runChunk={runChunk}
            score={score}
            coherence={coherence}
            attentionWindow={attentionWindow}
            promptStatus={promptStatus}
            lastActiveAction={lastActiveAction}
            debugVisible={debugVisible}
            paused={paused}
            localPreview={localPreview}
            activeButtons={activeButtons}
            inputMethod={inputMethod}
            axesRef={controller.axesRef}
            buttonsRef={controller.buttonsRef}
            resetToken={hudResetToken}
            onActionDown={actionDown}
            onActionUp={actionUp}
            onExit={exitToCabinet}
            onReset={resetToFirstFrame}
            onToggleDebug={toggleDebug}
          />
        </div>
      )}
      <GamepadViewerOverlay
        axesRef={controller.axesRef}
        buttonsRef={controller.buttonsRef}
        connected={controller.connected}
        inputMethod={inputMethod}
        keysRef={controller.keysRef}
        mode={phase}
        mouseLookActive={controller.pointerLocked || softMouseLook}
      />
    </main>
  );
}

export function ArcadeExperience({ configured }: { configured: boolean }) {
  return (
    <LingbotWorld2Provider apiUrl={API_URL} getJwt={fetchToken}>
      <ReactorArcade configured={configured} />
    </LingbotWorld2Provider>
  );
}
