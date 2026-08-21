"use client";

import {
  ContactShadows,
  MeshReflectorMaterial,
  PerspectiveCamera,
  RoundedBox,
  useTexture,
} from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { MutableRefObject, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import type { ControllerAxes } from "@/hooks/useControllerInput";
import { GAMES, type ArcadeGame } from "@/lib/games";

type ScenePhase = "room" | "cabinet" | "booting";

type ArcadeSceneProps = {
  phase: ScenePhase;
  game: ArcadeGame;
  axesRef: MutableRefObject<ControllerAxes>;
  onNearChange: (near: boolean) => void;
};

const BODY = "#30322f";
const BODY_EDGE = "#3a3c38";
const SIGNAL = "#c7c099";
const FLOOR_NORMAL_SCALE = new THREE.Vector2(0.72, 0.72);
const WALL_NORMAL_SCALE = new THREE.Vector2(0.5, 0.5);
const ROOM_CAMERA_YAW_SPEED = 1.15;
const ROOM_CAMERA_PITCH_SPEED = 0.86;
function useAttractTexture() {
  const [texture, setTexture] = useState<THREE.CanvasTexture | null>(null);

  useEffect(() => {
    let cancelled = false;
    const canvas = document.createElement("canvas");
    canvas.width = 1024;
    canvas.height = 576;
    const context = canvas.getContext("2d");
    if (!context) return;

    context.fillStyle = "#000000";
    context.fillRect(0, 0, canvas.width, canvas.height);

    const nextTexture = new THREE.CanvasTexture(canvas);
    nextTexture.colorSpace = THREE.SRGBColorSpace;
    nextTexture.anisotropy = 4;
    nextTexture.needsUpdate = true;
    setTexture(nextTexture);

    const loadImage = (src: string) =>
      new Promise<HTMLImageElement>((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = reject;
        image.src = src;
      });

    Promise.all([
      loadImage("/brand/reactor-arcade-prescreen-v1.png"),
      document.fonts.load('700 72px "Doto"'),
    ])
      .then(([backdrop]) => {
        if (cancelled) return;
        context.clearRect(0, 0, canvas.width, canvas.height);
        context.drawImage(backdrop, 0, 0, canvas.width, canvas.height);

        const titleField = context.createLinearGradient(0, 0, 760, 0);
        titleField.addColorStop(0, "rgba(0, 0, 0, 0.94)");
        titleField.addColorStop(0.54, "rgba(0, 0, 0, 0.7)");
        titleField.addColorStop(1, "rgba(0, 0, 0, 0)");
        context.fillStyle = titleField;
        context.fillRect(0, 0, canvas.width, canvas.height);

        context.save();
        context.fillStyle = "#fdf5c6";
        context.shadowColor = "rgba(253, 245, 198, 0.56)";
        context.shadowBlur = 17;
        context.font = '700 70px "Doto", "IBM Plex Mono", monospace';
        context.letterSpacing = "1px";
        context.fillText("WORLD MODEL", 68, 268);
        context.font = '700 88px "Doto", "IBM Plex Mono", monospace';
        context.letterSpacing = "6px";
        context.fillText("ARCADE", 68, 366);
        context.shadowBlur = 0;
        context.fillStyle = "#ffffff";
        context.font = '700 70px "Doto", "IBM Plex Mono", monospace';
        context.letterSpacing = "1px";
        context.fillText("WORLD MODEL", 68, 268);
        context.font = '700 88px "Doto", "IBM Plex Mono", monospace';
        context.letterSpacing = "6px";
        context.fillText("ARCADE", 68, 366);
        context.restore();

        const glassFalloff = context.createRadialGradient(512, 278, 130, 512, 278, 660);
        glassFalloff.addColorStop(0, "rgba(0, 0, 0, 0)");
        glassFalloff.addColorStop(0.62, "rgba(0, 0, 0, 0.045)");
        glassFalloff.addColorStop(1, "rgba(0, 0, 0, 0.46)");
        context.fillStyle = glassFalloff;
        context.fillRect(0, 0, canvas.width, canvas.height);

        context.fillStyle = "rgba(0, 0, 0, 0.14)";
        for (let y = 1; y < canvas.height; y += 3) {
          context.fillRect(0, y, canvas.width, 1);
        }

        nextTexture.needsUpdate = true;
      })
      .catch(() => {
        if (cancelled) return;
        context.fillStyle = "#ffffff";
        context.font = '700 72px "Doto", "IBM Plex Mono", monospace';
        context.letterSpacing = "3px";
        context.fillText("WORLD MODEL ARCADE", 68, 310);
        nextTexture.needsUpdate = true;
      });

    return () => {
      cancelled = true;
      nextTexture.dispose();
    };
  }, []);

  return texture;
}

function CrtGlassOverlay() {
  const material = useRef<THREE.ShaderMaterial>(null);
  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
    }),
    [],
  );

  useFrame(({ clock }) => {
    uniforms.uTime.value = clock.elapsedTime;
  });

  return (
    <shaderMaterial
      ref={material}
      uniforms={uniforms}
      toneMapped={false}
      transparent
      depthWrite={false}
      vertexShader={`
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `}
      fragmentShader={`
        uniform float uTime;
        varying vec2 vUv;

        void main() {
          vec2 centered = vUv - 0.5;
          float scanline = 0.5 + 0.5 * sin(vUv.y * 576.0 * 3.14159265);
          float grille = 0.5 + 0.5 * sin(vUv.x * 1024.0 * 2.0943951);
          float edge = smoothstep(0.34, 0.72, length(centered * vec2(1.0, 0.84)));
          float refresh = exp(-pow((fract(vUv.y + uTime * 0.075) - 0.5) * 13.0, 2.0));
          float flicker = 0.5 + 0.5 * sin(uTime * 16.0);
          float alpha = 0.045 + (1.0 - scanline) * 0.13 + grille * 0.025 + edge * 0.24 + flicker * 0.012;
          vec3 overlay = vec3(0.0);
          overlay += vec3(0.78, 0.75, 0.6) * refresh * 0.55;
          gl_FragColor = vec4(overlay, min(alpha + refresh * 0.055, 0.38));
        }
      `}
    />
  );
}

type SurfaceMaps = {
  albedo: THREE.CanvasTexture;
  roughness: THREE.CanvasTexture;
  bump: THREE.CanvasTexture;
};

type ArchitecturalMaps = {
  floor: SurfaceMaps;
  wall: SurfaceMaps;
  cabinet: SurfaceMaps;
};

function useArchitecturalMaps() {
  const [maps, setMaps] = useState<ArchitecturalMaps | null>(null);

  useEffect(() => {
    let seed = 712367;
    const random = () => {
      seed = (seed * 16807) % 2147483647;
      return (seed - 1) / 2147483646;
    };
    const makeMonochromeTexture = (
      base: number,
      variation: number,
      repeat: [number, number],
      scratches = false,
    ) => {
      const canvas = document.createElement("canvas");
      canvas.width = 512;
      canvas.height = 512;
      const context = canvas.getContext("2d");
      if (!context) return null;
      context.fillStyle = `rgb(${base}, ${base}, ${base})`;
      context.fillRect(0, 0, canvas.width, canvas.height);

      for (let index = 0; index < 1700; index += 1) {
        const value = Math.round(base + (random() - 0.5) * variation);
        const radius = 0.5 + random() * 6;
        context.fillStyle = `rgba(${value}, ${value}, ${value}, ${0.055 + random() * 0.12})`;
        context.beginPath();
        context.arc(random() * 512, random() * 512, radius, 0, Math.PI * 2);
        context.fill();
      }

      for (let index = 0; index < 44; index += 1) {
        const x = random() * 512;
        const y = random() * 512;
        const radius = 24 + random() * 92;
        const value = Math.round(base + (random() - 0.5) * variation * 0.7);
        const gradient = context.createRadialGradient(x, y, 0, x, y, radius);
        gradient.addColorStop(0, `rgba(${value}, ${value}, ${value}, 0.2)`);
        gradient.addColorStop(1, `rgba(${value}, ${value}, ${value}, 0)`);
        context.fillStyle = gradient;
        context.fillRect(x - radius, y - radius, radius * 2, radius * 2);
      }

      if (scratches) {
        context.lineCap = "round";
        for (let index = 0; index < 38; index += 1) {
          const startX = random() * 512;
          const startY = random() * 512;
          context.strokeStyle = `rgba(35, 35, 35, ${0.12 + random() * 0.2})`;
          context.lineWidth = 0.4 + random() * 1.5;
          context.beginPath();
          context.moveTo(startX, startY);
          context.lineTo(startX + (random() - 0.5) * 110, startY + (random() - 0.5) * 28);
          context.stroke();
        }
      }

      const texture = new THREE.CanvasTexture(canvas);
      texture.wrapS = THREE.RepeatWrapping;
      texture.wrapT = THREE.RepeatWrapping;
      texture.repeat.set(...repeat);
      texture.anisotropy = 4;
      texture.needsUpdate = true;
      return texture;
    };

    const makeAlbedoTexture = (
      base: [number, number, number],
      repeat: [number, number],
      floorLike: boolean,
    ) => {
      const canvas = document.createElement("canvas");
      canvas.width = 512;
      canvas.height = 512;
      const context = canvas.getContext("2d");
      if (!context) return null;
      const image = context.createImageData(512, 512);
      for (let y = 0; y < 512; y += 1) {
        for (let x = 0; x < 512; x += 1) {
          const offset = (y * 512 + x) * 4;
          const broad =
            Math.sin(x * 0.021) * 0.42 +
            Math.cos(y * 0.027) * 0.34 +
            Math.sin((x + y) * 0.009) * 0.48;
          const grain = (random() - 0.5) * (floorLike ? 34 : 25);
          const shift = broad * (floorLike ? 20 : 14) + grain;
          image.data[offset] = THREE.MathUtils.clamp(base[0] + shift, 0, 255);
          image.data[offset + 1] = THREE.MathUtils.clamp(base[1] + shift, 0, 255);
          image.data[offset + 2] = THREE.MathUtils.clamp(base[2] + shift, 0, 255);
          image.data[offset + 3] = 255;
        }
      }
      context.putImageData(image, 0, 0);

      context.lineCap = "round";
      const fissures = floorLike ? 28 : 13;
      for (let index = 0; index < fissures; index += 1) {
        let x = random() * 512;
        let y = random() * 512;
        context.strokeStyle = floorLike ? "rgba(43, 37, 29, 0.18)" : "rgba(45, 42, 36, 0.16)";
        context.lineWidth = 0.35 + random() * (floorLike ? 0.82 : 0.64);
        context.beginPath();
        context.moveTo(x, y);
        for (let point = 0; point < 5; point += 1) {
          x += (random() - 0.5) * 42;
          y += 9 + random() * 28;
          context.lineTo(x, y);
        }
        context.stroke();
      }

      for (let index = 0; index < (floorLike ? 860 : 980); index += 1) {
        const value = floorLike ? 45 + random() * 105 : 36 + random() * 70;
        context.fillStyle = `rgba(${value}, ${value * 0.94}, ${value * 0.82}, ${0.06 + random() * 0.14})`;
        context.beginPath();
        context.arc(random() * 512, random() * 512, 0.4 + random() * 2.8, 0, Math.PI * 2);
        context.fill();
      }

      const texture = new THREE.CanvasTexture(canvas);
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.wrapS = THREE.RepeatWrapping;
      texture.wrapT = THREE.RepeatWrapping;
      texture.repeat.set(...repeat);
      texture.anisotropy = 8;
      texture.needsUpdate = true;
      return texture;
    };

    const makeCabinetPaintTexture = () => {
      const canvas = document.createElement("canvas");
      canvas.width = 512;
      canvas.height = 512;
      const context = canvas.getContext("2d");
      if (!context) return null;
      const image = context.createImageData(512, 512);

      for (let y = 0; y < 512; y += 1) {
        for (let x = 0; x < 512; x += 1) {
          const offset = (y * 512 + x) * 4;
          const broad =
            Math.sin(x * 0.034) * 2.4 +
            Math.cos(y * 0.029) * 1.8 +
            Math.sin((x + y) * 0.011) * 2.1;
          const grain = (random() - 0.5) * 15;
          image.data[offset] = THREE.MathUtils.clamp(153 + broad + grain, 0, 255);
          image.data[offset + 1] = THREE.MathUtils.clamp(154 + broad + grain, 0, 255);
          image.data[offset + 2] = THREE.MathUtils.clamp(151 + broad + grain, 0, 255);
          image.data[offset + 3] = 255;
        }
      }
      context.putImageData(image, 0, 0);

      for (let index = 0; index < 520; index += 1) {
        const value = 52 + random() * 62;
        context.fillStyle = `rgba(${value}, ${value}, ${value * 0.92}, ${0.025 + random() * 0.055})`;
        context.beginPath();
        context.arc(random() * 512, random() * 512, 0.25 + random() * 1.15, 0, Math.PI * 2);
        context.fill();
      }

      context.lineCap = "round";
      for (let index = 0; index < 22; index += 1) {
        const x = random() * 512;
        const y = random() * 512;
        context.strokeStyle = `rgba(210, 201, 174, ${0.035 + random() * 0.045})`;
        context.lineWidth = 0.3 + random() * 0.55;
        context.beginPath();
        context.moveTo(x, y);
        context.lineTo(x + (random() - 0.5) * 76, y + (random() - 0.5) * 16);
        context.stroke();
      }

      const texture = new THREE.CanvasTexture(canvas);
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.wrapS = THREE.RepeatWrapping;
      texture.wrapT = THREE.RepeatWrapping;
      texture.repeat.set(2.45, 3.15);
      texture.anisotropy = 8;
      texture.needsUpdate = true;
      return texture;
    };

    const floor = {
      albedo: makeAlbedoTexture([158, 151, 135], [3.2, 4.1], true),
      roughness: makeMonochromeTexture(191, 108, [3.2, 4.1]),
      bump: makeMonochromeTexture(128, 118, [3.2, 4.1], true),
    };
    const wall = {
      albedo: makeAlbedoTexture([126, 120, 108], [2.15, 1.8], false),
      roughness: makeMonochromeTexture(218, 78, [2.15, 1.8]),
      bump: makeMonochromeTexture(126, 94, [2.15, 1.8], true),
    };
    const cabinet = {
      albedo: makeCabinetPaintTexture(),
      roughness: makeMonochromeTexture(214, 48, [2.45, 3.15], true),
      bump: makeMonochromeTexture(128, 28, [2.45, 3.15], true),
    };
    if (
      !floor.albedo || !floor.roughness || !floor.bump ||
      !wall.albedo || !wall.roughness || !wall.bump ||
      !cabinet.albedo || !cabinet.roughness || !cabinet.bump
    ) return;

    const nextMaps: ArchitecturalMaps = {
      floor: floor as SurfaceMaps,
      wall: wall as SurfaceMaps,
      cabinet: cabinet as SurfaceMaps,
    };
    setMaps(nextMaps);

    return () => {
      Object.values(nextMaps.floor).forEach((texture) => texture.dispose());
      Object.values(nextMaps.wall).forEach((texture) => texture.dispose());
      Object.values(nextMaps.cabinet).forEach((texture) => texture.dispose());
    };
  }, []);

  return maps;
}

type PbrSurfaceMaps = {
  albedo: THREE.Texture;
  normal: THREE.Texture;
  roughness: THREE.Texture;
  height: THREE.Texture;
};

function usePbrSurfaceMaps() {
  const textures = useTexture([
    "/materials/floor-concrete040/color.jpg",
    "/materials/floor-concrete040/normal.jpg",
    "/materials/floor-concrete040/roughness.jpg",
    "/materials/floor-concrete040/height.jpg",
    "/materials/wall-concrete024/color.jpg",
    "/materials/wall-concrete024/normal.jpg",
    "/materials/wall-concrete024/roughness.jpg",
    "/materials/wall-concrete024/height.jpg",
  ]) as THREE.Texture[];

  return useMemo(() => {
    const [
      floorAlbedo,
      floorNormal,
      floorRoughness,
      floorHeight,
      wallAlbedo,
      wallNormal,
      wallRoughness,
      wallHeight,
    ] = textures;

    const configure = (
      texture: THREE.Texture,
      repeat: [number, number],
      colorTexture = false,
    ) => {
      texture.wrapS = THREE.RepeatWrapping;
      texture.wrapT = THREE.RepeatWrapping;
      texture.repeat.set(...repeat);
      texture.anisotropy = 8;
      texture.colorSpace = colorTexture ? THREE.SRGBColorSpace : THREE.NoColorSpace;
      texture.needsUpdate = true;
    };

    [floorAlbedo, floorNormal, floorRoughness, floorHeight].forEach((texture, index) =>
      configure(texture, [1.9, 2.65], index === 0),
    );
    [wallAlbedo, wallNormal, wallRoughness, wallHeight].forEach((texture, index) =>
      configure(texture, [2.35, 1.9], index === 0),
    );

    return {
      floor: {
        albedo: floorAlbedo,
        normal: floorNormal,
        roughness: floorRoughness,
        height: floorHeight,
      } satisfies PbrSurfaceMaps,
      wall: {
        albedo: wallAlbedo,
        normal: wallNormal,
        roughness: wallRoughness,
        height: wallHeight,
      } satisfies PbrSurfaceMaps,
    };
  }, [textures]);
}

function drawCover(
  context: CanvasRenderingContext2D,
  image: HTMLImageElement,
  x: number,
  y: number,
  width: number,
  height: number,
) {
  const scale = Math.max(width / image.naturalWidth, height / image.naturalHeight);
  const sourceWidth = width / scale;
  const sourceHeight = height / scale;
  const sourceX = (image.naturalWidth - sourceWidth) / 2;
  const sourceY = (image.naturalHeight - sourceHeight) / 2;
  context.drawImage(
    image,
    sourceX,
    sourceY,
    sourceWidth,
    sourceHeight,
    x,
    y,
    width,
    height,
  );
}

function drawButtonPrompt(
  context: CanvasRenderingContext2D,
  button: "A" | "X",
  label: string,
  x: number,
  y: number,
  width: number,
  emphasized = false,
) {
  context.fillStyle = emphasized ? "#c7c099" : "rgba(5, 8, 10, 0.82)";
  context.strokeStyle = emphasized ? "#c7c099" : "rgba(241, 242, 237, 0.38)";
  context.lineWidth = 2;
  context.beginPath();
  context.roundRect(x, y, width, 62, 5);
  context.fill();
  context.stroke();

  context.fillStyle = emphasized ? "#11171a" : "#071014";
  context.strokeStyle = emphasized ? "rgba(17, 23, 26, 0.62)" : button === "A" ? "#7fa276" : "#7196ac";
  context.beginPath();
  context.arc(x + 31, y + 31, 17, 0, Math.PI * 2);
  context.fill();
  context.stroke();

  context.fillStyle = emphasized ? "#f4e8ca" : button === "A" ? "#bad8b1" : "#acd0e3";
  context.font = "700 18px Arial, sans-serif";
  context.textAlign = "center";
  context.fillText(button, x + 31, y + 38);
  context.textAlign = "left";
  context.fillStyle = emphasized ? "#11171a" : "#f1f2ed";
  context.font = "700 17px Arial, sans-serif";
  context.fillText(label, x + 59, y + 38);
}

function useCabinetMenuTexture(game: ArcadeGame) {
  const [texture, setTexture] = useState<THREE.CanvasTexture | null>(null);

  useEffect(() => {
    let cancelled = false;
    const canvas = document.createElement("canvas");
    canvas.width = 1038;
    canvas.height = 600;
    const context = canvas.getContext("2d");
    if (!context) return;

    const nextTexture = new THREE.CanvasTexture(canvas);
    nextTexture.colorSpace = THREE.SRGBColorSpace;
    nextTexture.anisotropy = 4;

    const render = (images: HTMLImageElement[] = []) => {
      const detailWidth = 650;
      const menuX = detailWidth;
      const selectedIndex = GAMES.findIndex((item) => item.id === game.id);
      const image = images[selectedIndex];

      context.clearRect(0, 0, canvas.width, canvas.height);
      context.fillStyle = "#071014";
      context.fillRect(0, 0, canvas.width, canvas.height);

      if (image?.naturalWidth) {
        drawCover(context, image, 0, 0, detailWidth, canvas.height);
      }

      const imageShade = context.createLinearGradient(0, 78, 0, canvas.height);
      imageShade.addColorStop(0, "rgba(5, 8, 10, 0.08)");
      imageShade.addColorStop(0.5, "rgba(5, 8, 10, 0.34)");
      imageShade.addColorStop(1, "rgba(5, 8, 10, 0.98)");
      context.fillStyle = imageShade;
      context.fillRect(0, 0, detailWidth, canvas.height);

      const edgeShade = context.createLinearGradient(455, 0, detailWidth, 0);
      edgeShade.addColorStop(0, "rgba(5, 8, 10, 0)");
      edgeShade.addColorStop(1, "rgba(5, 8, 10, 0.9)");
      context.fillStyle = edgeShade;
      context.fillRect(455, 0, detailWidth - 455, canvas.height);

      context.fillStyle = "rgba(7, 10, 12, 0.96)";
      context.fillRect(menuX, 0, canvas.width - menuX, canvas.height);
      context.fillStyle = "#c7c099";
      context.fillRect(0, 0, detailWidth, 9);
      context.fillStyle = "rgba(241, 242, 237, 0.16)";
      context.fillRect(menuX, 0, 2, canvas.height);

      context.fillStyle = "#c7c099";
      context.font = "700 16px Arial, sans-serif";
      context.letterSpacing = "3px";
      context.fillText(game.genre, 42, 324);

      context.fillStyle = "#f1f2ed";
      context.font = "700 56px Arial, sans-serif";
      context.letterSpacing = "-2px";
      context.fillText(game.title, 38, 398);

      context.fillStyle = "rgba(241, 242, 237, 0.76)";
      context.font = "500 20px Arial, sans-serif";
      context.letterSpacing = "0px";
      context.fillText(game.deck, 42, 440, 560);

      drawButtonPrompt(context, "A", "ENTER WORLD", 42, 486, 214, true);
      drawButtonPrompt(context, "X", "PREVIEW HUD", 270, 486, 214);

      context.fillStyle = "#879198";
      context.font = "700 13px Arial, sans-serif";
      context.letterSpacing = "3px";
      context.fillText("SELECT A WORLD", menuX + 24, 43);
      context.fillStyle = "#c7c099";
      context.font = "700 18px Arial, sans-serif";
      context.textAlign = "right";
      context.fillText(String(GAMES.length).padStart(2, "0"), canvas.width - 24, 44);
      context.textAlign = "left";

      const rowTop = 62;
      const menuBottom = 508;
      const rowHeight = Math.min(88, (menuBottom - rowTop) / GAMES.length);
      GAMES.forEach((item, index) => {
        const selected = item.id === game.id;
        const y = rowTop + index * rowHeight;
        const thumbnailHeight = Math.min(56, rowHeight - 18);
        if (selected) {
          context.fillStyle = "#1b201f";
          context.fillRect(menuX + 2, y, canvas.width - menuX - 2, rowHeight - 2);
          context.fillStyle = "#c7c099";
          context.fillRect(menuX + 2, y, 6, rowHeight - 2);
        }
        context.fillStyle = "rgba(241, 242, 237, 0.13)";
        context.fillRect(menuX + 22, y + rowHeight - 2, canvas.width - menuX - 44, 1);
        if (images[index]?.naturalWidth) {
          drawCover(context, images[index], menuX + 18, y + 9, 84, thumbnailHeight);
          context.fillStyle = selected ? "rgba(199, 192, 153, 0.95)" : "rgba(7, 10, 12, 0.3)";
          context.fillRect(menuX + 18, y + 9 + thumbnailHeight, 84, selected ? 2 : 1);
        }
        context.fillStyle = selected ? "#f1f2ed" : "rgba(241, 242, 237, 0.56)";
        context.font = "700 18px Arial, sans-serif";
        context.letterSpacing = "-1px";
        context.fillText(item.shortTitle, menuX + 116, y + 31);
        context.fillStyle = selected ? "#c7c099" : "rgba(241, 242, 237, 0.36)";
        context.font = "600 8px Arial, sans-serif";
        context.letterSpacing = "1.2px";
        context.fillText(item.genre, menuX + 116, y + 50);
        context.fillStyle = selected ? "#c7c099" : "rgba(241, 242, 237, 0.26)";
        context.font = "700 10px Arial, sans-serif";
        context.textAlign = "right";
        context.fillText(String(index + 1).padStart(2, "0"), canvas.width - 22, y + 31);
        context.textAlign = "left";
      });

      context.fillStyle = "#101619";
      context.fillRect(menuX + 2, 508, canvas.width - menuX - 2, 92);
      context.fillStyle = "#f1f2ed";
      context.font = "700 15px Arial, sans-serif";
      context.letterSpacing = "2px";
      context.fillText("LS", menuX + 24, 544);
      context.fillStyle = "#c7c099";
      context.fillText("UP / DOWN", menuX + 62, 544);
      context.fillStyle = "rgba(241, 242, 237, 0.52)";
      context.font = "600 11px Arial, sans-serif";
      context.letterSpacing = "2px";
      context.fillText("SELECT", menuX + 24, 570);
      context.textAlign = "right";
      context.fillText("VIEW  BACK", canvas.width - 22, 570);
      context.textAlign = "left";

      nextTexture.needsUpdate = true;
    };

    render();
    setTexture(nextTexture);

    Promise.all(
      GAMES.map(
        (item) =>
          new Promise<HTMLImageElement>((resolve) => {
            const image = new Image();
            image.onload = () => resolve(image);
            image.onerror = () => resolve(image);
            image.src = item.image;
          }),
      ),
    ).then((images) => {
      if (!cancelled) render(images);
    });

    return () => {
      cancelled = true;
      nextTexture.dispose();
    };
  }, [game]);

  return texture;
}

function CabinetScrew({ position }: { position: [number, number, number] }) {
  return (
    <mesh position={position} castShadow>
      <circleGeometry args={[0.026, 16]} />
      <meshStandardMaterial color="#5b5f5f" roughness={0.3} metalness={0.88} />
    </mesh>
  );
}

function Cabinet({
  game,
  phase,
  surfaceMaps,
}: {
  game: ArcadeGame;
  phase: ScenePhase;
  surfaceMaps?: SurfaceMaps;
}) {
  const attractTexture = useAttractTexture();
  const menuTexture = useCabinetMenuTexture(game);
  const wordmarkTexture = useTexture("/brand/reactor-lockup-white.png");
  const displayTexture = phase === "room" ? attractTexture : menuTexture;
  wordmarkTexture.colorSpace = THREE.SRGBColorSpace;
  wordmarkTexture.anisotropy = 4;

  const screws: [number, number, number][] = [
    [-1.08, 3.58, 0.575],
    [1.08, 3.58, 0.575],
    [-1.08, 3.28, 0.575],
    [1.08, 3.28, 0.575],
    [-1.03, 2.98, 0.602],
    [1.03, 2.98, 0.602],
    [-1.03, 1.98, 0.602],
    [1.03, 1.98, 0.602],
    [-0.96, 1.46, 0.646],
    [0.96, 1.46, 0.646],
    [-0.96, 0.18, 0.596],
    [0.96, 0.18, 0.596],
  ];

  return (
    <group position={[0, 0, 0]}>
      <RoundedBox args={[2.24, 1.72, 1.16]} position={[0, 0.86, 0]} radius={0.045} smoothness={5} castShadow>
        <meshStandardMaterial
          color={BODY}
          map={surfaceMaps?.albedo}
          roughness={0.74}
          metalness={0.22}
          roughnessMap={surfaceMaps?.roughness}
          bumpMap={surfaceMaps?.bump}
          bumpScale={0.012}
        />
      </RoundedBox>
      <RoundedBox args={[2.42, 1.72, 1.02]} position={[0, 2.38, 0.05]} radius={0.045} smoothness={5} castShadow>
        <meshStandardMaterial
          color={BODY}
          map={surfaceMaps?.albedo}
          roughness={0.7}
          metalness={0.24}
          roughnessMap={surfaceMaps?.roughness}
          bumpMap={surfaceMaps?.bump}
          bumpScale={0.011}
        />
      </RoundedBox>

      <RoundedBox args={[2.7, 0.54, 1.06]} position={[0, 3.44, 0.03]} radius={0.035} smoothness={5} castShadow>
        <meshStandardMaterial
          color="#171612"
          map={surfaceMaps?.albedo}
          roughness={0.62}
          metalness={0.34}
          roughnessMap={surfaceMaps?.roughness}
          bumpMap={surfaceMaps?.bump}
          bumpScale={0.01}
        />
      </RoundedBox>
      {[-1.26, 1.26].map((x) => (
        <RoundedBox key={`marquee-bracket-${x}`} args={[0.17, 0.48, 0.12]} position={[x, 3.44, 0.57]} radius={0.018} smoothness={3} castShadow>
          <meshStandardMaterial color="#26231d" roughness={0.35} metalness={0.78} />
        </RoundedBox>
      ))}
      <RoundedBox args={[2.35, 0.34, 0.055]} position={[0, 3.44, 0.575]} radius={0.018} smoothness={3}>
        <meshStandardMaterial
          color="#090909"
          map={surfaceMaps?.albedo}
          roughness={0.72}
          metalness={0.22}
          roughnessMap={surfaceMaps?.roughness}
          bumpMap={surfaceMaps?.bump}
          bumpScale={0.008}
        />
      </RoundedBox>
      <mesh position={[0, 3.44, 0.607]}>
        <planeGeometry args={[1.72, 0.197]} />
        <meshBasicMaterial
          map={wordmarkTexture}
          transparent
          alphaTest={0.1}
          opacity={0.94}
          color="#ffffff"
          toneMapped={false}
        />
      </mesh>
      {[-1, 1].flatMap((side) =>
        [3.56, 3.32].map((y) => (
          <mesh key={`marquee-bolt-${side}-${y}`} position={[side * 1.26, y, 0.64]}>
            <circleGeometry args={[0.035, 20]} />
            <meshStandardMaterial color="#6b6251" roughness={0.26} metalness={0.92} />
          </mesh>
        )),
      )}
      {[3.69, 3.18].map((y) => (
        <mesh key={`marquee-rail-${y}`} position={[0, y, 0.605]}>
          <boxGeometry args={[2.36, 0.028, 0.055]} />
          <meshStandardMaterial color="#766a53" roughness={0.25} metalness={0.86} />
        </mesh>
      ))}

      <RoundedBox args={[2.28, 0.2, 0.46]} position={[0, 3.13, 0.37]} radius={0.018} smoothness={3} castShadow>
        <meshStandardMaterial
          color="#171612"
          map={surfaceMaps?.albedo}
          roughness={0.64}
          metalness={0.3}
          roughnessMap={surfaceMaps?.roughness}
          bumpMap={surfaceMaps?.bump}
          bumpScale={0.009}
        />
      </RoundedBox>
      {[-1.12, 1.12].map((x) => (
        <RoundedBox key={`marquee-shoulder-${x}`} args={[0.14, 0.34, 0.24]} position={[x, 3.1, 0.48]} radius={0.015} smoothness={3} castShadow>
          <meshStandardMaterial color="#2e2a23" roughness={0.36} metalness={0.78} />
        </RoundedBox>
      ))}

      <group position={[0, 3.145, 0.585]}>
        {Array.from({ length: 18 }, (_, index) => (
          <mesh key={index} position={[-0.7 + index * 0.082, 0, 0]}>
            <planeGeometry args={[0.045, 0.018]} />
            <meshBasicMaterial color="#020303" />
          </mesh>
        ))}
      </group>

      <RoundedBox args={[2.2, 1.34, 0.08]} position={[0, 2.48, 0.58]} radius={0.025} smoothness={3}>
        <meshStandardMaterial color="#030404" roughness={0.24} metalness={0.62} side={THREE.BackSide} />
      </RoundedBox>
      {[-1.08, 1.08].map((x) => (
        <mesh key={`screen-side-trim-${x}`} position={[x, 2.48, 0.64]}>
          <boxGeometry args={[0.035, 1.3, 0.05]} />
          <meshStandardMaterial color="#6d6250" roughness={0.3} metalness={0.88} />
        </mesh>
      ))}
      {[1.82, 3.14].map((y) => (
        <mesh key={`screen-horizontal-trim-${y}`} position={[0, y, 0.64]}>
          <boxGeometry args={[2.18, 0.035, 0.05]} />
          <meshStandardMaterial color="#5e5648" roughness={0.32} metalness={0.82} />
        </mesh>
      ))}
      <mesh position={[0, 2.48, 0.626]}>
        <planeGeometry args={[2.04, 1.18]} />
        <meshBasicMaterial
          key={displayTexture?.uuid ?? `${phase}-screen-fallback`}
          color={displayTexture ? "#ffffff" : "#071014"}
          map={displayTexture ?? undefined}
          toneMapped={false}
        />
      </mesh>
      {phase === "room" && (
        <mesh position={[0, 2.48, 0.631]}>
          <planeGeometry args={[2.04, 1.18]} />
          <CrtGlassOverlay />
        </mesh>
      )}
      <mesh position={[0, 2.48, 0.634]}>
        <planeGeometry args={[2.04, 1.18]} />
        <meshPhysicalMaterial
          color="#dfe7e8"
          transparent
          opacity={0.07}
          roughness={0.16}
          metalness={0}
          clearcoat={1}
          clearcoatRoughness={0.12}
          depthWrite={false}
        />
      </mesh>

      <RoundedBox
        args={[2.48, 0.24, 1.42]}
        position={[0, 1.63, 0.33]}
        rotation={[-0.08, 0, 0]}
        radius={0.025}
        smoothness={5}
        castShadow
      >
        <meshStandardMaterial
          color={BODY_EDGE}
          map={surfaceMaps?.albedo}
          roughness={0.66}
          metalness={0.28}
          roughnessMap={surfaceMaps?.roughness}
          bumpMap={surfaceMaps?.bump}
          bumpScale={0.01}
        />
      </RoundedBox>
      <RoundedBox
        args={[2.16, 0.055, 1.02]}
        position={[0, 1.765, 0.3]}
        rotation={[-0.08, 0, 0]}
        radius={0.022}
        smoothness={4}
        castShadow
      >
        <meshStandardMaterial
          color="#151713"
          map={surfaceMaps?.albedo}
          roughness={0.86}
          metalness={0.14}
          roughnessMap={surfaceMaps?.roughness}
          bumpMap={surfaceMaps?.bump}
          bumpScale={0.007}
        />
      </RoundedBox>
      <mesh position={[0, 1.735, 1.005]} rotation={[-0.08, 0, 0]}>
        <boxGeometry args={[2.32, 0.035, 0.045]} />
        <meshStandardMaterial color="#77786f" roughness={0.36} metalness={0.72} />
      </mesh>

      <group position={[-0.5, 1.81, 0.72]} rotation={[-0.08, 0, 0]}>
        <mesh castShadow>
          <cylinderGeometry args={[0.135, 0.135, 0.03, 32]} />
          <meshStandardMaterial color="#050605" metalness={0.48} roughness={0.48} />
        </mesh>
        <mesh position={[0, 0.042, 0]} castShadow>
          <cylinderGeometry args={[0.092, 0.112, 0.06, 28]} />
          <meshStandardMaterial color="#11120f" metalness={0.36} roughness={0.42} />
        </mesh>
        <mesh position={[0, 0.12, 0]} castShadow>
          <cylinderGeometry args={[0.027, 0.027, 0.16, 20]} />
          <meshStandardMaterial color="#a8aaa3" metalness={0.92} roughness={0.16} />
        </mesh>
        <mesh position={[0, 0.225, 0]} castShadow>
          <sphereGeometry args={[0.092, 32, 20]} />
          <meshPhysicalMaterial
            color="#a8782b"
            roughness={0.28}
            metalness={0.08}
            clearcoat={0.7}
            clearcoatRoughness={0.18}
          />
        </mesh>
      </group>

      {[0.16, 0.38, 0.6, 0.82].map((x, index) => (
        <group key={x} position={[x, 1.81, 0.7]} rotation={[-0.08, 0, 0]}>
          <mesh castShadow>
            <cylinderGeometry args={[0.09, 0.09, 0.025, 28]} />
            <meshStandardMaterial color="#080908" metalness={0.5} roughness={0.42} />
          </mesh>
          <mesh position={[0, 0.03, 0]} castShadow>
            <cylinderGeometry args={[0.074, 0.08, 0.042, 28]} />
            <meshPhysicalMaterial
              color={["#596b50", "#827648", "#49646b", "#6c4843"][index]}
              metalness={0.12}
              roughness={0.3}
              clearcoat={0.55}
              clearcoatRoughness={0.22}
            />
          </mesh>
        </group>
      ))}

      <RoundedBox args={[0.86, 0.72, 0.035]} position={[0, 0.73, 0.596]} radius={0.015} smoothness={3}>
        <meshStandardMaterial
          color="#14130f"
          map={surfaceMaps?.albedo}
          roughness={0.58}
          metalness={0.26}
          roughnessMap={surfaceMaps?.roughness}
          bumpMap={surfaceMaps?.bump}
          bumpScale={0.008}
        />
      </RoundedBox>
      {[
        [0, 1.17, 2.05, 0.035],
        [0, 0.29, 0.96, 0.032],
        [-0.47, 0.73, 0.032, 0.88],
        [0.47, 0.73, 0.032, 0.88],
      ].map(([x, y, width, height], index) => (
        <mesh key={`front-seam-${index}`} position={[x, y, 0.622]}>
          <boxGeometry args={[width, height, 0.022]} />
          <meshStandardMaterial color="#756a56" roughness={0.32} metalness={0.82} />
        </mesh>
      ))}
      <mesh position={[0, 0.76, 0.626]}>
        <cylinderGeometry args={[0.038, 0.038, 0.018, 20]} />
        <meshStandardMaterial color="#766b58" roughness={0.26} metalness={0.9} />
      </mesh>
      {[-0.78, 0.78].flatMap((centerX) =>
        Array.from({ length: 5 }, (_, column) =>
          Array.from({ length: 6 }, (_, row) => (
            <mesh
              key={`speaker-${centerX}-${column}-${row}`}
              position={[centerX + (column - 2) * 0.072, 0.18 + row * 0.065, 0.626]}
            >
              <circleGeometry args={[0.018, 12]} />
              <meshBasicMaterial color="#020303" />
            </mesh>
          )),
        ),
      )}
      {[-0.39, 0.39].flatMap((x) =>
        [0.4, 1.06].map((y) => (
          <mesh key={`service-bolt-${x}-${y}`} position={[x, y, 0.65]}>
            <circleGeometry args={[0.023, 16]} />
            <meshStandardMaterial color="#716754" roughness={0.28} metalness={0.92} />
          </mesh>
        )),
      )}
      {[-1.08, 1.08].map((x) => (
        <mesh key={`deck-bolt-${x}`} position={[x, 1.7, 0.992]} rotation={[-0.08, 0, 0]}>
          <circleGeometry args={[0.03, 18]} />
          <meshStandardMaterial color="#85775d" roughness={0.24} metalness={0.94} />
        </mesh>
      ))}
      {screws.map((position) => (
        <CabinetScrew key={position.join("-")} position={position} />
      ))}
      {[-1.04, 1.04].map((x) => (
        <mesh key={`cabinet-rail-${x}`} position={[x, 0.82, 0.61]}>
          <boxGeometry args={[0.045, 1.52, 0.045]} />
          <meshStandardMaterial color="#5e5749" roughness={0.3} metalness={0.84} />
        </mesh>
      ))}
      {[-0.86, 0.86].map((x) => (
        <RoundedBox key={`cabinet-foot-${x}`} args={[0.34, 0.1, 0.62]} position={[x, 0.03, 0.06]} radius={0.018} smoothness={3} castShadow>
          <meshStandardMaterial color="#0a0a09" roughness={0.72} metalness={0.26} />
        </RoundedBox>
      ))}
      <mesh position={[0, 0.15, 0.61]}>
        <boxGeometry args={[0.72, 0.055, 0.025]} />
        <meshBasicMaterial color={SIGNAL} toneMapped={false} />
      </mesh>
      <mesh position={[0, 0.08, 0.605]}>
        <boxGeometry args={[2.08, 0.035, 0.04]} />
        <meshStandardMaterial color="#655c4c" roughness={0.3} metalness={0.86} />
      </mesh>
    </group>
  );
}

function SceneRig({
  phase,
  game,
  axesRef,
  onNearChange,
}: ArcadeSceneProps) {
  const { camera } = useThree();
  const architecturalMaps = useArchitecturalMaps();
  const pbrMaps = usePbrSurfaceMaps();
  const player = useRef({ x: 0, z: 7.25, yaw: 0, pitch: 0.018 });
  const lastNear = useRef(false);
  const desired = useMemo(() => new THREE.Vector3(), []);
  const lookTarget = useMemo(() => new THREE.Vector3(), []);
  const viewDirection = useMemo(() => new THREE.Vector3(), []);
  const screenDirection = useMemo(() => new THREE.Vector3(), []);

  useFrame((_, delta) => {
    if (phase === "room") {
      const frameDelta = Math.min(delta, 0.05);
      const axes = axesRef.current;
      player.current.yaw += axes.rx * frameDelta * ROOM_CAMERA_YAW_SPEED;
      player.current.pitch = THREE.MathUtils.clamp(
        player.current.pitch - axes.ry * frameDelta * ROOM_CAMERA_PITCH_SPEED,
        -0.74,
        0.62,
      );

      const forwardAmount = -axes.ly;
      const strafeAmount = axes.lx;
      const inputLength = Math.max(1, Math.hypot(forwardAmount, strafeAmount));
      const moveSpeed = 3.05 * frameDelta;
      const sinYaw = Math.sin(player.current.yaw);
      const cosYaw = Math.cos(player.current.yaw);
      const deltaX =
        ((strafeAmount * cosYaw + forwardAmount * sinYaw) / inputLength) * moveSpeed;
      const deltaZ =
        ((strafeAmount * sinYaw - forwardAmount * cosYaw) / inputLength) * moveSpeed;
      let nextX = THREE.MathUtils.clamp(player.current.x + deltaX, -4.65, 4.65);
      let nextZ = THREE.MathUtils.clamp(player.current.z + deltaZ, -7.6, 11.2);

      const enteringCabinet =
        Math.abs(nextX) < 1.58 && nextZ < 1.34 && nextZ > -0.78;
      if (enteringCabinet) {
        if (Math.abs(player.current.x) >= 1.58) nextX = player.current.x;
        else nextZ = player.current.z;
      }

      player.current.x = nextX;
      player.current.z = nextZ;
      desired.set(player.current.x, 2.15, player.current.z);
      viewDirection.set(
        Math.sin(player.current.yaw) * Math.cos(player.current.pitch),
        Math.sin(player.current.pitch),
        -Math.cos(player.current.yaw) * Math.cos(player.current.pitch),
      );
      screenDirection
        .set(-player.current.x, 2.48 - 2.15, 0.63 - player.current.z)
        .normalize();
      const cabinetDistance = Math.hypot(
        player.current.x,
        player.current.z - 1.35,
      );
      const near =
        cabinetDistance < 2.65 &&
        (viewDirection.dot(screenDirection) > 0.32 || cabinetDistance < 1.72);
      if (near !== lastNear.current) {
        lastNear.current = near;
        onNearChange(near);
      }
    } else {
      desired.set(0, 2.48, 3.05);
      viewDirection.set(0, 0, -1);
      if (!lastNear.current) {
        lastNear.current = true;
        onNearChange(true);
      }
    }
    const ease = 1 - Math.exp(-delta * (phase === "room" ? 11 : 5.5));
    camera.position.lerp(desired, ease);
    if (phase === "room") lookTarget.copy(camera.position).add(viewDirection);
    else lookTarget.set(0, 2.6, 0.08);
    camera.lookAt(lookTarget);
  });

  return (
    <>
      <PerspectiveCamera makeDefault fov={39} position={[0, 2.15, 7.25]} />
      <color attach="background" args={["#0b0a08"]} />
      <fog attach="fog" args={["#0b0a08", 13.5, 25]} />
      <ambientLight color="#c8c5ba" intensity={1.04} />
      <hemisphereLight args={["#e6dfd1", "#1b1914", 1.2]} />
      <spotLight
        position={[0, 6.0, 3.35]}
        angle={0.49}
        penumbra={0.8}
        intensity={86}
        color="#ede7da"
        castShadow
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
      />
      <spotLight
        position={[-2.75, 4.55, 4.45]}
        angle={0.38}
        penumbra={0.78}
        intensity={47}
        color="#d4b274"
        castShadow
      />
      <spotLight
        position={[2.9, 3.75, 3.65]}
        angle={0.42}
        penumbra={0.9}
        intensity={23}
        color="#e8d5ae"
      />
      <pointLight position={[-2.6, 2.6, 3.2]} color="#ccb98f" intensity={5.2} distance={6.8} />
      <pointLight position={[2.7, 2.5, 3.0]} color="#d5c6a9" intensity={4.4} distance={6.4} />
      <pointLight position={[0, 3.35, 1.12]} color={SIGNAL} intensity={6.2} distance={5.5} />
      <pointLight position={[-3.55, 3.15, -1.55]} color="#c9b078" intensity={4.2} distance={5.4} decay={1.45} />
      <pointLight position={[3.55, 3.15, -1.55]} color="#c2aa78" intensity={3.8} distance={5.4} decay={1.45} />
      <pointLight position={[-4.45, 3.2, 2.2]} color="#d9c28c" intensity={4.2} distance={5.8} />
      <pointLight position={[4.45, 3.2, 2.2]} color="#d9c28c" intensity={4.2} distance={5.8} />
      <pointLight position={[-3.85, 2.75, -8.65]} color="#d3b378" intensity={11} distance={6.5} decay={1.6} />
      <pointLight position={[3.85, 2.75, -8.65]} color="#d3b378" intensity={11} distance={6.5} decay={1.6} />
      <pointLight position={[0, 0.68, 2.25]} color="#b7a277" intensity={4.0} distance={4.2} />
      <rectAreaLight
        position={[0, 4.65, 3.35]}
        rotation={[-Math.PI / 2, 0, 0]}
        color="#cfc1a8"
        intensity={5.4}
        width={5.6}
        height={3.2}
      />
      <Cabinet
        game={game}
        phase={phase}
        surfaceMaps={architecturalMaps?.cabinet}
      />

      <mesh position={[0, 0, 1]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <planeGeometry args={[11.8, 24]} />
        <MeshReflectorMaterial
          key={pbrMaps.floor.roughness.uuid}
          blur={[170, 54]}
          resolution={1024}
          mixBlur={0.72}
          mixStrength={1.82}
          mirror={0.4}
          depthScale={0.62}
          minDepthThreshold={0.42}
          maxDepthThreshold={1.45}
          color="#67666d"
          map={pbrMaps.floor.albedo}
          normalMap={pbrMaps.floor.normal}
          normalScale={FLOOR_NORMAL_SCALE}
          roughness={0.74}
          roughnessMap={pbrMaps.floor.roughness}
          bumpMap={pbrMaps.floor.height}
          bumpScale={0.062}
          metalness={0.05}
        />
      </mesh>

      <mesh position={[0, 3.05, -9.72]} receiveShadow>
        <boxGeometry args={[10.15, 6.1, 0.18]} />
        <meshStandardMaterial
          color="#45413b"
          map={pbrMaps.wall.albedo}
          normalMap={pbrMaps.wall.normal}
          normalScale={WALL_NORMAL_SCALE}
          roughness={0.9}
          roughnessMap={pbrMaps.wall.roughness}
          bumpMap={pbrMaps.wall.height}
          bumpScale={0.022}
          metalness={0.02}
        />
      </mesh>
      <mesh position={[-5.12, 3.05, 1]} receiveShadow>
        <boxGeometry args={[0.18, 6.1, 24]} />
        <meshStandardMaterial
          color="#3b3936"
          map={pbrMaps.wall.albedo}
          normalMap={pbrMaps.wall.normal}
          normalScale={WALL_NORMAL_SCALE}
          roughness={0.91}
          roughnessMap={pbrMaps.wall.roughness}
          bumpMap={pbrMaps.wall.height}
          bumpScale={0.02}
          metalness={0.02}
        />
      </mesh>
      <mesh position={[5.12, 3.05, 1]} receiveShadow>
        <boxGeometry args={[0.18, 6.1, 24]} />
        <meshStandardMaterial
          color="#3b3936"
          map={pbrMaps.wall.albedo}
          normalMap={pbrMaps.wall.normal}
          normalScale={WALL_NORMAL_SCALE}
          roughness={0.91}
          roughnessMap={pbrMaps.wall.roughness}
          bumpMap={pbrMaps.wall.height}
          bumpScale={0.02}
          metalness={0.02}
        />
      </mesh>

      <mesh position={[0, 6.08, 1]} receiveShadow>
        <boxGeometry args={[10.15, 0.16, 24]} />
        <meshStandardMaterial
          color="#2b2a28"
          map={pbrMaps.wall.albedo}
          normalMap={pbrMaps.wall.normal}
          normalScale={WALL_NORMAL_SCALE}
          roughness={0.9}
          roughnessMap={pbrMaps.wall.roughness}
          bumpMap={pbrMaps.wall.height}
          bumpScale={0.014}
          metalness={0.02}
        />
      </mesh>
      {[-4.12, -2.47, -0.82, 0.82, 2.47, 4.12].map((x) => (
        <mesh key={`back-panel-${x}`} position={[x, 3.05, -9.613]} receiveShadow>
          <boxGeometry args={[1.36, 5.72, 0.035]} />
          <meshStandardMaterial
            color="#504b43"
            map={pbrMaps.wall.albedo}
            normalMap={pbrMaps.wall.normal}
            normalScale={WALL_NORMAL_SCALE}
            roughness={0.93}
            roughnessMap={pbrMaps.wall.roughness}
            bumpMap={pbrMaps.wall.height}
            bumpScale={0.024}
            metalness={0.015}
          />
        </mesh>
      ))}
      {[-1, 1].flatMap((side) =>
        [-8.6, -6.2, -3.8, -1.4, 1, 3.4, 5.8, 8.2, 10.6].map((z) => (
          <mesh key={`side-panel-${side}-${z}`} position={[side * 5.013, 3.05, z]} receiveShadow>
            <boxGeometry args={[0.035, 5.72, 1.72]} />
            <meshStandardMaterial
              color="#47433d"
              map={pbrMaps.wall.albedo}
              normalMap={pbrMaps.wall.normal}
              normalScale={WALL_NORMAL_SCALE}
              roughness={0.92}
              roughnessMap={pbrMaps.wall.roughness}
              bumpMap={pbrMaps.wall.height}
              bumpScale={0.022}
              metalness={0.015}
            />
          </mesh>
        )),
      )}
      {[-1, 1].map((side) => (
        <group key={`sconce-${side}`} position={[side * 4.91, 3.28, 1.4]}>
          <mesh rotation={[0, side > 0 ? -Math.PI / 2 : Math.PI / 2, 0]}>
            <boxGeometry args={[0.52, 0.09, 0.035]} />
            <meshBasicMaterial color={SIGNAL} toneMapped={false} />
          </mesh>
        </group>
      ))}
      <mesh position={[0, 4.65, -0.18]}>
        <cylinderGeometry args={[0.42, 0.5, 0.12, 40]} />
        <meshStandardMaterial color="#353738" roughness={0.26} metalness={0.82} />
      </mesh>
      <mesh position={[0, 4.58, -0.18]} rotation={[Math.PI / 2, 0, 0]}>
        <circleGeometry args={[0.39, 40]} />
        <meshBasicMaterial color="#f3efe1" toneMapped={false} />
      </mesh>
      {[-8.9, -5.4, -1.9, 1.6, 5.1, 8.6].map((z) => (
        <mesh key={`ceiling-beam-${z}`} position={[0, 5.94, z]} castShadow receiveShadow>
          <boxGeometry args={[9.95, 0.13, 0.16]} />
          <meshStandardMaterial color="#3f3a32" roughness={0.52} metalness={0.42} />
        </mesh>
      ))}
      <ContactShadows position={[0, 0.018, 0.08]} opacity={0.68} scale={5.2} blur={2.1} far={3.4} />
      {[-1, 1].map((side) => (
        <mesh key={`base-trim-${side}`} position={[side * 4.99, 0.12, 1]}>
          <boxGeometry args={[0.06, 0.22, 23.72]} />
          <meshStandardMaterial color="#4f493e" roughness={0.42} metalness={0.5} />
        </mesh>
      ))}
      <mesh position={[0, 0.12, -9.59]}>
        <boxGeometry args={[9.9, 0.22, 0.06]} />
        <meshStandardMaterial color="#4f493e" roughness={0.42} metalness={0.5} />
      </mesh>
    </>
  );
}

export function ArcadeScene(props: ArcadeSceneProps) {
  return (
    <Canvas
      shadows
      dpr={[1, 1.65]}
      gl={{ antialias: true, powerPreference: "high-performance" }}
      onCreated={({ gl }) => {
        gl.toneMapping = THREE.ACESFilmicToneMapping;
        gl.toneMappingExposure = 1.7;
      }}
      className="arcade-canvas"
    >
      <SceneRig {...props} />
    </Canvas>
  );
}
