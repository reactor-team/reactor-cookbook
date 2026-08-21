export type FrameSignature = {
  cells: number[];
  edges: number[];
  averageLuma: number;
  contrast: number;
};

export type CapturedFrame = {
  blob: Blob;
  signature: FrameSignature;
};

const SAMPLE_WIDTH = 96;
const SAMPLE_HEIGHT = 54;
const GRID_COLUMNS = 8;
const GRID_ROWS = 5;

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function createSampleCanvas() {
  const canvas = document.createElement("canvas");
  canvas.width = SAMPLE_WIDTH;
  canvas.height = SAMPLE_HEIGHT;
  return canvas;
}

function signatureFromCanvas(canvas: HTMLCanvasElement): FrameSignature {
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("A 2D canvas context is unavailable.");
  const pixels = context.getImageData(0, 0, SAMPLE_WIDTH, SAMPLE_HEIGHT).data;
  const cells: number[] = [];
  const edges: number[] = [];
  const lumas: number[] = [];
  const cellWidth = SAMPLE_WIDTH / GRID_COLUMNS;
  const cellHeight = SAMPLE_HEIGHT / GRID_ROWS;

  for (let row = 0; row < GRID_ROWS; row += 1) {
    for (let column = 0; column < GRID_COLUMNS; column += 1) {
      let red = 0;
      let green = 0;
      let blue = 0;
      let luma = 0;
      let edge = 0;
      let count = 0;
      const startX = Math.floor(column * cellWidth);
      const endX = Math.floor((column + 1) * cellWidth);
      const startY = Math.floor(row * cellHeight);
      const endY = Math.floor((row + 1) * cellHeight);
      for (let y = startY; y < endY; y += 2) {
        for (let x = startX; x < endX; x += 2) {
          const index = (y * SAMPLE_WIDTH + x) * 4;
          const pixelLuma =
            pixels[index] * 0.2126 +
            pixels[index + 1] * 0.7152 +
            pixels[index + 2] * 0.0722;
          red += pixels[index];
          green += pixels[index + 1];
          blue += pixels[index + 2];
          luma += pixelLuma;
          if (x + 2 < SAMPLE_WIDTH) {
            const next = index + 8;
            const nextLuma =
              pixels[next] * 0.2126 +
              pixels[next + 1] * 0.7152 +
              pixels[next + 2] * 0.0722;
            edge += Math.abs(pixelLuma - nextLuma);
          }
          count += 1;
        }
      }
      const safeCount = Math.max(1, count);
      cells.push(red / safeCount, green / safeCount, blue / safeCount);
      edges.push(edge / safeCount);
      lumas.push(luma / safeCount);
    }
  }

  const averageLuma = lumas.reduce((sum, value) => sum + value, 0) / lumas.length;
  const contrast = Math.sqrt(
    lumas.reduce((sum, value) => sum + (value - averageLuma) ** 2, 0) /
      lumas.length,
  );
  return { cells, edges, averageLuma, contrast };
}

export function compareFrameSignatures(
  reference: FrameSignature,
  candidate: FrameSignature,
) {
  const colorError =
    reference.cells.reduce(
      (sum, value, index) => sum + Math.abs(value - candidate.cells[index]),
      0,
    ) /
    reference.cells.length /
    255;
  const edgeError =
    reference.edges.reduce(
      (sum, value, index) => sum + Math.abs(value - candidate.edges[index]),
      0,
    ) /
    reference.edges.length /
    96;
  const lumaError = Math.abs(reference.averageLuma - candidate.averageLuma) / 255;
  const contrastError = Math.abs(reference.contrast - candidate.contrast) / 96;
  const error =
    colorError * 0.5 +
    clamp(edgeError, 0, 1) * 0.26 +
    lumaError * 0.14 +
    clamp(contrastError, 0, 1) * 0.1;
  return Math.round(clamp((1 - error) * 100, 0, 100));
}

export async function signatureFromImageBlob(blob: Blob) {
  const canvas = createSampleCanvas();
  const context = canvas.getContext("2d");
  if (!context) throw new Error("A 2D canvas context is unavailable.");
  const bitmap = await createImageBitmap(blob);
  context.drawImage(bitmap, 0, 0, SAMPLE_WIDTH, SAMPLE_HEIGHT);
  bitmap.close();
  return signatureFromCanvas(canvas);
}

export async function captureWorldFrame(): Promise<CapturedFrame | null> {
  const video = document.querySelector<HTMLVideoElement>(
    "video.world-video, .world-video video, .game-shell video",
  );
  if (!video || video.readyState < 2 || !video.videoWidth || !video.videoHeight) {
    return null;
  }
  const captureCanvas = document.createElement("canvas");
  captureCanvas.width = Math.min(960, video.videoWidth);
  captureCanvas.height = Math.round(
    captureCanvas.width * (video.videoHeight / video.videoWidth),
  );
  const captureContext = captureCanvas.getContext("2d");
  if (!captureContext) return null;
  captureContext.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
  const sampleCanvas = createSampleCanvas();
  const sampleContext = sampleCanvas.getContext("2d");
  if (!sampleContext) return null;
  sampleContext.drawImage(captureCanvas, 0, 0, SAMPLE_WIDTH, SAMPLE_HEIGHT);
  const signature = signatureFromCanvas(sampleCanvas);
  const blob = await new Promise<Blob | null>((resolve) =>
    captureCanvas.toBlob(resolve, "image/png"),
  );
  return blob ? { blob, signature } : null;
}
