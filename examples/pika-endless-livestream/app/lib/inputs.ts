// The model's raw inputs, as Pika H3 Flex takes them per clip:
//   references: up to 9 images (Picture N), 3 audio clips (Audio N), videos (Video N)
//   frames:     a first frame and/or a last frame
// A clip uses references or frames, never both. References ride on every clip
// through latent continuation; frames apply to one clip, which then cannot
// continue the clip before it.

export const MAX_IMAGES = 9;
export const MAX_AUDIO = 3;
/** With a latent continuation Pika takes at most 2 videos (the previous clip's tail is the third). */
export const MAX_VIDEOS = 2;
export const MAX_AUDIO_SECONDS = 15;
/** With a latent continuation the videos may total 12.67 s (the tail takes 2.33 s). */
export const MAX_VIDEO_SECONDS = 12.67;
export const MAX_FILE_BYTES = 25 * 1024 * 1024;

export type Item = { id: string; file: File; url: string; seconds?: number };

export type Inputs = {
  images: Item[];
  audios: Item[];
  videos: Item[];
  first: Item | null;
  last: Item | null;
};

export const EMPTY_INPUTS: Inputs = { images: [], audios: [], videos: [], first: null, last: null };

export function hasReferences(i: Inputs): boolean {
  return i.images.length + i.audios.length + i.videos.length > 0;
}

export function frameSet(i: Pick<Inputs, "first" | "last">): "first" | "last" | "both" | null {
  if (i.first && i.last) return "both";
  if (i.first) return "first";
  if (i.last) return "last";
  return null;
}

/**
 * Pika's alignment line for a clip with frames, which opens the prompt. The
 * first frame is <Picture 1>; the last is <Picture 2>, or <Picture 1> alone.
 */
export function frameAlignmentLine(frames: "first" | "last" | "both", seconds: number): string {
  const end = seconds.toFixed(2);
  if (frames === "first") return "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.";
  if (frames === "last") return `How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the ${end}-second mark of the target video.`;
  return `How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the 0.00-second mark of the target video; <Picture 2> (from [Shot 1]) aligns with the ${end}-second mark of the target video.`;
}

/** Media duration in seconds, read from the browser's decoder. */
export function mediaSeconds(file: File, kind: "audio" | "video"): Promise<number> {
  return new Promise((resolve, reject) => {
    const el = document.createElement(kind);
    const url = URL.createObjectURL(file);
    el.preload = "metadata";
    el.onloadedmetadata = () => {
      URL.revokeObjectURL(url);
      resolve(el.duration);
    };
    el.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error(`Could not read ${file.name}.`));
    };
    el.src = url;
  });
}

/** A small JPEG of an image, base64 without the data: prefix, for the story writer to look at. */
export async function thumbnail(file: File, max = 512): Promise<string> {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, max / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  canvas.getContext("2d")!.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close();
  return canvas.toDataURL("image/jpeg", 0.8).split(",")[1];
}

/** A plain list of the reference labels, for the writers. */
export function referenceList(i: Pick<Inputs, "images" | "audios" | "videos">): string {
  const lines = [
    ...i.images.map((_, n) => `<Picture ${n + 1}>: an image (shown to you)`),
    ...i.audios.map((a, n) => `<Audio ${n + 1}>: an audio clip, ${a.seconds?.toFixed(1) ?? "?"} s, file "${a.file.name}"`),
    ...i.videos.map((v, n) => `<Video ${n + 1}>: a video clip, ${v.seconds?.toFixed(1) ?? "?"} s, file "${v.file.name}"`),
  ];
  return lines.join("\n");
}
