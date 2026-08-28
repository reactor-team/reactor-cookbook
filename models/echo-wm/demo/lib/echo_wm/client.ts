// Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.

// Generated from the Echo-WM Flash Reactor schema.
// Model: echo-wm-flash v0.0.0

import { FileRef, Reactor } from "@reactor-team/js-sdk";

export { FileRef };

export const MODEL_NAME = "echo-wm-flash" as const;
export const MODEL_VERSION = "v0.0.0" as const;

/** Media tracks declared by the Echo-WM Flash schema. */
export const EchoWmFlashTracks = [
  { name: "main_video", kind: "video", direction: "recvonly" },
  { name: "main_audio", kind: "audio", direction: "recvonly" },
] as const;

export type EchoWmFlashRecvTrackName = "main_video" | "main_audio";

export interface EchoWmFlashSetImageParams {
  image: FileRef;
  prompt?: string;
  seed?: number;
}

export interface EchoWmFlashSetPromptParams {
  prompt?: string;
}

export interface EchoWmFlashSetCameraMotionParams {
  forward?: number;
  strafe?: number;
  pitch?: number;
  yaw?: number;
}

export interface EchoWmFlashSetFovParams {
  fov_degrees?: number;
}

export interface EchoWmFlashResetParams {
  seed?: number;
}

export interface EchoWmFlashStateUpdateMessage {
  type: "state_update";
  image_source: "uploaded" | "built_in" | null;
  image_name: string | null;
  prompt: string | null;
  active_prompt: string | null;
  seed: number;
  reset_queued: boolean;
  generating: boolean;
  completed_chunks: number;
  next_chunk: number | null;
  max_chunks: number;
  forward: number;
  strafe: number;
  pitch: number;
  yaw: number;
  fov_degrees: number;
}

export interface EchoWmFlashImageSelectedMessage {
  type: "image_selected";
  source: "uploaded" | "built_in";
  filename: string;
  prompt: string;
  seed: number;
}

export interface EchoWmFlashPromptQueuedMessage {
  type: "prompt_queued";
  prompt: string;
  applies_to_chunk: number;
}

export interface EchoWmFlashCameraMotionChangedMessage {
  type: "camera_motion_changed";
  forward: number;
  strafe: number;
  pitch: number;
  yaw: number;
  fov_degrees: number;
  applies_to_chunk: number | null;
}

export interface EchoWmFlashRolloutResetQueuedMessage {
  type: "rollout_reset_queued";
  seed: number;
  replaced_chunks: number;
}

export interface EchoWmFlashChunkCompletedMessage {
  type: "chunk_completed";
  chunk: number;
  video_frames: number;
  audio_samples: number;
  generation_seconds: number;
  denoise_seconds: number | null;
  cache_commit_seconds: number | null;
  video_decode_seconds: number | null;
  audio_decode_seconds: number | null;
  cuda_total_seconds: number | null;
  prompt: string;
  forward: number;
  strafe: number;
  pitch: number;
  yaw: number;
}

export interface EchoWmFlashAutomaticResetQueuedMessage {
  type: "automatic_reset_queued";
  completed_chunks: number;
  max_chunks: number;
  seed: number;
}

export type EchoWmFlashMessage =
  | EchoWmFlashStateUpdateMessage
  | EchoWmFlashImageSelectedMessage
  | EchoWmFlashPromptQueuedMessage
  | EchoWmFlashCameraMotionChangedMessage
  | EchoWmFlashRolloutResetQueuedMessage
  | EchoWmFlashChunkCompletedMessage
  | EchoWmFlashAutomaticResetQueuedMessage;

export type EchoWmFlashOptions = Omit<
  ConstructorParameters<typeof Reactor>[0],
  "modelName" | "modelTracks"
>;

function unwrapMessage<T>(raw: unknown): T {
  const envelope = raw as { type?: string; data?: Record<string, unknown> };
  if (
    envelope &&
    typeof envelope === "object" &&
    envelope.data &&
    typeof envelope.data === "object"
  ) {
    return { ...envelope.data, type: envelope.type } as T;
  }
  return raw as T;
}

/** Strongly typed standalone client for Echo-WM Flash. */
export class EchoWmFlashModel extends Reactor {
  constructor(options?: EchoWmFlashOptions) {
    super({
      ...options,
      modelName: MODEL_NAME,
      modelTracks: [...EchoWmFlashTracks],
    });
  }

  async setImage(params: EchoWmFlashSetImageParams): Promise<void> {
    await this.sendCommand("set_image", params);
  }

  async randomImage(): Promise<void> {
    await this.sendCommand("random_image", {});
  }

  async setPrompt(params: EchoWmFlashSetPromptParams): Promise<void> {
    await this.sendCommand("set_prompt", params);
  }

  async setCameraMotion(
    params: EchoWmFlashSetCameraMotionParams,
  ): Promise<void> {
    await this.sendCommand("set_camera_motion", params);
  }

  async releaseCamera(): Promise<void> {
    await this.sendCommand("release_camera", {});
  }

  async setFov(params: EchoWmFlashSetFovParams): Promise<void> {
    await this.sendCommand("set_fov", params);
  }

  async reset(params: EchoWmFlashResetParams): Promise<void> {
    await this.sendCommand("reset", params);
  }

  onMessage(handler: (message: EchoWmFlashMessage) => void): () => void {
    const wrapped = (raw: unknown) => handler(unwrapMessage(raw));
    this.on("message", wrapped);
    return () => this.off("message", wrapped);
  }

  onMainVideo(
    handler: (track: MediaStreamTrack, stream: MediaStream) => void,
  ): () => void {
    const wrapped = (
      name: string,
      track: MediaStreamTrack,
      stream: MediaStream,
    ) => {
      if (name === "main_video") handler(track, stream);
    };
    this.on("trackReceived", wrapped);
    return () => this.off("trackReceived", wrapped);
  }

  onMainAudio(
    handler: (track: MediaStreamTrack, stream: MediaStream) => void,
  ): () => void {
    const wrapped = (
      name: string,
      track: MediaStreamTrack,
      stream: MediaStream,
    ) => {
      if (name === "main_audio") handler(track, stream);
    };
    this.on("trackReceived", wrapped);
    return () => this.off("trackReceived", wrapped);
  }
}
