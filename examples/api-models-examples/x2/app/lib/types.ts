// The input source the client streams into the model. All three publish to
// the `source` track — "webcam" sends the live webcam, "video" plays a chosen
// clip and streams its frames in, and "image" repeats a still image as a
// constant 24 fps stream (the drag-to-animate setup from the model's demos:
// the picture stays put until the pointer steers the subject). The model
// edits whatever arrives on `source`; the app never uploads source media,
// it streams it.
export type X2SourceMode = "webcam" | "video" | "image";

// The UI's projection of the model's session, reduced from the `state_update`
// snapshot the model broadcasts on connect and after every observable state
// change — the model is the source of truth. `referenceAccepted` (the decoded
// image's dimensions) is the one field carried from a discrete ack message
// (`reference_image_accepted`), since the snapshot only says whether a
// reference is set.
export interface X2UiState {
  /** Whether a generation run is currently producing `main_video` frames. */
  generating: boolean;
  /** The active editing instruction, or null when none is set. */
  activePrompt: string | null;
  /** Output resolution, fixed for the session once the first run starts. */
  outputWidth: number | null;
  outputHeight: number | null;
  /** Whether a reference image is currently conditioning generation. */
  hasReference: boolean;
  /** Current source-frame policy (see the set_keep_backlog command). */
  keepBacklog: boolean;
  /** Last reference_image_accepted ack ({width, height} of the decoded image). */
  referenceAccepted: { width: number; height: number } | null;
}

export const DEFAULT_UI_STATE: X2UiState = {
  generating: false,
  activePrompt: null,
  outputWidth: null,
  outputHeight: null,
  hasReference: false,
  keepBacklog: false,
  referenceAccepted: null,
};

// The drag pointer, echoed by the model in every `state_update`. Deliberately
// kept out of X2UiState: it changes ~30 Hz while a drag is in flight, so
// folding it into the shared snapshot would defeat reduce()'s bail-out and
// re-render the whole workspace every frame. Only the Pointer readout reads
// it, and it subscribes to the stream on its own (see PointerPanel).
export interface X2PointerReadout {
  /** Drag-pointer position, normalized to the output frame (0..1). */
  x: number;
  y: number;
  /** Whether a drag gesture is currently steering the edited subject. */
  active: boolean;
}

export const DEFAULT_POINTER_READOUT: X2PointerReadout = {
  // The model's documented defaults: centered, not pressed.
  x: 0.5,
  y: 0.5,
  active: false,
};
