// Curated "world events" the user can throw at the live scene.
//
// Each entry is a self-contained sentence describing an environmental
// or atmospheric change ("rain begins", "fog rolls in"). The
// DynamicEvents panel appends one of these to the active prompt and
// hot-swaps the model's prompt mid-stream via `set_prompt`. Lingbot
// picks up the new prompt on the next chunk and the scene visibly
// changes — no restart, no flash, the reference image stays put.
//
// Authoring rules (so events compose cleanly with any starting scene):
//
//   1. One sentence per event. Anything longer competes with the
//      starting prompt and produces garbled output. Anything shorter
//      gives the model too little to lock onto.
//
//   2. Describe ATMOSPHERE, not the subject. The starting prompt has
//      already framed the subject (dragon, retriever, rider, 4x4,
//      boat) and the camera. World events stay in the environmental
//      layer — weather, light, sky, time-of-day — so they slot onto
//      any subject without contradicting it.
//
//   3. Write in the present continuous, matching the starting prompt's
//      voice. "Rain begins to fall…" composes; "It rained yesterday."
//      does not.

export interface DynamicEvent {
  id: string;
  /** Short label shown on the button. */
  label: string;
  /** Single emoji used as the button icon. Decorative — no a11y role. */
  icon: string;
  /** Sentence appended to the base prompt when this event is active. */
  text: string;
}

export const DYNAMIC_EVENTS: ReadonlyArray<DynamicEvent> = [
  {
    id: "rain",
    label: "Rain begins",
    icon: "🌧️",
    text: "Rain begins to fall softly across the scene, droplets catching the light and beading on every surface, the air growing humid and the ground darkening as water gathers in puddles.",
  },
  {
    id: "snow",
    label: "Snow falls",
    icon: "❄️",
    text: "Soft snowflakes drift down across the scene, settling on every surface and muffling the world into a quiet, white-edged stillness with cool blue shadows pooling under each object.",
  },
  {
    id: "fog",
    label: "Fog rolls in",
    icon: "🌫️",
    text: "A thick fog rolls in across the scene, softening every silhouette into a pale haze and shrinking the visible world to a few metres around the subject, sounds growing muffled and distant.",
  },
  {
    id: "sunset",
    label: "Golden sunset",
    icon: "🌇",
    text: "The sky deepens to gold and amber as the sun sinks low, casting long warm shadows across the scene and bathing every surface in honeyed light that catches the edges of each silhouette.",
  },
  {
    id: "night",
    label: "Night falls",
    icon: "🌙",
    text: "Night falls over the scene as the sky deepens to indigo and a scattering of stars emerges, with cool moonlight rimming every silhouette in silver and the air growing crisp.",
  },
  {
    id: "storm",
    label: "Storm strikes",
    icon: "⚡",
    text: "A dramatic thunderstorm cracks overhead, sheets of rain hammering the scene as forked lightning briefly washes everything in stark blue-white light and dark rolling clouds churn across the sky.",
  },
];
