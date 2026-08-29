// Preset storyboards for the LongLive 2 director example.
//
// A storyboard is an ordered list of beats: the first is the opening shot
// (fired with `set_shot` before `start`), the rest are scheduled against the
// cumulative `session_chunk` clock with `schedule_shot` (soft, same scene) or
// `schedule_scene_cut` (hard, new scene + fresh 48-chunk budget).
//
// Keep each scene's beats inside its 48-chunk budget, or put a cut before the
// ceiling — a beat scheduled past where its scene auto-completes never fires.
//
// PROMPT STYLE: LongLive conditions weakly on terse prompts and the output
// degrades. Write dense, cinematic paragraphs — subject + wardrobe/detail,
// action, setting, time of day, lighting, lens + camera move, color palette,
// texture, and mood. Within a scene, soft shots re-establish the same subject
// and world so continuity holds; cuts describe a wholly new world.

import type { BeatKind } from "./storyboard-store";

export interface StoryboardBeat {
  kind: BeatKind;
  prompt: string;
  /** Absolute session chunk. The opener is 0. ~1.2s per chunk. */
  atChunk: number;
}

export interface PresetStoryboard {
  id: string;
  label: string;
  description: string;
  beats: ReadonlyArray<StoryboardBeat>;
}

export const STORYBOARDS: ReadonlyArray<PresetStoryboard> = [
  {
    id: "astronaut",
    label: "Martian outpost",
    description:
      "A drone reveal of the hero's hab, in through the front door to suit up, a wave hello, then off into the empty desert.",
    beats: [
      {
        kind: "shot",
        atChunk: 0,
        prompt:
          "A vast windswept Martian landscape of deep rust-red dunes and wind-carved rock stretching unbroken to the horizon at golden hour, fine ochre dust streaming off the ridgelines, a pale ringed planet hanging low and enormous in the dusty amber sky. Long sweeping aerial drone shot gliding steadily forward over ridge after ridge, wide-angle, deep focus, slow majestic continuous move, warm crimson-and-amber palette, volumetric haze, photoreal, 35mm film look.",
      },
      {
        kind: "cut",
        atChunk: 5,
        prompt:
          "A high, wide aerial view across the rust-red Martian basin with a low, spartan futuristic research habitat sitting far off in the distance, small but clearly visible on the open plain — a cluster of interlocking white composite-panel modules and cylindrical pods, an angled bank of dark solar arrays, and a single slender comms mast, dwarfed by the surrounding dunes. Slow high drone glide holding the distant outpost in frame, ultra-wide, deep focus, raking golden light across the basin, warm amber palette, volumetric haze, photoreal, 35mm film look.",
      },
      {
        kind: "cut",
        atChunk: 10,
        prompt:
          "A ground-level forward-facing tracking shot on the Martian surface that begins on a medium shot of a spartan futuristic research habitat — the whole low cluster of white composite-panel modules visible across the rust-red regolith — then travels steadily forward, closing the distance toward the front airlock door, the ribbed hatch and its lit control panel growing from small to large in frame as the camera covers ground toward the entrance. Eye-level wide-angle, deep focus, pronounced forward dolly, immersive and grounded, warm crimson-and-amber palette, volumetric haze, photoreal, 35mm film look.",
      },
      {
        kind: "cut",
        atChunk: 15,
        prompt:
          "Interior of a clean, spartan Martian habitat airlock bay: a lone astronaut stands at an open equipment rack methodically suiting up, pulling on a worn white-and-orange pressure suit and latching the chest panel, helmet waiting on the bench beside him. Cool white LED light from overhead, a small porthole glowing with red Martian daylight. Medium close-up, 35mm lens, shallow depth of field, slow push-in, muted steel-and-white palette with warm accents, photoreal, cinematic.",
      },
      {
        kind: "cut",
        atChunk: 20,
        prompt:
          "Interior tracking shot following an astronaut in a worn white-and-orange pressure suit, now fully suited with the helmet on, walking purposefully down a narrow corridor inside the spartan Martian habitat toward the airlock at the far end, clean white composite walls and softly glowing LED strip lighting lining the passage, cabling and equipment racks set along the panels. Steady forward Steadicam tracking from behind, 35mm lens, shallow depth of field, cool white interior light with warm accents, muted steel-and-white palette, photoreal, cinematic.",
      },
      {
        kind: "cut",
        atChunk: 25,
        prompt:
          "Exterior on the Martian surface: the astronaut in the worn white-and-orange pressure suit stands directly in front of his spartan research habitat, the white composite-panel modules, recessed portholes, angled solar arrays and slender comms mast rising clearly directly behind him, ochre dust drifting past his boots across the rust-red ground. Golden low sunlight rakes across suit and structure alike, wide cinematic shot, long lens, shallow depth of field, slow dolly, warm crimson-and-amber palette, volumetric haze, photoreal, 35mm film look.",
      },
      {
        kind: "shot",
        atChunk: 30,
        prompt:
          "The same astronaut, still standing in front of his habitat on the rust-red Martian surface, raises an arm and waves warmly toward the camera, the white composite-panel modules and slender comms mast rising behind him, ochre dust drifting past his boots. Golden low sunlight rakes across suit and structure, wide cinematic shot, long lens, shallow depth of field, slow dolly, warm crimson-and-amber palette, volumetric haze, photoreal, 35mm film look.",
      },
      {
        kind: "cut",
        atChunk: 35,
        prompt:
          "An astronaut in a worn white-and-orange pressure suit walks across a vast, utterly empty Martian expanse, a single small figure crossing an immense open plain of rust-red dunes with nothing but bare horizon ahead, boots kicking up trailing veils of fine dust. Wide dramatic tracking shot following from behind, long lens, deep focus, the emptiness stretching huge around him, warm crimson-and-amber palette, volumetric haze, photoreal, 35mm film look, epic and cinematic.",
      },
      {
        kind: "cut",
        atChunk: 40,
        prompt:
          "An extreme wide aerial drone shot of the Martian desert, a lone astronaut in a white-and-orange pressure suit reduced to a tiny speck far below crossing the colossal sweep of rust-red dunes, a single track of footprints trailing behind him, the ringed planet vast on the distant horizon. Towering high-altitude drone view craning up and pulling back, ultra-wide, deep focus, the figure dwarfed to almost nothing, warm crimson-and-amber palette, volumetric haze, photoreal, 35mm film look, epic and cinematic.",
      },
    ],
  },
  {
    id: "chef",
    label: "Ratatouille service",
    description:
      "Hard cut after hard cut following one bowl of ratatouille from the cutting board to the guest's table.",
    beats: [
      {
        kind: "shot",
        atChunk: 0,
        prompt:
          "A focused chef in a crisp white double-breasted jacket rhythmically chops glossy purple eggplant, red and yellow bell peppers and a white onion on a worn wooden board in a warm sunlit professional kitchen, the knife rocking quickly as neat slices fan out, late-morning light through a tall window glinting off the blade and the polished steel behind. Close-up, 50mm lens, shallow depth of field, slow push-in, rich saturated vegetable colors, warm natural palette, photoreal, cinematic.",
      },
      {
        kind: "cut",
        atChunk: 5,
        prompt:
          "Close-up over a black skillet on a gas range as diced eggplant, bell peppers and onion sizzle and toss in glistening olive oil, steam and a lick of flame curling up while a chef's hand shakes the pan and the vegetables leap, their caramelized edges glistening. Warm flame-lit glow from below and soft window light above, macro 50mm, shallow depth of field, lively handheld energy, rich reds, purples and golds, photoreal, cinematic.",
      },
      {
        kind: "cut",
        atChunk: 10,
        prompt:
          "Overhead close-up on a clean white bowl on a marble counter as a chef's hands spoon the finished ratatouille into it, glossy stewed eggplant, peppers and onion arranged in a neat swirl with a drizzle of olive oil and a single basil leaf on top, steam rising gently. Soft directional window light, macro 50mm, shallow depth of field, rich saturated vegetable colors against the white bowl, photoreal, cinematic.",
      },
      {
        kind: "cut",
        atChunk: 15,
        prompt:
          "A waiter in a black apron carries that same clean white bowl of ratatouille — glossy stewed eggplant, peppers and onion in a neat swirl with a drizzle of olive oil and a single basil leaf on top — balanced on a small tray through an elegant warm-lit restaurant dining room, weaving past softly blurred candlelit tables and seated guests. Tracking shot moving alongside him, eye-level, 35mm lens, shallow depth of field, warm amber ambiance with golden bokeh highlights, photoreal, cinematic.",
      },
      {
        kind: "cut",
        atChunk: 20,
        prompt:
          "At a candlelit restaurant table, the waiter sets that same clean white bowl of ratatouille — glossy stewed eggplant, peppers and onion in a neat swirl with a drizzle of olive oil and a single basil leaf on top — down in front of a smiling seated guest, a crisp white tablecloth and a glass of red wine beside it. Medium close-up, 35mm lens, shallow depth of field, warm amber candlelight, elegant and inviting, photoreal, cinematic.",
      },
    ],
  },
  {
    id: "wildlife",
    label: "Wildlife montage",
    description: "Hard cuts across animals in natural settings.",
    beats: [
      {
        kind: "shot",
        atChunk: 0,
        prompt:
          "A powerful Bengal tiger prowls slowly through tall sun-bleached savanna grass at golden hour, muscles rolling beneath its orange-and-black striped fur, amber eyes fixed ahead, whiskers catching the light. Warm low sunlight rakes through the swaying grass. Long telephoto, very shallow depth of field, slow tracking dolly, rich golden-amber palette, fine fur detail, photoreal nature-documentary look.",
      },
      {
        kind: "cut",
        atChunk: 10,
        prompt:
          "A pod of bottlenose dolphins leaps in unison through the glittering turquoise surface of a calm open ocean, sleek grey bodies arcing clear of the water, sheets of spray catching the bright midday sun. Low angle just above the waterline, crisp high-shutter clarity, deep blue-green palette with sparkling highlights, photoreal.",
      },
      {
        kind: "cut",
        atChunk: 20,
        prompt:
          "A herd of African elephants — towering adults sheltering a small calf — wade and drink at a broad watering hole at sunset, trunks curling up to spray arcs of water, acacia trees silhouetted on the horizon. Wide cinematic shot, golden-hour backlight, dust and mist in the air, earthy amber tones, serene and majestic, photoreal.",
      },
      {
        kind: "cut",
        atChunk: 30,
        prompt:
          "A bald eagle soars on broad outstretched wings over snow-capped mountain peaks and dark evergreen forest, its white head turning to scan a glacial valley far below, feathers ruffling in the cold high-altitude wind. Sweeping aerial tracking shot gliding beside the bird, crisp clear mountain light, cool blue-and-white palette, razor-sharp detail, photoreal.",
      },
    ],
  },
];

export function findStoryboard(id: string): PresetStoryboard | null {
  return STORYBOARDS.find((s) => s.id === id) ?? null;
}
