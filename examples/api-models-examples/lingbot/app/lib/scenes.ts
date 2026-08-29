// Curated starting scenes for the Lingbot demo.
//
// Each scene bundles together a reference image and an initial prompt.
// Lingbot requires BOTH a prompt AND an image before generation can
// start, so the natural "one click to begin" surface is an
// image+prompt pair.
//
// PROMPT STYLE — why each prompt is a long paragraph, not a tagline:
//
// Lingbot produces dramatically more coherent scenes when prompts
// describe the subject, the environment, the camera framing AND the
// motion style in full. Single-sentence prompts produce visually
// unstable output because the model has to invent everything else
// from scratch each chunk.
//
// Each scene's prompt also explicitly describes a strict centred
// third-person rear-view camera — the camera tracks the subject from
// behind, and arrow-key look-input turns the SUBJECT'S heading
// rather than orbiting the camera. That phrasing is what lets the
// model interpret the realtime movement commands (set_movement,
// set_look_horizontal, set_look_vertical) as character / vehicle
// actions, not as edits to the camera.
//
// The prompts are adapted from the curated scenarios in
// reactor-team/public-demos.

export interface Scene {
  id: string;
  label: string;
  description: string;
  imageUrl: string;
  prompt: string;
}

export const SCENES: ReadonlyArray<Scene> = [
  {
    id: "dragon_ride",
    label: "Dragon Flight",
    description: "Third-person flight over a jungle castle",
    imageUrl: "/images/dragon_ride.jpg",
    prompt:
      "This is a third-person-view video of a colossal dragon — its neck a column of dense obsidian-black scales rippling with muscle, its enormous bat-like wings veined with pulsing crimson, leading edges sharp and dark. Ahead, a towering ancient castle of crumbling stone spires and moss-covered battlements rises above dense jungle canopy, gothic arches half-swallowed by creeping vines. A vast primordial forest stretches in every direction, pale mist drifting between trunks, dappled golden sunlight filtering through humid air, winding river gorges far below. Strict centred third-person rear view: the dragon is locked at the exact centre of the frame. The camera tracks the dragon from above and behind as it moves forward and never rotates around it; arrow-key look-input turns the dragon's heading instead, preserving the rear-view framing. The dragon's wings beat the air with powerful rhythmic strokes — sweeping high on the upstroke until the membranes nearly meet overhead, then driving downward in a thunderous arc. The neck stretches forward into the wind and air streams past the leading edges of the wings as the dragon drives forward through the sky.",
  },
  {
    id: "spring_valley",
    label: "Spring Valley",
    description: "Golden retriever in a watercolor meadow",
    imageUrl: "/images/spring_valley.png",
    prompt:
      "This is a third-person-view video of a golden retriever in a sun-warmed meadow beside a cozy encampment. To the left lies a patterned floral rug on the grass, to the right a weathered stone bench bearing an open book and a small potted seedling, a cardboard box of chew bones at its corner, a lamppost and a hand-lettered wooden signpost flanking the encampment. Wildflower fields roll down toward the open valley ahead, where pink cherry blossoms in full bloom stand among rounded green oaks. Soft layered hills rest against a pale blue spring sky scattered with cottony clouds. Tender hand-painted watercolor storybook atmosphere. Strict centred third-person over-the-shoulder rear view: the retriever is locked at the exact centre of the frame. The camera tracks the retriever from behind as it travels forward and never rotates around the dog; arrow-key look-input turns the retriever's heading instead, preserving the over-the-shoulder framing. The retriever trots forward through the warm meadow grass at an easy pace, paws padding through wildflowers, ears bouncing lightly with each step, tail swinging in a relaxed wag, tongue lolling out the side of its mouth.",
  },
  {
    id: "misted_kingdom",
    label: "Misted Kingdom",
    description: "Sword-slung rider on horseback toward a far castle",
    imageUrl: "/images/misted_kingdom.png",
    prompt:
      "This is a third-person-view video of a sword-slung rider in white tunic and dark sash, hair tied in a high topknot, seated firmly on a brown horse, in an open valley of wildflower meadows. Violet lupines and crimson poppies stretch between weathered boulders; a hamlet of half-timbered cottages, stone watchtowers, and a moss-eaten ruined portal arch stands in the misted middle distance. Far ahead on a craggy peak, a many-spired castle with crimson pennants stands against a vast ringed gas giant and a pale crescent moon in the peach-tinted twilight sky. Painterly fantasy storybook atmosphere. Strict centred third-person over-the-shoulder rear view: the rider and horse are locked at the exact centre of the frame. The camera tracks the pair from behind and slightly above the rider as they travel forward and never rotates around them; arrow-key look-input turns the horse's heading instead, preserving the over-the-shoulder framing. The horse moves forward at a steady walk through the wildflower meadow, hooves padding through lupines and poppies and stirring the valley mist into thin curling wisps, its long braided tail swaying behind. The rider's tunic and sash shift gently with the horse's motion, and the topknot bobs with each step.",
  },
  {
    id: "citadel_approach",
    label: "Citadel Approach",
    description: "Vintage 4x4 rolling through a coral desert canyon",
    imageUrl: "/images/citadel_approach.png",
    prompt:
      "This is a third-person-view video of a battered grey-green vintage Defender 4x4 deep in a coral-lit desert canyon. Prickly pear cacti tipped with magenta blooms, scattered red poppies, and weather-pitted boulders dot the open desert floor; smooth ochre dunes sweep up toward towering sandstone mesas that wall the valley on the left. Ahead, a cliff-built sandstone citadel of white-washed houses, crenellated battlements, and slender minarets stands against a hazy peach-orange sunset sky. Warm painterly desert storybook atmosphere. Strict centred third-person rear view: the Defender is locked at the exact centre of the frame at all times — horizontally centred, vertically centred, and the camera sits on a fixed offset directly behind the vehicle's rear axle. The camera tracks the Defender from directly behind as it travels forward and never rotates, orbits, or pans around it under any input. Arrow-key look-input turns the Defender's heading instead, so the rear-view framing is preserved frame-by-frame. The Defender rolls forward across the open desert sand, plumes of pale golden dust kicking up from its tires and trailing behind the rear hatch, the suspension flexing softly as it crests the dunes, faint heat shimmer rising from the tailpipe.",
  },
  {
    id: "storm_crossing",
    label: "Storm Crossing",
    description: "Work boat punching through dark north-sea swells",
    imageUrl: "/images/storm_crossing.png",
    prompt:
      "This is a third-person-view video of a battered grey aluminum-hulled work boat — weather-stripped wheelhouse glowing with a single warm yellow cabin lamp, twin whip antennas above, the open wooden deck damp with spray. Slate-grey swells roll under the hull, cool salt mist drifting around the cabin glass. Overhead, dense dark storm clouds fill the entire sky from horizon to horizon, a low heavy charcoal-black ceiling pressing down in oppressive layers and draining the sea of color. Cinematic photorealistic gloomy overcast seascape atmosphere. Strict centred third-person stern view: the boat is locked at the exact centre of the frame with its square transom (flat stern) always facing directly toward the camera and the bow pointing away into the far distance — the camera stays directly behind the boat and never moves to a forward, side, or beam-on angle. The camera tracks the boat from directly astern as it travels through the seas, and rotates with the hull to keep the boat locked at the exact centre of the frame when arrow-key look-input turns the boat's heading. The engines drive at full power, the bow rising over each swell and crashing down into the next trough, a churning white wake boiling astern, salt mist torn back from the deck as the boat punches forward through the seas.",
  },
];

/** Look up a scene by id. */
export function findSceneById(id: string | null | undefined): Scene | null {
  if (!id) return null;
  return SCENES.find((s) => s.id === id) ?? null;
}
