// Curated scenes for the Helios demo.
//
// Each scene is a self-contained narrative: one `initial` prompt that
// starts the scene plus a small list of `evolutions` that continue or
// pivot it. The frontend uses the same list in two places:
//
//   - Setup phase (PromptComposer + ImageStarter) reads `initial` to
//     populate the "Try a prompt" presets and example image cards.
//   - Live phase (EvolveScene) matches the active `current_prompt`
//     against `initial` and `evolutions` to find which scene the
//     session belongs to, then renders that scene's evolutions as
//     hot-swap suggestions.
//
// PROMPT STYLE — why each prompt is a long paragraph, not a tagline:
//
// Helios produces dramatically smoother scenes when prompts describe
// the subject, the action, the environment, the lighting, AND the
// camera shot in full. Equally important: each evolution prompt
// re-establishes the subject in the same setup (e.g. "Leo remains on
// the rocky outcrop, his golden mane catching the light…") BEFORE
// introducing the new action. That visual continuity is what lets
// the model hot-swap the prompt mid-stream without the scene
// jumping or resetting visually.
//
// This is the most important prompt-engineering lesson the demo
// teaches. Treat the prompts as a curated dataset, not as throwaway
// strings.

export interface Prompt {
  /** Short headline used as the button label in the UI. */
  title: string;
  /** Full paragraph sent to the model. */
  text: string;
}

export interface Scene {
  id: string;
  label: string;
  initial: Prompt;
  evolutions: ReadonlyArray<Prompt>;
  /** Reference image URL. Present only on image-backed scenes. */
  imageUrl?: string;
}

export const SCENES: ReadonlyArray<Scene> = [
  // ────────────────────────────────────────────────────────────────
  // Text-only scenes (rendered as presets in PromptComposer)
  // ────────────────────────────────────────────────────────────────
  {
    id: "jungle-king",
    label: "King of the Jungle",
    initial: {
      title: "Leo Appears",
      text: "A majestic lion named Leo stands regally in the heart of a dense jungle, embodying the essence of a king. Leo has a golden mane that flows gracefully around his broad shoulders, and his piercing amber eyes survey the landscape with confidence and authority. He is positioned on a rocky outcrop, towering over the lush greenery below. The background showcases a vibrant jungle scene with tall trees, cascading vines, and dappled sunlight filtering through the canopy. Leo's posture is proud and commanding, with his tail held high. The scene is captured from a medium close-up perspective, emphasizing Leo's powerful stance and the regal aura surrounding him.",
    },
    evolutions: [
      {
        title: "The Roar",
        text: "Leo shifts his powerful weight slightly on the rocky outcrop, his golden mane rippling as he does so. Suddenly, he opens his massive jaws to release a deep, resonant roar that vibrates through the humid air, revealing his formidable white teeth. The dappled sunlight filtering through the canopy dances across his fur as his chest expands with the effort. The vibrant jungle remains lush and green around him, with tall trees and cascading vines framing his commanding figure. A medium close-up perspective emphasizing Leo's powerful stance and the regal aura surrounding him.",
      },
      {
        title: "The Butterfly",
        text: "Leo maintains his regal position on the rocky outcrop as the humid jungle air settles around his broad shoulders. He suddenly lowers his massive head to sniff a vibrant blue butterfly that has fluttered near his nose, his piercing amber eyes momentarily softening with curiosity. The dappled sunlight continues to filter through the tall trees and cascading vines, illuminating the golden hues of his mane against the lush green background. His tail remains held high, a symbol of his enduring authority over the landscape. A medium close-up perspective emphasizing Leo's powerful stance and the regal aura surrounding him.",
      },
      {
        title: "The Yawn",
        text: "Leo stretches his massive body across the rocky outcrop in the heart of the dense jungle, his golden mane spreading wide as he extends his powerful front legs forward. He opens his enormous jaws in a wide, lazy yawn, revealing rows of formidable white teeth and a deep pink tongue curling upward. His piercing amber eyes squeeze shut momentarily with the effort as dappled sunlight filters through the canopy, warming his golden fur. The lush greenery of tall trees and cascading vines surrounds his relaxed form. A medium close-up perspective emphasizing Leo's powerful stance and the regal aura surrounding him.",
      },
      {
        title: "The Parrot",
        text: "Leo remains on the rocky outcrop in the heart of the dense jungle, his golden mane catching the filtered light. He suddenly turns his head sharply to the left, his ears twitching as he focuses on a brightly colored parrot that lands on a nearby branch. His piercing amber eyes lock onto the bird with intense focus, analyzing the new arrival amidst the tall trees and cascading vines. The lush greenery provides a vibrant backdrop as he stands with authority, his posture commanding and proud. A medium close-up perspective emphasizing Leo's powerful stance and the regal aura surrounding him.",
      },
    ],
  },
  {
    id: "rainy-evening",
    label: "Rainy Evening",
    initial: {
      title: "In the Rain",
      text: "A young man standing in the rain, looking up at the sky with a warm, inviting smile on his face. He is dressed in a fitted dark jacket and a white t-shirt that clings to his frame as droplets of water fall around him. His hair is gently tousled from the rain, framing his sharp features. The background shows a blurred cityscape with tall buildings and the faint glow of streetlights. The scene captures the serene beauty of a rainy evening. Medium close-up, static shot focusing on the man's face and upper body.",
    },
    evolutions: [
      {
        title: "Catching Drops",
        text: "The young man remains framed against the soft blur of city lights, the rain now glistening on his skin as he slowly extends his right hand palm-up to catch the falling droplets. His expression shifts slightly to one of quiet wonder as the water pools in his cupped fingers. The dark fabric of his jacket darkens further with the moisture, emphasizing the damp atmosphere. The distant streetlights create bokeh orbs behind his silhouette. Medium close-up, static shot focusing on the man's face and upper body.",
      },
      {
        title: "The Sparrow",
        text: "The young man in the soaked dark jacket keeps his right hand extended palm-up in the rain. A small sparrow flutters down through the falling droplets and lands gently on his open palm, its tiny feet gripping his wet fingers. He watches the bird with a soft, surprised smile, barely breathing so as not to startle it. The sparrow ruffles its damp feathers and tilts its head, meeting his gaze. Rain continues to fall steadily around them both, with the blurred gray cityscape and warm streetlights glowing behind. Medium close-up, static shot focusing on the man's face and upper body.",
      },
      {
        title: "The Flock",
        text: "The young man in the soaked dark jacket stands still in the rain as dozens of small birds descend from the gray sky, landing on his shoulders, arms, and head. Sparrows and finches perch along both shoulders of his wet jacket, their tiny claws gripping the damp fabric. Several more flutter around him, wings beating against the falling raindrops. He laughs softly with genuine joy, his eyes bright as the birds nestle close to him for warmth. The blurred cityscape and warm streetlights glow behind the extraordinary scene. Medium close-up, static shot focusing on the man's face and upper body covered in birds.",
      },
    ],
  },
  {
    id: "flower-bloom",
    label: "Flower in Bloom",
    initial: {
      title: "The Bud",
      text: "A tight macro shot of a closed flower bud resting on a slender green stem in soft morning light. The bud is deep purple with tightly wrapped petals forming a perfect teardrop shape, covered in tiny droplets of morning dew that glisten gently. The background is softly blurred, showing hints of green foliage in a garden setting. The air is still and peaceful. The scene captures a moment of quiet anticipation. Extreme close-up macro shot with shallow depth of field, focusing on the flower bud.",
    },
    evolutions: [
      {
        title: "First Opening",
        text: "The purple flower bud on its slender green stem now shows the first signs of opening in soft morning light. The tightly wrapped petals have begun to separate at the tip, revealing small gaps where delicate inner petals peek through in lighter shades of lavender. The morning dew droplets still cling to the curved surfaces as the petals shift position. The background remains softly blurred with hints of green foliage in the garden setting. The still, peaceful air surrounds the transformation beginning to unfold. Extreme close-up macro shot with shallow depth of field, focusing on the flower bud.",
      },
      {
        title: "Petals Unfurl",
        text: "The purple flower continues to unfurl on its slender green stem in soft morning light, now half-open with petals curling outward gracefully. The outer petals have spread wider, revealing the intricate interior with delicate lavender inner petals and the first glimpse of golden-yellow stamens at the center. Dew droplets still glisten on the petals, catching the gentle light. The softly blurred background shows green foliage in the garden setting. The peaceful atmosphere remains as the flower opens further. Extreme close-up macro shot with shallow depth of field, focusing on the flower.",
      },
      {
        title: "Bee Lands",
        text: "The purple flower is now fully opened on its slender green stem in soft morning light, its petals spread wide in a beautiful circular display revealing the golden-yellow stamens clustered at the center. The rich purple outer petals fade to soft lavender near the middle, creating a stunning gradient. Dew droplets still glisten on the opened petals. Suddenly, a fuzzy bumblebee with distinctive yellow and black stripes lands delicately on the edge of one petal, its weight causing the flower to dip slightly. The bee's legs reach toward the stamens, its body dusted with yellow pollen. The softly blurred green foliage background creates a gentle bokeh effect. Extreme close-up macro shot with shallow depth of field, focusing on the flower and bee.",
      },
    ],
  },
  {
    id: "max-birthday",
    label: "Max's Birthday",
    initial: {
      title: "Birthday Boy",
      text: "A young boy named Max sits at a wooden dining table in a sunlit suburban living room, his small hands resting on the edge as he gazes at a frosted birthday cake in front of him. Five lit candles flicker gently on top of the cake, their small flames casting warm glows on his face. He wears a bright blue party hat slightly tilted on his head and a red t-shirt. The background shows a beige couch, family photos on the wall, and soft linen curtains filtering bright afternoon sunlight into the room. The scene captures the festive, happy atmosphere of a child's birthday party. Shot on a 90s VHS Handicam with characteristic grain, slight tracking lines, and warm, slightly oversaturated colors typical of old home movies. Medium close-up shot focusing on Max and the birthday cake.",
    },
    evolutions: [
      {
        title: "Taking a Breath",
        text: "The young boy named Max remains seated at the wooden dining table in the sunlit suburban living room, the frosted birthday cake still in front of him with five lit candles flickering. He suddenly leans forward slightly, his small face illuminated by the candlelight as he takes a deep breath in preparation to blow. His bright blue party hat stays tilted on his head while his red t-shirt catches the warm afternoon light streaming through the linen curtains. The beige couch and family photos remain visible in the background as the VHS grain pulses slightly across the frame. Shot on a 90s VHS Handicam with characteristic grain, slight tracking lines, and warm, slightly oversaturated colors typical of old home movies. Medium close-up shot focusing on Max and the birthday cake.",
      },
      {
        title: "Make a Wish",
        text: "The young boy named Max sits at the wooden dining table in the bright suburban living room, leaning slightly forward toward the frosted birthday cake. He now purses his lips and blows forcefully at the five lit candles, causing the small flames to flicker wildly before extinguishing completely, all five candles going out. Thin wisps of white smoke curl upward from the extinguished wicks, drifting through the warm sunlight streaming through the linen curtains. His bright blue party hat remains slightly tilted as his red t-shirt moves with the effort of his breath. The beige couch and family photos stay visible behind him as the VHS tracking lines judder horizontally across the bottom of the frame. Shot on a 90s VHS Handicam with characteristic grain, slight tracking lines, and warm, slightly oversaturated colors typical of old home movies. Medium close-up shot focusing on Max and the birthday cake.",
      },
      {
        title: "Smile for the Camera",
        text: "The young boy named Max sits at the wooden dining table in the bright suburban living room, the frosted birthday cake with five extinguished, unlit candles still before him. Thin wisps of smoke continue to drift from the extinguished wicks. He suddenly turns his head to look directly into the camera lens, his eyes widening with excitement as a huge, gap-toothed smile spreads across his face. His bright blue party hat sits crooked on his head, nearly falling to one side, while his red t-shirt glows in the sunlight streaming through the linen curtains. The beige couch and family photos blur slightly in the background as the VHS autofocus struggles momentarily, creating a brief soft halo around his smiling face. Shot on a 90s VHS Handicam with characteristic grain, slight tracking lines, and warm, slightly oversaturated colors typical of old home movies. Medium close-up shot focusing on Max and the birthday cake.",
      },
      {
        title: "Wave Hello",
        text: 'The young boy named Max remains at the wooden dining table in the sunlit suburban living room, still looking toward the camera with his gap-toothed smile. He now raises his right hand and waves enthusiastically at the camera, his small fingers spreading wide in an excited greeting. His bright blue party hat has slipped completely to one side of his head, held only by the elastic string under his chin, while his red t-shirt shifts with the motion of his arm. The frosted birthday cake with extinguished, unlit candles sits beside him, smoke wisps now fully dissipated. The warm afternoon light continues to filter through the linen curtains, illuminating the beige couch and family photos in the soft background as the VHS timestamp "05/12/1997" flickers in the bottom right corner. Shot on a 90s VHS Handicam with characteristic grain, slight tracking lines, and warm, slightly oversaturated colors typical of old home movies. Medium close-up shot focusing on Max and the birthday cake.',
      },
    ],
  },

  // ────────────────────────────────────────────────────────────────
  // Image-backed scenes (rendered as example cards in ImageStarter)
  // ────────────────────────────────────────────────────────────────
  {
    id: "village-puppy",
    label: "Village Puppy",
    imageUrl: "/images/puppy.jpg",
    initial: {
      title: "On the Doorstep",
      text: "A fluffy golden retriever puppy wearing a small red bandana sits on a stone doorstep of a charming European village, number 12 on the blue wooden door behind it. The puppy has big round dark eyes and tilts its head slightly with curiosity. Warm afternoon sunlight fills the cobblestone street lined with colorful flower pots of red geraniums, purple hydrangeas, and pink blooms. Stone and pastel-painted buildings stretch into the background. Pixar-style 3D render with vibrant saturated colors. Medium close-up shot focusing on the puppy.",
    },
    evolutions: [
      {
        title: "Chasing a Butterfly",
        text: "The fluffy golden retriever puppy with the small red bandana lifts off the stone doorstep, his attention caught by a vibrant orange butterfly fluttering between the colorful flower pots. He bounds forward on small paws, ears flopping with each excited movement, tail wagging furiously behind him. The cobblestone street lined with red geraniums, purple hydrangeas, and pink blooms catches the warm afternoon sunlight. The blue wooden door with the number 12 remains visible behind him, framed by pastel-painted buildings stretching into the background. Pixar-style 3D render with vibrant saturated colors. Medium close-up shot focusing on the puppy mid-bound.",
      },
      {
        title: "A Child Greets Him",
        text: "The fluffy golden retriever puppy with the red bandana sits up alertly on the stone doorstep, his tail wagging excitedly as a small child wearing a yellow sundress runs out of the blue wooden door behind him. The little girl crouches down beside the puppy, her hands extended toward his face with delight as he licks her fingers in return. The cobblestone street stays bathed in warm afternoon sunlight, with colorful flower pots of red geraniums, purple hydrangeas, and pink blooms surrounding the joyful scene. Pastel-painted buildings stretch into the background. Pixar-style 3D render with vibrant saturated colors. Medium close-up shot focusing on the puppy and the child.",
      },
      {
        title: "Gentle Rain",
        text: "The fluffy golden retriever puppy with the red bandana stays on the stone doorstep as a gentle rain begins to fall on the cobblestone street. Water droplets glisten in his golden fur and bead on the bandana around his neck. His big round dark eyes look upward curiously at the falling drops, head tilting as one lands on his tiny black nose. The flower pots of red geraniums, purple hydrangeas, and pink blooms catch the rain, their colors deepening in the wet light. The pastel-painted village buildings appear softer through the light shower. Pixar-style 3D render with vibrant saturated colors. Medium close-up shot focusing on the puppy in the rain.",
      },
      {
        title: "Belly Up",
        text: "The fluffy golden retriever puppy with the red bandana now lies on his back across the warm stone doorstep, all four paws curled in the air as he wiggles with joy. His big round dark eyes squeeze in a happy squint, mouth open in a playful pant, tail thumping the cobblestones beneath him. The red bandana fans out underneath his neck. Warm afternoon sunlight bathes the scene, with colorful flower pots of red geraniums, purple hydrangeas, and pink blooms surrounding him. The blue wooden door with the number 12 stays visible among the pastel-painted buildings in the background. Pixar-style 3D render with vibrant saturated colors. Medium close-up shot focusing on the playful puppy.",
      },
    ],
  },
  {
    id: "boombox-cat",
    label: "Boombox Cat",
    imageUrl: "/images/boombox-cat.jpg",
    initial: {
      title: "The Performance",
      text: "A white cat wearing black sunglasses is dancing in the middle of an old-fashioned bar, holding up and playing a boombox that's sitting on top of its head. The background shows people laughing at the cat while they eat their food around other tables. The image appears to be shot in the style of Kodak Gold 200 film, with warm grain and slightly desaturated colors.",
    },
    evolutions: [
      {
        title: "Breakdance Spin",
        text: "The white cat wearing black sunglasses drops the boombox onto the wooden bar counter and launches into an enthusiastic breakdance, spinning on its back across the smooth surface. Its white fur flies in every direction as the boombox continues blasting music nearby. The patrons in the old-fashioned bar erupt in cheers and clap along, some standing up from their tables, laughter and excitement filling the warm-lit room. The wood paneling and yellow incandescent lights cast a nostalgic glow over the chaotic scene. Shot in the style of Kodak Gold 200 film with warm grain and slightly desaturated colors.",
      },
      {
        title: "Disco Lights",
        text: "The white cat wearing black sunglasses stands triumphantly back on its hind legs in the middle of the old-fashioned bar, the boombox once again balanced on top of its head and still playing music. Colorful disco lights now begin pulsing from the ceiling — reds, blues, greens, and yellows sweeping across the room. The cat's white fur catches each shifting color as the patrons around the tables burst out laughing harder, some pointing in delight. The polished wood paneling reflects the rotating colored beams. Shot in the style of Kodak Gold 200 film with warm grain and slightly desaturated colors.",
      },
      {
        title: "Patrons Join In",
        text: "The white cat wearing black sunglasses continues to dance in the middle of the old-fashioned bar with the boombox balanced on its head. Now several patrons have abandoned their meals and stood up to dance around the cat — a man in a checkered shirt and a woman in a red dress sway alongside it, while the rest of the diners stay seated but clap and cheer. The boombox blasts music as the warm yellow incandescent lights of the bar bathe the impromptu dance floor in a nostalgic glow. Shot in the style of Kodak Gold 200 film with warm grain and slightly desaturated colors.",
      },
      {
        title: "Zoom on Shades",
        text: "The camera slowly zooms into a tight shot of the white cat's black sunglasses in the dim, warm-lit bar. The cat remains still in the center of the frame, the boombox still balanced on its head and playing music. Reflected in the dark mirrored lenses are the cheering, laughing patrons at their tables and the colorful disco lights pulsing across the room — a tiny chaotic celebration captured in the cat's shades. The white fur around its face is sharp and detailed in the close-up. Shot in the style of Kodak Gold 200 film with warm grain and slightly desaturated colors.",
      },
    ],
  },
];

/** Text-only scenes — used as "Try a prompt" presets in setup. */
export const TEXT_SCENES: ReadonlyArray<Scene> = SCENES.filter(
  (s) => !s.imageUrl,
);

/** Image-backed scenes — used as example cards in setup. */
export const IMAGE_SCENES: ReadonlyArray<Scene & { imageUrl: string }> =
  SCENES.filter((s): s is Scene & { imageUrl: string } => !!s.imageUrl);

/**
 * Look up which scene a given prompt belongs to. Returns the matching
 * scene if `prompt` is either the scene's `initial.text` or one of
 * its `evolutions[].text`; otherwise null (the user has typed a
 * custom prompt we don't have a curated continuation for).
 */
export function findSceneForPrompt(
  prompt: string | null | undefined,
): Scene | null {
  if (!prompt) return null;
  return (
    SCENES.find(
      (s) =>
        s.initial.text === prompt ||
        s.evolutions.some((e) => e.text === prompt),
    ) ?? null
  );
}
