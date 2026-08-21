export type FaceButton = "a" | "b" | "x" | "y";

export type QuickAction = {
  button: FaceButton;
  label: string;
  prompt: string;
  movementOverride?: string;
  pose?: {
    profile?: "jump";
    rx?: number;
    ry?: number;
    rz?: number;
    ty?: number;
  };
};

export type MovementDirection =
  | "forward"
  | "back"
  | "strafe_left"
  | "strafe_right";

export type WorldContract = {
  identity: string;
  camera: string;
  environment: string;
  invariants: string;
};

export type ArcadeGame = {
  id:
    | "good-dog-sf"
    | "windward"
    | "dustline"
    | "deep-signal"
    | "arch-runner"
    | "blue-mesa"
    | "free-range";
  title: string;
  shortTitle: string;
  genre: string;
  deck: string;
  image: string;
  seed: number;
  worldContract: WorldContract;
  staticPrompt: string;
  movingPrompt: string;
  movementPrompts?: Partial<Record<MovementDirection, string>>;
  objective: string;
  actions: Record<FaceButton, QuickAction>;
};

export const GAMES: ArcadeGame[] = [
  {
    id: "good-dog-sf",
    title: "GOOD DOG, SF",
    shortTitle: "GOOD DOG",
    genre: "OPEN-WORLD SNIFFING",
    deck: "A whole city at nose height. Follow whatever smells interesting.",
    image: "/games/good-dog-sf-seed.png",
    seed: 1847,
    objective: "FOLLOW THE COFFEE SCENT",
    worldContract: {
      identity:
        "One friendly medium-sized golden-brown dog with realistic short fur, four natural legs, two alert ears, one tail, and a simple dark collar. Preserve the same coat color, proportions, collar, face, and complete silhouette.",
      camera:
        "Stable third-person follow camera about one meter behind and slightly above the dog, looking forward down the sidewalk. Preserve camera height, distance, lens, horizon, and full-body readability.",
      environment:
        "The same steep San Francisco residential street with painted Victorian facades, sidewalk trees, bay haze, damp pavement, the fire hydrant, curb, and downhill route. Grounded neural-arcade realism with subtle analog grain.",
      invariants:
        "Never add another dog, change breed or collar, remove limbs, switch streets, reverse the downhill route, jump to first person, or relocate permanent landmarks. Maintain time of day, weather, palette, and spatial continuity.",
    },
    staticPrompt:
      "The dog stands still on the sidewalk with all four paws planted, ears alert, breathing gently and shifting its weight naturally while the street remains calm.",
    movingPrompt:
      "The dog trots directly away from the camera along the sidewalk with a gentle rhythmic gait, paws contacting the pavement in a natural cadence while its tail sways behind it.",
    actions: {
      a: {
        button: "a",
        label: "Jump",
        prompt:
          "The dog springs upward from all four paws, lifting clearly off the sidewalk before dropping back down and landing firmly on all four paws.",
        pose: { profile: "jump" },
      },
      b: {
        button: "b",
        label: "Roll over",
        prompt:
          "The dog tucks its paws, rolls completely across its back from one side to the other, then rises naturally with all four paws planted on the sidewalk.",
        movementOverride:
          "The dog lowers onto the same patch of sidewalk and performs a compact playful roll-over cycle, keeping its complete body visible and grounded in place.",
      },
      x: {
        button: "x",
        label: "Dig",
        prompt:
          "The dog paws quickly at a small patch of soft earth beside the sidewalk, scattering a little dirt while the surrounding curb and plants stay intact.",
        pose: { rx: -0.02, ty: 0.025 },
      },
      y: {
        button: "y",
        label: "Zoomies",
        prompt:
          "The dog twists its head toward its own tail and keeps chasing it with focused, playful energy, its tail always visibly leading the chase.",
        movementOverride:
          "The dog runs continuously in one tight playful circle on the same patch of sidewalk, rapidly chasing its own tail with quick grounded pawsteps and a compact circular path.",
      },
    },
  },
  {
    id: "windward",
    title: "WINDWARD",
    shortTitle: "WINDWARD",
    genre: "OPEN-OCEAN SAILING",
    deck: "Read the wind. Thread the island passage before the weather closes.",
    image: "/game-concepts/windward.png",
    seed: 2911,
    objective: "CLEAR THE WINDWARD PASSAGE",
    worldContract: {
      identity:
        "One capable sailor planted at the wheel of the same large working wooden sailing ship, with both hands naturally operating the wheel. Preserve the sailor, salt-worn timber, brass compass binnacle, hull proportions, coherent masts, rigging, and filled canvas sail plan.",
      camera:
        "Stable third-person helm camera directly behind and slightly above the sailor, looking down the deck toward the bow and island passage. Preserve camera height, distance, lens, deck alignment, and horizon.",
      environment:
        "The same open ocean of rolling cold-blue swells, sea spray, layered clouds, warm late-afternoon light, and navigable island passage ahead. Grounded neural-arcade realism with practical maritime materials.",
      invariants:
        "The player controls the whole ship, never a sailor walking around. Never change ship type, sail count, rigging, wheel, island layout, weather, time of day, or camera side. Preserve believable water, wind, wake, and spatial continuity.",
    },
    staticPrompt:
      "The sailor holds the wheel steady while the ship rises and settles naturally on the swell. The camera remains directly behind and slightly above the sailor, preserving a clear view down the deck toward the island passage.",
    movingPrompt:
      "The player controls the sailing ship, not the sailor walking around. The sailor remains planted at the wheel and visibly steers while the whole ship advances under wind power with believable heel, sail pressure, wake, wave response, and gradual momentum. The third-person helm camera remains stable while the deck, rigging, sea, and island passage move coherently.",
    movementPrompts: {
      forward:
        "The player is holding forward. The entire sailing ship gains headway toward the island passage as the sails draw and the wake strengthens. The sailor stays at the wheel; do not make the sailor walk or run down the deck.",
      back:
        "The player is holding back. The sailor eases the ship's speed by feathering the sails and applying counter-rudder; the ship loses headway gradually rather than stopping instantly.",
      strafe_left:
        "The player is steering left. The sailor turns the wheel to port and the whole ship follows a broad believable arc to port with natural heel. Do not slide the ship sideways.",
      strafe_right:
        "The player is steering right. The sailor turns the wheel to starboard and the whole ship follows a broad believable arc to starboard with natural heel. Do not slide the ship sideways.",
    },
    actions: {
      a: {
        button: "a",
        label: "Trim sails",
        prompt:
          "The sailor hauls the nearby sheet and the crew completes one coordinated sail trim; the canvas tightens, fills cleanly, and the ship gains visible speed.",
      },
      b: {
        button: "b",
        label: "Hard turn",
        prompt:
          "The sailor spins the wheel into one decisive hard turn; the rudder bites, the ship heels visibly, and the bow carves a controlled arc through the swell.",
        pose: { ry: 0.045, rz: 0.025 },
      },
      x: {
        button: "x",
        label: "Drop anchor",
        prompt:
          "The anchor drops from the bow with a visible run of chain and a strong splash, slowing the ship progressively as the line draws taut.",
      },
      y: {
        button: "y",
        label: "Spyglass",
        prompt:
          "The sailor briefly raises a brass spyglass toward the island passage, revealing the safest channel between the rocks before returning a hand to the wheel.",
      },
    },
  },
  {
    id: "dustline",
    title: "DUSTLINE",
    shortTitle: "DUSTLINE",
    genre: "DESERT OFF-ROAD",
    deck: "Pick a line through the basin. The suspension can take it if you can.",
    image: "/game-concepts/dustline.png",
    seed: 4049,
    objective: "CATCH THE RIDGE MARKER",
    worldContract: {
      identity:
        "One detailed weathered desert trophy truck with the same body panels, paint, wide stance, realistic suspension, and twin spare tires. Preserve vehicle proportions, wheel count, damage, color, and rival vehicle identity.",
      camera:
        "Stable third-person chase camera centered behind and slightly above the trophy truck. Preserve camera height, follow distance, lens, truck scale, horizon, and forward route visibility.",
      environment:
        "The same vast high-desert basin with granular sand, red-rock formations, branching natural routes, distant radio tower, late-afternoon amber light, and long blue shadows. Grounded neural-arcade realism.",
      invariants:
        "Never change vehicle class, tires, body color, camera side, radio-tower position, terrain biome, time of day, or rival vehicle. Do not create roads, crowds, buildings, or impossible rock motion. Maintain route and landmark continuity.",
    },
    staticPrompt:
      "The trophy truck idles at the top of a rocky descent with the open basin and rival vehicle clearly visible ahead from a stable chase camera.",
    movingPrompt:
      "The truck accelerates over uneven terrain with believable tire grip, suspension compression, and a coherent dust plume. The chase camera remains stable while the selected route flows naturally beneath the vehicle.",
    actions: {
      a: {
        button: "a",
        label: "Nitro",
        prompt:
          "A brief nitrous burst pushes the trophy truck forward with stronger acceleration, a denser dust wake, and controlled suspension load.",
      },
      b: {
        button: "b",
        label: "Handbrake",
        prompt:
          "The truck performs a controlled dirt handbrake turn, rotating into the chosen path and regaining traction without rolling.",
        pose: { ry: 0.055, rz: 0.035 },
      },
      x: {
        button: "x",
        label: "Quick fix",
        prompt:
          "The truck briefly slows as an onboard repair system secures a loose body panel and restores stable suspension response.",
      },
      y: {
        button: "y",
        label: "Route ping",
        prompt:
          "A navigation pulse identifies two viable paths through the rocks, one fast and exposed and one narrow but sheltered.",
      },
    },
  },
  {
    id: "deep-signal",
    title: "DEEP SIGNAL",
    shortTitle: "DEEP SIGNAL",
    genre: "ABYSSAL EXPLORATION",
    deck: "Descend past the light. Find out why the relay woke up.",
    image: "/game-concepts/deep-signal.png",
    seed: 6029,
    objective: "REACH THE RELAY CORE",
    worldContract: {
      identity:
        "One compact yellow research submersible with the same practical pressure hull, twin articulated thrusters, paired manipulator arms, windows, lights, markings, and proportions. Preserve the station modules and distant manta silhouette.",
      camera:
        "Stable third-person follow camera centered behind and slightly above the submersible, showing the complete craft, searchlight cones, trench route, and relay station. Preserve distance, lens, orientation, and scale.",
      environment:
        "The same vast dark-blue ocean trench with black volcanic rock, suspended particles, bioluminescent organisms, and drowned relay station built into the trench. Grounded neural-arcade realism with coherent underwater depth.",
      invariants:
        "Never change submersible color, hull, thruster or arm count, station architecture, trench topology, manta identity, water color, or camera side. Do not surface, cut to an interior, or introduce unrelated creatures. Maintain spatial continuity.",
    },
    staticPrompt:
      "The submersible hovers above the trench mouth with its searchlights illuminating the nearest relay module from a steady third-person follow camera.",
    movingPrompt:
      "The submersible moves with slow underwater inertia and precise articulated thrusters. The camera follows steadily while suspended particles, rock walls, and relay modules maintain coherent depth.",
    actions: {
      a: {
        button: "a",
        label: "Sonar",
        prompt:
          "The submersible emits one sonar pulse that briefly reveals the trench walls, open tunnels, and relay core location through the water.",
      },
      b: {
        button: "b",
        label: "Ballast",
        prompt:
          "The ballast system vents and the submersible makes a controlled rapid descent while maintaining a level orientation.",
        pose: { rx: -0.025, ty: 0.045 },
      },
      x: {
        button: "x",
        label: "Manipulator",
        prompt:
          "The right manipulator arm extends toward a loose relay component and grips it carefully without disturbing the surrounding station.",
      },
      y: {
        button: "y",
        label: "Floodlights",
        prompt:
          "The high-output floodlights engage, revealing more of the station entrance and nearby bioluminescent life in crisp detail.",
      },
    },
  },
  {
    id: "arch-runner",
    title: "ARCH RUNNER",
    shortTitle: "ARCH RUNNER",
    genre: "CANYON FLIGHT",
    deck: "Thread the arches. Keep the wings level when the canyon narrows.",
    image: "/game-concepts/arch-runner.png",
    seed: 7817,
    objective: "CLEAR THE ARCH COURSE",
    worldContract: {
      identity:
        "One single-engine vintage biplane with coherent twin cream-canvas wings, the same weathered Dune-gold fuselage, struts, wheels, tail, control surfaces, visible pilot, and spinning propeller. Preserve aircraft anatomy, paint, proportions, and pilot.",
      camera:
        "Stable third-person chase camera directly behind and slightly above the biplane, keeping the full aircraft readable with the next arch centered ahead. Preserve distance, lens, aircraft scale, horizon, and forward flight direction.",
      environment:
        "The same southern Utah red-rock landscape with layered mesas, fins, hoodoos, deep ravines, sparse desert scrub, warm late-afternoon sun, cool blue shadows, and many large natural sandstone arches forming navigable aerial routes. Grounded neural-arcade realism.",
      invariants:
        "Never change aircraft type, wing count, fuselage color, pilot, camera side, arch geology, sun direction, time of day, or route orientation. Never add other aircraft, weapons, buildings, roads, floating rocks, or fantasy structures. Maintain spatial continuity.",
    },
    staticPrompt:
      "The biplane flies level toward the first large sandstone arch from a steady third-person chase camera, with several later arches clearly visible beyond it.",
    movingPrompt:
      "The player controls the biplane in sustained powered flight. The aircraft advances continuously with believable propeller thrust, lift, banking, pitch, control-surface response, and gradual momentum. The chase camera remains stable while the arches and canyon route flow coherently past the aircraft.",
    movementPrompts: {
      forward:
        "The player is holding forward. The biplane increases throttle and flies forward toward the next sandstone arch; the propeller blur and airspeed increase while the aircraft maintains stable lift.",
      back:
        "The player is holding back. The pilot gently pitches the biplane upward and reduces airspeed without flying backward or stalling; the aircraft continues moving forward through the canyon.",
      strafe_left:
        "The player is steering left. The biplane banks into a coordinated left turn around the rock formation, using realistic roll and yaw. Do not slide the aircraft sideways.",
      strafe_right:
        "The player is steering right. The biplane banks into a coordinated right turn around the rock formation, using realistic roll and yaw. Do not slide the aircraft sideways.",
    },
    actions: {
      a: {
        button: "a",
        label: "Throttle boost",
        prompt:
          "The pilot advances the throttle for one strong burst; the propeller blur tightens, engine power rises, and the biplane accelerates cleanly toward the next arch.",
      },
      b: {
        button: "b",
        label: "Barrel roll",
        prompt:
          "The biplane performs one smooth complete barrel roll in clear air, returning wings-level on the same flight path without striking the canyon.",
        pose: { rz: 0.045 },
      },
      x: {
        button: "x",
        label: "Air brake",
        prompt:
          "The pilot briefly deploys aerodynamic braking and raises the nose slightly, shedding speed while maintaining stable lift before the next narrow arch.",
      },
      y: {
        button: "y",
        label: "Route flare",
        prompt:
          "A small bright navigation flare launches ahead and arcs through the safest visible sandstone arch, clearly marking the next route without changing the terrain.",
      },
    },
  },
  {
    id: "blue-mesa",
    title: "BLUE MESA",
    shortTitle: "BLUE MESA",
    genre: "OPEN-WATER KAYAKING",
    deck: "Pick an island. Follow the morning light across open water.",
    image: "/game-concepts/blue-mesa.png",
    seed: 9143,
    objective: "REACH THE ARCH ISLAND",
    worldContract: {
      identity:
        "The same red touring kayak, pointed bow, deck fittings, and black double-bladed paddle. In the idle reference, exactly one black-gloved left hand is visible gripping the shaft; the right hand stays outside the frame. Preserve their appearance, proportions, and anatomy.",
      camera:
        "Stable first-person seated camera just above the waterline, with the red bow small and centered low. At idle, preserve the reference exactly: the raised left paddle blade and shaft remain fixed diagonally in the lower-left foreground. Preserve lens, horizon, scale, and orientation.",
      environment:
        "The same immense turquoise desert lake at sunrise, with separated reed islets, sandstone stacks, a natural arch island, and tree-covered islands. Preserve the broad open-water channels, distant mesas, haze, reflections, and spacious horizon.",
      invariants:
        "Never create rapids, a narrow river, or an enclosed canyon. Do not change the kayak, camera, paddle, gloves, islands, arch, sunrise, water color, or mesas. Add no people, boats, buildings, roads, or docks. Keep temporary water and weather effects physically believable.",
    },
    staticPrompt:
      "A locked idle pose. The visible left glove keeps the same closed grip, wrist angle, and pixel position as the reference. The paddle shaft and raised blade remain rigidly fixed at the same diagonal, fully out of the water. The right hand stays outside the frame. No hand, finger, wrist, arm, or paddle motion; no gesturing or stroking. Only the lake reflections and tiny bow ripples move while the kayak drifts.",
    movingPrompt:
      "The player paddles the kayak across the calm lake with smooth alternating strokes, visible paddle entry, small blade eddies, and believable forward glide. The first-person seated camera remains stable while separated islands shift coherently across the wide horizon.",
    movementPrompts: {
      forward:
        "The player is holding forward. Both gloved hands perform calm alternating forward paddle strokes and the kayak glides toward the chosen island, leaving a narrow wake. Do not make the player walk, swim, motor, or enter rapids.",
      back:
        "The player is holding back. The paddler uses controlled reverse strokes to slow the kayak and gently move backward while keeping the bow and horizon stable.",
      strafe_left:
        "The player is steering left. The paddler makes a broad sweep stroke on the right and the kayak turns through a smooth forward arc to port. Do not slide the kayak sideways.",
      strafe_right:
        "The player is steering right. The paddler makes a broad sweep stroke on the left and the kayak turns through a smooth forward arc to starboard. Do not slide the kayak sideways.",
    },
    actions: {
      a: {
        button: "a",
        label: "Paddle burst",
        prompt:
          "The paddler performs two strong alternating forward strokes, each blade clearly entering and leaving the water, and the red kayak gains a short burst of speed with a stronger narrow wake.",
        pose: { rx: -0.012, ty: -0.008 },
      },
      b: {
        button: "b",
        label: "Storm",
        prompt:
          "A violent thunderstorm sweeps across the entire lake: towering charcoal clouds swallow the sunrise, forked lightning flashes over the islands, heavy rain lashes the water, and strong wind drives whitecaps across the darkened surface.",
      },
      x: {
        button: "x",
        label: "Choppy waters",
        prompt:
          "Short, irregular waves spread across the lake and make the kayak bob naturally. Change only the water; keep the paddle and hands still unless movement is commanded.",
      },
      y: {
        button: "y",
        label: "Dolphin fins",
        prompt:
          "A peaceful pod of dolphins moves through the lake. Several natural-scale dorsal fins break the surface around and ahead of the kayak, glide alongside it, dip below, and resurface nearby. Preserve the weather and kayak course.",
      },
    },
  },
  {
    id: "free-range",
    title: "FREE RANGE",
    shortTitle: "FREE RANGE",
    genre: "FIELD RUNNER",
    deck: "Cut through the meadow. The red gate is open if you can reach it.",
    image: "/game-concepts/free-range.png",
    seed: 11273,
    objective: "REACH THE RED GATE",
    worldContract: {
      identity:
        "One russet-brown adult hen with the same feather pattern, red comb, compact body, two wings, one tail, and exactly two yellow legs. Preserve the chicken's anatomy, scale, plumage, and complete silhouette.",
      camera:
        "Stable tight third-person chase camera directly behind and slightly above the chicken at chicken height. Keep the full chicken large in the lower center with the route and red gate visible ahead. Preserve distance, lens, horizon, and forward orientation.",
      environment:
        "The same broad late-afternoon meadow with tall green grass, buttercups, seed heads, scattered hay bales, distant hedgerows, open run lanes, and one red farm gate ahead. Preserve the warm sunlight and spacious field.",
      invariants:
        "Never add another chicken or animal, change plumage, add limbs, crop the chicken, switch to first person, move the red gate, or replace the meadow. Eggs may appear only when explicitly laid. Maintain anatomy, lighting, landmarks, and spatial continuity.",
    },
    staticPrompt:
      "The chicken stands alert on both feet in the meadow lane, facing the red gate. Its body remains grounded and readable while feathers and nearby grass move subtly in the breeze.",
    movingPrompt:
      "The chicken runs directly away from the camera through the meadow with quick alternating steps, natural head-bobbing, tucked wings, and believable foot contact. The tight chase camera follows smoothly while the red gate grows closer.",
    movementPrompts: {
      forward:
        "The player is holding forward. The chicken runs toward the red gate through the open grass lane with fast natural steps and stable two-legged anatomy.",
      back:
        "The player is holding back. The chicken slows and takes short cautious backward steps while remaining oriented toward the red gate.",
      strafe_left:
        "The player is steering left. The chicken veers through a smooth running arc to the left without sliding sideways.",
      strafe_right:
        "The player is steering right. The chicken veers through a smooth running arc to the right without sliding sideways.",
    },
    actions: {
      a: {
        button: "a",
        label: "Jump",
        prompt:
          "The chicken makes one clear short jump: both feet leave the ground, its wings open slightly for balance, then both feet land firmly before it resumes the prior movement.",
        pose: { profile: "jump" },
      },
      b: {
        button: "b",
        label: "Lay egg",
        prompt:
          "The hen briefly crouches and lays exactly one small pale egg onto the grass directly behind her, then rises normally. The egg remains on the ground; preserve the hen's body and feathers.",
        movementOverride:
          "The hen stops in the same meadow lane, holds a low stable crouch, lays one egg, and returns to a natural two-footed stance.",
      },
      x: {
        button: "x",
        label: "Peck",
        prompt:
          "The chicken dips its head for one quick precise peck at a seed in the grass, then raises its head with the same beak and neck anatomy.",
      },
      y: {
        button: "y",
        label: "Wing sprint",
        prompt:
          "The chicken sprints with rapid grounded steps while spreading both wings slightly for balance, rustling the grass and gaining a playful burst of speed.",
        movementOverride:
          "The chicken runs continuously and quickly toward the red gate with both wings held slightly open, two feet cycling naturally, and the chase camera close behind.",
      },
    },
  },
];

export function nextGameIndex(current: number, direction: 1 | -1) {
  return (current + direction + GAMES.length) % GAMES.length;
}

export function serializeWorldContract(game: ArcadeGame) {
  return [
    `IMMUTABLE SUBJECT CONTRACT: ${game.worldContract.identity}`,
    `IMMUTABLE CAMERA CONTRACT: ${game.worldContract.camera}`,
    `IMMUTABLE ENVIRONMENT CONTRACT: ${game.worldContract.environment}`,
    `NON-NEGOTIABLE CONTINUITY RULES: ${game.worldContract.invariants}`,
  ].join(" ");
}

export function composeGamePrompt(
  game: ArcadeGame,
  moving: boolean,
  heldButtons: FaceButton[],
  movementDirections: MovementDirection[] = [],
) {
  const movementOverride = heldButtons
    .map((button) => game.actions[button].movementOverride)
    .find(Boolean);
  return [
    serializeWorldContract(game),
    movementOverride ?? (moving ? game.movingPrompt : game.staticPrompt),
    ...(movementOverride
      ? []
      : movementDirections.flatMap((direction) => {
          const prompt = game.movementPrompts?.[direction];
          return prompt ? [prompt] : [];
        })),
    ...heldButtons.map((button) => game.actions[button].prompt),
  ].join(" ");
}
