export type CookingProp = {
  id: string;
  name: string;
  imageUrl: string;
  instruction: string;
};

export const COOKING_PROPS: CookingProp[] = [
  { id: "banana", name: "Banana", imageUrl: "/props/banana.png", instruction: "a ripe yellow banana; have the chef handle it and cook with it in an intuitive, entertaining way" },
  { id: "chef-knife", name: "Chef knife", imageUrl: "/props/chef-knife.png", instruction: "a professional chef knife; introduce it safely as the natural tool for the next preparation step" },
  { id: "meat-cleaver", name: "Meat cleaver", imageUrl: "/props/meat-cleaver.png", instruction: "a heavy meat cleaver; make it part of a safe, purposeful chopping or preparation beat" },
  { id: "tomato", name: "Tomato", imageUrl: "/props/tomato.png", instruction: "a ripe red heirloom tomato; have the chef visibly prepare or cook it" },
  { id: "red-onion", name: "Red onion", imageUrl: "/props/red-onion.png", instruction: "a whole red onion; have the chef peel, slice, season, or cook it naturally" },
  { id: "garlic", name: "Garlic", imageUrl: "/props/garlic.png", instruction: "a whole garlic bulb; have the chef crush, chop, roast, or otherwise use it naturally" },
  { id: "carrot", name: "Carrot", imageUrl: "/props/carrot.png", instruction: "a whole carrot with its leaves; have the chef trim and use it in the dish" },
  { id: "eggplant", name: "Eggplant", imageUrl: "/props/eggplant.png", instruction: "a glossy purple eggplant; have the chef cut and cook it in a recognizable way" },
  { id: "red-bell-pepper", name: "Red pepper", imageUrl: "/props/red-bell-pepper.png", instruction: "a red bell pepper; have the chef seed, slice, roast, or saute it naturally" },
  { id: "lemon", name: "Lemon", imageUrl: "/props/lemon.png", instruction: "a fresh lemon; have the chef zest, squeeze, slice, or season with it" },
  { id: "salmon-fillet", name: "Salmon", imageUrl: "/props/salmon-fillet.png", instruction: "a fresh salmon fillet; have the chef season and cook it safely and convincingly" },
  { id: "ribeye-steak", name: "Ribeye", imageUrl: "/props/ribeye-steak.png", instruction: "a marbled ribeye steak; have the chef season, sear, or rest it as part of the show" },
  { id: "whole-chicken", name: "Chicken", imageUrl: "/props/whole-chicken.png", instruction: "a whole prepared chicken; have the chef season or prepare it safely for cooking" },
  { id: "spaghetti", name: "Spaghetti", imageUrl: "/props/spaghetti.png", instruction: "a bundle of dry spaghetti; have the chef open, cook, or combine it with the developing dish" },
  { id: "butter", name: "Butter", imageUrl: "/props/butter.png", instruction: "a stick of butter; have the chef cut and use it naturally for cooking, basting, or finishing" },
  { id: "olive-oil", name: "Olive oil", imageUrl: "/props/olive-oil.png", instruction: "a glass pourer of olive oil; have the chef drizzle or cook with it naturally" },
];
