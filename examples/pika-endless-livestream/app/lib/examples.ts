import { mediaSeconds, type Inputs, type Item } from "./inputs";

// Starting shows. The scenes, pictures and voices come from the Pika H3 Flex
// examples on reactor.inc. Reactor made every picture and voice. Each one is
// recast as a show that can run without end. A picture or voice is named in the
// text by its reference label, in the order it is listed here.
export type Example = {
  id: string;
  name: string;
  /** Shown on the card only; the model does not receive it. */
  thumbnail?: string;
  concept: string;
  pictures?: string[];
  audios?: string[];
};

export const EXAMPLES: Example[] = [
  {
    id: "dog-news",
    name: "Dog news",
    concept:
      "A dog news show: a golden retriever anchor covers the neighborhood's biggest stories, with the weather and a reporter on the scene.",
  },
  {
    id: "treehouse",
    name: "Treehouse",
    thumbnail: "/examples/treehouse/frame.jpg",
    concept:
      "A hand-drawn 2D cartoon series in warm watercolour, soft golden afternoon light and a gentle storybook palette. A girl of about eight in a red bandana and overalls has small adventures in and around her wooden treehouse in a big old oak in the backyard: building, exploring, inventing games. Bouncy ukulele and glockenspiel music, birdsong, creaking wood.",
  },
  {
    id: "product-reveal",
    name: "Product reveal",
    thumbnail: "/examples/product-reveal/motorcycle.jpg",
    concept:
      "A nonstop cinematic launch film for the electric motorcycle in <Picture 1> (sand-white body, copper frame, LED light bar), on the salt flats at golden hour. Crane moves, extreme macros along the copper frame, low tracking shots at speed with a rider in a sand-white suit, overhead arcs carving trails in the salt. Keep the motorcycle exactly like <Picture 1> in every shot. No on-screen text and no dialogue; a pulsing electronic score.",
    pictures: ["/examples/product-reveal/motorcycle.jpg"],
  },
  {
    id: "sitcom",
    name: "Sitcom",
    thumbnail: "/examples/sitcom/apartment.jpg",
    concept:
      "A 1990s multi-camera sitcom with bright studio lighting and a live studio audience. Marcy (<Picture 1>, curly red hair, denim jacket over a mustard turtleneck) and Dev (<Picture 2>, tortoiseshell glasses, brown cardigan) share the Apartment (<Picture 4>, an orange sofa), always short on rent. Gus (<Picture 3>, the bald, moustached landlord in a teal bowling shirt) keeps walking in at the worst moment. Short spoken lines with lip-sync, laughs after every joke, a slap-bass sting between scenes.",
    pictures: ["/examples/sitcom/marcy.jpg", "/examples/sitcom/dev.jpg", "/examples/sitcom/gus.jpg", "/examples/sitcom/apartment.jpg"],
  },
  {
    id: "food-truck",
    name: "Food truck",
    thumbnail: "/examples/food-truck/nora.jpg",
    concept:
      "A rainy-night street drama at a taco truck under warm string lights, with a cinematic shallow depth of field. Nora (<Picture 1>, silver bob, bright yellow raincoat, voice of <Audio 1>) is a regular and a food critic. Theo (<Picture 2>, green beanie, canvas apron, voice of <Audio 2>) runs the truck. They trade wry lines over the counter while other customers come and go. Rain on the awning, the grill sizzling, no music.",
    pictures: ["/examples/food-truck/nora.jpg", "/examples/food-truck/theo.jpg"],
    audios: ["/examples/food-truck/nora.mp3", "/examples/food-truck/theo.mp3"],
  },
];

let nextId = 0;

async function load(src: string, kind?: "audio"): Promise<Item> {
  const res = await fetch(src);
  if (!res.ok) throw new Error(`Could not load ${src}.`);
  const blob = await res.blob();
  const file = new File([blob], src.split("/").pop() ?? "example", { type: blob.type });
  const seconds = kind ? await mediaSeconds(file, kind) : undefined;
  return { id: `example-${++nextId}`, file, url: URL.createObjectURL(file), seconds };
}

/** An example's references, as the Inputs panel holds them. */
export async function exampleInputs(example: Example): Promise<Inputs> {
  const [images, audios] = await Promise.all([
    Promise.all((example.pictures ?? []).map((src) => load(src))),
    Promise.all((example.audios ?? []).map((src) => load(src, "audio"))),
  ]);
  return { images, audios, videos: [], first: null, last: null };
}
