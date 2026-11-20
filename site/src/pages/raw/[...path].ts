// Raw file contents for the Files tab: every text file and small image in a
// model folder, served at /raw/<model-id>/<path>.
import type { APIRoute, GetStaticPaths } from "astro";
import { readFileSync } from "node:fs";
import path from "node:path";
import { MODELS_ROOT, loadCatalog, rawContentType } from "../../lib/catalog";

export const getStaticPaths: GetStaticPaths = () =>
  loadCatalog().models.flatMap((m) =>
    m.files
      .filter((f) => f.viewable)
      .map((f) => ({ params: { path: `${m.id}/${f.path}` } })),
  );

export const GET: APIRoute = ({ params }) => {
  const rel = params.path!;
  const body = readFileSync(path.join(MODELS_ROOT, rel));
  return new Response(body, { headers: { "Content-Type": rawContentType(rel) } });
};
