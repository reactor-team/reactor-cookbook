import { defineConfig } from "astro/config";
import { existsSync, readdirSync, utimesSync } from "node:fs";
import path from "node:path";

// The model folders live outside what the dev server watches, and Astro keeps
// a route's page list until that page file changes. So the dev server lists the
// model folders every couple of seconds (a directory read, cheap) and touches
// the model page when the set changes.
const models = path.join(path.resolve(process.env.CATALOG_REPO_DIR ?? ".."), "models");
const modelFolders = () =>
  readdirSync(models, { withFileTypes: true })
    .filter((e) => e.isDirectory() && existsSync(path.join(models, e.name, "reactor.yaml")))
    .map((e) => e.name)
    .sort()
    .join("\n");
const refreshModelPages = {
  name: "gallery-refresh-model-pages",
  configureServer() {
    const page = path.resolve("src/pages/models/[...id].astro");
    let last = modelFolders();
    const timer = setInterval(() => {
      const now = modelFolders();
      if (now === last) return;
      last = now;
      const t = new Date();
      utimesSync(page, t, t);
    }, 2000);
    timer.unref();
  },
};

export default defineConfig({
  output: "static",
  // "/" by default; set CATALOG_BASE when the site is served from a sub-path.
  base: process.env.CATALOG_BASE ?? "/",
  devToolbar: { enabled: false },
  vite: {
    plugins: [refreshModelPages],
    // Bundle the browser-side libraries at startup. Found lazily, they are
    // re-bundled mid-session under a new hash, and a page already holding the
    // old one gets a 504 for it.
    optimizeDeps: { include: ["highlight.js/lib/common"] },
    // dist/ holds the copied model files after a build; watching it swamps the
    // dev server.
    server: { watch: { ignored: ["**/dist/**"] } },
  },
});
