// Build-time reader: everything the gallery shows comes from this repository
// (the folders under models/, their reactor.yaml, README.md and git history)
// plus categories.yaml. Nothing here runs in the browser.
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import path from "node:path";
import { Lexer } from "marked";
import { parse } from "yaml";
import { buildContext, type ContextReport } from "./context";

export const SITE_DIR = process.cwd();
// The repository root: CATALOG_REPO_DIR, else the parent of site/. It needs its
// git history for the per-file commits.
export const REPO_ROOT = path.resolve(process.env.CATALOG_REPO_DIR ?? path.resolve(SITE_DIR, ".."));
export const MODELS_DIR = "models";
export const MODELS_ROOT = path.join(REPO_ROOT, MODELS_DIR);
export const GITHUB_REPO = "https://github.com/reactor-team/reactor-cookbook";
// Links point at main: a local branch's commits are not on GitHub.
export const GITHUB_REF = process.env.CATALOG_GITHUB_REF ?? "main";

const RAW_TEXT_MAX = 512 * 1024;
const RAW_IMAGE_MAX = 4 * 1024 * 1024;
const IMAGE_TYPES: Record<string, string> = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".webp": "image/webp",
  ".svg": "image/svg+xml",
};

export interface ModelFile {
  path: string; // relative to the model folder
  size: number;
  viewable: "text" | "image" | null; // served under /raw, or GitHub only
  commit: number; // index into Model.commits: the last commit that touched it, -1 if none
}

export interface Commit {
  sha: string;
  date: string; // ISO 8601
  subject: string;
}

export interface Resources {
  gpuType: string | null;
  gpuCount: number;
  cpuRequest: string | null;
  cpuLimit: string | null;
  memRequest: string | null;
  memLimit: string | null;
}

// How the weights reach the node, which decides the shape of the run command.
export type WeightsKind = "hf" | "auto" | "manual" | "readme";
// What the reader has to do: one command, a download by hand and then a
// command, or read the README.
export type StartKind = "one" | "two" | "readme";

export const START_LABEL: Record<StartKind, string> = {
  one: "One command",
  two: "Two steps",
  readme: "See README",
};
export const WEIGHTS_LABEL: Record<WeightsKind, string> = {
  hf: "Hugging Face reference in reactor.yaml",
  auto: "Downloaded by the model on first start",
  manual: "Downloaded by hand, as the README shows",
  readme: "Prepared as the README describes",
};

export interface RunStep {
  title: string;
  code: string;
  verbatim: boolean; // copied unchanged from the README
}

export interface RunPlan {
  start: StartKind;
  weights: WeightsKind;
  weightsPath: string | null; // reactor.yaml runtime.weights_path as written
  evidence: { file: string; line: number; text: string } | null; // where the model downloads its weights
  download: string | null; // the size the README gives
  gpus: string; // the --gpus value
  gpuNote: string | null;
  env: string[];
  notes: string[];
  steps: RunStep[];
  firstBuild: string | null; // the image compiles something heavy
}

// Serving the folder on Reactor under the reader's own account.
export type ServeKind = "ready" | "blocked" | "readme";
export const SERVE_LABEL: Record<ServeKind, string> = {
  ready: "Deployable as is",
  blocked: "Needs changes to deploy",
  readme: "See README",
};

export interface ServeStep {
  title: string;
  text: string | null; // markdown, inline
  code: string | null;
}

export interface ServePlan {
  status: ServeKind;
  name: string; // model.name: the reader's model is <their-org>/<name>
  version: string;
  gpu: string; // type × count from model.resources
  regions: { region: string; count: number }[];
  blockers: string[]; // markdown, why it does not deploy as it is
  context: ContextReport;
  steps: ServeStep[];
  notes: string[]; // markdown
}

export interface Frontend {
  link: string; // a #files/ anchor inside the model page
  readme: boolean; // demo/README.md exists
  apiKeySwitch: boolean; // demo/.env.example names REACTOR_API_KEY
  modelNameVar: boolean; // demo/.env.example names REACTOR_MODEL_NAME
}

export interface Model {
  id: string; // folder name under models/
  path: string; // folder path relative to the repo root
  name: string;
  displayName: string | null;
  version: string | null;
  description: string;
  resources: Resources;
  runtimeVersion: string | null;
  entrypoint: string | null;
  configFile: string | null;
  buildRoute: "Dockerfile" | "build block" | null;
  recording: boolean;
  audio: boolean;
  domainId: string;
  domainTitle: string;
  domainIcon: string;
  embodiment: string | null;
  groupId: string;
  groupTitle: string;
  groupIcon: string;
  typeId: string;
  typeTitle: string;
  typeShort: string;
  typeIcon: string;
  lastCommit: Commit | null;
  commits: Commit[];
  readme: string | null;
  files: ModelFile[];
  run: RunPlan;
  serve: ServePlan;
  frontend: Frontend | null;
}

export interface CategoryType {
  id: string;
  title: string;
  short: string;
  icon: string;
  summary: string;
  models: Model[];
}

export interface CategoryGroup {
  id: string;
  domainId: string;
  title: string;
  icon: string;
  summary: string;
  types: CategoryType[];
}

export interface Domain {
  id: string;
  title: string;
  icon: string;
  summary: string;
  groups: CategoryGroup[];
}

export interface Catalog {
  domains: Domain[];
  groups: CategoryGroup[];
  models: Model[];
}

function git(...args: string[]): string {
  return execFileSync("git", ["-C", REPO_ROOT, ...args], {
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
  });
}

function str(v: unknown): string | null {
  return v === undefined || v === null ? null : String(v);
}

function isBinary(abs: string): boolean {
  const fd = readFileSync(abs);
  const n = Math.min(fd.length, 8000);
  for (let i = 0; i < n; i++) if (fd[i] === 0) return true;
  return false;
}

// One pass over the folder's history: the first commit (newest first) that
// names a path is the last one that touched it.
function history(dir: string): { commits: Commit[]; byPath: Map<string, number> } {
  const out = git("log", "--format=%x00%h%x09%cI%x09%s", "--name-only", "--", dir);
  const commits: Commit[] = [];
  const byPath = new Map<string, number>();
  for (const rec of out.split("\0").filter(Boolean)) {
    const [head, ...names] = rec.split("\n");
    const [sha, date, ...rest] = head.split("\t");
    const idx = commits.push({ sha, date, subject: rest.join("\t") }) - 1;
    for (const n of names) {
      if (!n) continue;
      const rel = n.slice(dir.length + 1);
      if (!byPath.has(rel)) byPath.set(rel, idx);
    }
  }
  return { commits, byPath };
}

function listFiles(dir: string, byPath: Map<string, number>): ModelFile[] {
  const out = git("ls-files", "-z", "--", dir);
  return out
    .split("\0")
    .filter(Boolean)
    .map((p) => {
      const abs = path.join(REPO_ROOT, p);
      const rel = p.slice(dir.length + 1);
      const commit = byPath.get(rel) ?? -1;
      if (!existsSync(abs)) return { path: rel, size: 0, viewable: null, commit };
      const size = statSync(abs).size;
      const ext = path.extname(p).toLowerCase();
      let viewable: ModelFile["viewable"] = null;
      if (IMAGE_TYPES[ext]) viewable = size <= RAW_IMAGE_MAX ? "image" : null;
      else if (size <= RAW_TEXT_MAX && !isBinary(abs)) viewable = "text";
      return { path: rel, size, viewable, commit };
    });
}

// Every model folder: a folder under models/ with a reactor.yaml.
function discoverModels(): string[] {
  return readdirSync(MODELS_ROOT, { withFileTypes: true })
    .filter((e) => e.isDirectory() && !e.name.startsWith(".") && existsSync(path.join(MODELS_ROOT, e.name, "reactor.yaml")))
    .map((e) => e.name)
    .sort();
}

function readManifest(id: string) {
  return parse(readFileSync(path.join(MODELS_ROOT, id, "reactor.yaml"), "utf8")) ?? {};
}

// ---- the run command -------------------------------------------------------

const CLONE = [
  `git clone --depth 1 --filter=blob:none --sparse ${GITHUB_REPO}`,
  "cd reactor-cookbook",
];

const join = (lines: string[]) => lines.join(" && \\\n");

// A weights path as a shell word, with ~ written as $HOME so it expands inside
// double quotes.
function shellPath(p: string): string {
  if (p.startsWith("~/")) return `"$HOME/${p.slice(2)}"`;
  if (/^[\w./-]+$/.test(p)) return p;
  return `"${p}"`;
}

// Every fenced code block in a README, nested ones (in lists, quotes) included.
function codeBlocks(md: string): string[] {
  const out: string[] = [];
  const walk = (tokens: any[]) => {
    for (const t of tokens) {
      if (t.type === "code") out.push(t.text);
      if (t.tokens) walk(t.tokens);
      if (t.items) walk(t.items);
    }
  };
  walk(Lexer.lex(md));
  return out;
}

function fail(id: string, msg: string): never {
  throw new Error(`[gallery] models/${id}: ${msg} (fix site/categories.yaml)`);
}

function findEvidence(id: string, spec: string): { file: string; line: number; text: string } {
  const at = spec.indexOf(":");
  if (at < 0) fail(id, `evidence "${spec}" must be <file>:<text>`);
  const file = spec.slice(0, at);
  const needle = spec.slice(at + 1);
  const abs = path.join(MODELS_ROOT, id, file);
  if (!existsSync(abs)) fail(id, `evidence file ${file} does not exist`);
  const lines = readFileSync(abs, "utf8").split("\n");
  const i = lines.findIndex((l) => l.includes(needle));
  if (i < 0) fail(id, `evidence file ${file} no longer contains ${needle}`);
  return { file, line: i + 1, text: lines[i].trim() };
}

function planRun(id: string, entry: any, manifest: any, gpuCount: number, readme: string | null): RunPlan {
  const kind = entry.weights as WeightsKind;
  if (!["hf", "auto", "manual", "readme"].includes(kind)) fail(id, `weights must be hf, auto, manual or readme, not ${kind}`);
  if (!Number.isInteger(gpuCount) || gpuCount < 1) fail(id, "reactor.yaml requests no GPU (model.resources.gpu.count)");
  const weightsPath = str(manifest.runtime?.weights_path);
  const isHf = Boolean(weightsPath?.startsWith("hf://"));
  if (kind === "hf" && !isHf) fail(id, "weights: hf, but reactor.yaml runtime.weights_path is not an hf:// reference");
  if (kind !== "hf" && isHf) fail(id, `weights: ${kind}, but reactor.yaml runtime.weights_path is an hf:// reference`);
  if (kind === "auto" && !entry.evidence) fail(id, "weights: auto needs an evidence entry");
  if (kind === "manual" && !(entry.readme_download ?? []).length) fail(id, "weights: manual needs readme_download");
  if ((kind === "manual" || kind === "readme") && !readme) fail(id, `weights: ${kind}, but the folder has no README.md`);

  const env: string[] = entry.env ?? [];
  const gpus = gpuCount === 1 ? "device=0" : "all";
  const devices = Array.from({ length: gpuCount }, (_, i) => i).join(",");
  const gpuNote =
    gpuCount === 1
      ? "`--gpus device=0` exposes host GPU 0. On a shared node, replace 0 with a free GPU."
      : `This model needs ${gpuCount} GPUs, and \`--gpus all\` exposes every GPU on the node. On a shared node, pin ${gpuCount} free ones instead, for example \`--gpus '"device=${devices}"'\`.`;
  const runCmd = [
    "reactor run",
    `--gpus ${gpus}`,
    ...env.map((e) => `-e ${e}`),
    ...(entry.run_args ? [String(entry.run_args)] : []),
    ...(kind === "hf" ? ["--yes"] : []),
  ].join(" ");
  const cd = [`git sparse-checkout set ${MODELS_DIR}/${id}`, `cd ${MODELS_DIR}/${id}`];

  const steps: RunStep[] = [];
  let start: StartKind;
  if (kind === "hf" || kind === "auto") {
    start = "one";
    const lines = [...CLONE, ...cd];
    // Docker creates a missing bind-mount source as root; make it first.
    if (kind === "auto" && weightsPath) lines.push(`mkdir -p ${shellPath(weightsPath)}`);
    lines.push(runCmd);
    steps.push({ title: "Download the code and its weights, and start it", code: join(lines), verbatim: false });
  } else if (kind === "manual") {
    start = "two";
    const blocks = codeBlocks(readme!);
    const download = blocks.filter((b) => entry.readme_download.some((s: string) => b.includes(s)));
    if (!download.length) fail(id, `no README code block contains ${entry.readme_download.join(" / ")}`);
    steps.push({ title: "Download the code", code: join([...CLONE, ...cd]), verbatim: false });
    steps.push({ title: "Download the weights, as the README shows", code: download.map((b) => b.trimEnd()).join("\n\n"), verbatim: true });
    if (entry.readme_run) {
      const run = (entry.readme_run as string[]).map((s) => blocks.find((b) => b.includes(s)) ?? fail(id, `no README code block contains ${s}`));
      steps.push({ title: "Build and start it, as the README shows", code: run.map((b) => b.trimEnd()).join("\n\n"), verbatim: true });
    } else {
      steps.push({ title: "Start it", code: runCmd, verbatim: false });
    }
  } else {
    start = "readme";
    steps.push({ title: "Download the code", code: join([...CLONE, ...cd]), verbatim: false });
  }

  // A heavy compile in the image build: the Dockerfile when there is one (it
  // takes the build over), else reactor.yaml's build block.
  const dockerfile = path.join(MODELS_ROOT, id, "Dockerfile");
  const buildText = existsSync(dockerfile) ? readFileSync(dockerfile, "utf8") : JSON.stringify(manifest.build ?? {});
  const compiles = HEAVY_BUILD.exec(buildText)?.[0] ?? null;
  if (compiles && !entry.first_build) fail(id, `the image build compiles (${compiles}) but the entry has no first_build note`);
  if (!compiles && entry.first_build) fail(id, "first_build is set, but the image build compiles nothing heavy");

  return {
    start,
    weights: kind,
    firstBuild: str(entry.first_build),
    weightsPath,
    evidence: kind === "auto" ? findEvidence(id, String(entry.evidence)) : null,
    download: str(entry.download),
    gpus,
    gpuNote,
    env,
    notes: entry.notes ?? [],
    steps,
  };
}

const HEAVY_BUILD = /--no-build-isolation|FLASH_ATTENTION_FORCE_BUILD|TORCH_CUDA_ARCH_LIST|\bnvcc\b|cmake --build/;

// ---- serving on Reactor ----------------------------------------------------

function planServe(id: string, entry: any, manifest: any, run: RunPlan, files: ModelFile[], frontend: Frontend | null): ServePlan {
  const name = str(manifest.model?.name);
  const version = str(manifest.model?.version);
  const gpu = manifest.model?.resources?.gpu;
  if (!name || !version) fail(id, "reactor.yaml has no model.name or model.version");
  if (!gpu?.type || !gpu?.count) fail(id, "reactor.yaml has no model.resources.gpu type and count");
  const dir = path.join(MODELS_ROOT, id);
  const deploymentYaml = path.join(dir, "deployment.yaml");
  const plan = existsSync(deploymentYaml) ? parse(readFileSync(deploymentYaml, "utf8")) : manifest.deployment;
  const regions = ((plan?.instances ?? []) as any[]).map((i) => ({ region: String(i.region), count: Number(i.count ?? 0) }));
  const context = buildContext(dir, files);
  const blockers: string[] = [];
  if (!regions.length)
    blockers.push("reactor.yaml has no `deployment:` block, and the folder has no `deployment.yaml`, so `reactor model deploy` has no instance plan to activate. Add one (step 2) first.");
  for (const o of context.overLimit) blockers.push(`The remote build context is over a \`reactor build --remote\` limit: ${o}.`);
  const wp = run.weightsPath;
  if (wp && !wp.startsWith("hf://") && !wp.startsWith("/") && !wp.startsWith("~") && !context.ignores(`${wp.replace(/^\.\//, "")}/x`))
    blockers.push(`The weights folder \`${wp}\` sits inside the model folder and .dockerignore does not exclude it, so a remote build would upload it as build context.`);
  if (frontend && !context.ignores("demo/node_modules/x"))
    blockers.push("`demo/node_modules` is not excluded by .dockerignore; an installed frontend would be uploaded as build context.");
  if (entry.serve_blocker) blockers.push(String(entry.serve_blocker));

  const status: ServeKind = run.weights === "readme" ? "readme" : blockers.length ? "blocked" : "ready";
  const publishWeights = entry.publish_weights ? ` --weights ${entry.publish_weights}` : "";
  const steps: ServeStep[] = [];
  steps.push({
    title: "Sign in, and see what the platform offers",
    text: "`reactor auth login` opens a browser to approve a key for this machine under your own account (`--with-api-key` reads an existing key instead). `reactor gpu list` lists the GPU types the platform accepts and `reactor region list` its regions; your account needs capacity for this model's GPUs in the region you pick.",
    code: ["reactor auth login", "reactor whoami", "reactor gpu list", "reactor region list"].join("\n"),
  });
  if (!regions.length)
    steps.push({
      title: "Add a deployment plan to reactor.yaml",
      text: "Append this block, with a region from `reactor region list`. `reactor validate` checks the file without contacting the server.",
      code: ["deployment:", "  instances:", "    - region: <your-region>", "      count: 1"].join("\n"),
    });
  if (run.weights === "auto") {
    const local = entry.publish_weights
      ? `mkdir -p ${entry.publish_weights} && \\\nreactor run --gpus ${run.gpus}${run.env.map((e) => ` -e ${e}`).join("")} --weights ${entry.publish_weights}`
      : null;
    steps.push({
      title: "Fill the weights folder: run it once on your own GPU",
      text: local
        ? "reactor.yaml names no weights folder, so give the local run one; publishing uploads the same folder. Wait until `curl -s localhost:8080/health` reports `available`, then stop it with Ctrl-C."
        : `Run the command in **Run it on your own GPU** above, wait until \`curl -s localhost:8080/health\` reports \`available\`, then stop it with Ctrl-C. The model has then downloaded its weights into \`${wp}\`, and publishing uploads that folder.`,
      code: local,
    });
  } else if (run.weights === "manual") {
    steps.push({
      title: "Download the weights",
      text: "The same README commands as step 2 of **Run it on your own GPU** above. Publishing uploads the folder they fill.",
      code: null,
    });
  }
  if (run.weights !== "readme") {
    steps.push({
      title: "Publish the release, check its weights, then deploy it",
      text: `From \`${MODELS_DIR}/${id}\`. Publish builds the image on the Reactor platform (\`--remote\`, no local Docker needed) and ${run.weights === "hf" ? `pins \`${wp}\` to a commit; no weight bytes are uploaded, and the platform pulls them when an instance starts` : `uploads ${entry.publish_weights ? `\`${entry.publish_weights}\`` : `\`${wp}\``} with a SHA-256 manifest, resuming if interrupted`}. \`reactor weights show\` fails with "manifest not yet uploaded" until the release's weights are readable; deploy only after it prints them.`,
      code: [`reactor model publish --remote${publishWeights}`, "reactor weights show", "reactor model deploy"].join("\n"),
    });
    steps.push({
      title: "Check that it serves",
      text: "`status --regions` shows desired and listed instances per region, `instances` each session slot and why it cannot serve, and `logs -f` follows the instances' output.",
      code: [`reactor model status ${name} --regions`, `reactor model instances ${name}`, "reactor logs -f"].join("\n"),
    });
    steps.push({
      title: `Connect a client to \`<your-org>/${name}\``,
      text: frontend?.apiKeySwitch
        ? `The frontend in \`demo/\` switches from a local container to the Reactor API when \`REACTOR_API_KEY\` is set in \`demo/.env\`${frontend.modelNameVar ? `, with \`REACTOR_MODEL_NAME=<your-org>/${name}\`` : ""}; see \`demo/.env.example\`.`
        : `Any Reactor SDK client connects with an API key from your account, naming the model \`<your-org>/${name}\`.${frontend ? " This model's `demo/` talks to a local container only." : ""}`,
      code: null,
    });
  }

  const mib = (n: number) => `${(n / 1024 / 1024).toFixed(1)} MiB`;
  const notes: string[] = [
    `A release that already has an image is never rebuilt: bump \`model.version\` (now \`${version}\`) in reactor.yaml before publishing a code change.`,
    `The remote build context is this folder minus .dockerignore: ${context.entries} entries, ${mib(context.totalBytes)} (limits: 4096 entries, 256 MiB, 64 MiB compressed, 32 MiB per file).`,
  ];
  if (run.weights === "hf") notes.push(`Later releases can go in one command, \`reactor model deploy --remote\`, which publishes and deploys together.`);
  if (run.weights === "auto")
    notes.push("Everything in the weights folder is uploaded, including any source checkout or cache the model keeps there. `--no-weights`, leaving the instance to download its weights when it starts, is untested: we have not verified that a platform instance can write its weights path and reach Hugging Face.");
  if (run.weights === "hf" || run.weights === "auto" || run.weights === "manual")
    if (run.download) notes.push(`The README gives the weights size: ${run.download}.`);
  if (run.env.includes("HF_TOKEN")) notes.push("`HF_TOKEN` is only forwarded to a local run; a deployed instance gets none, so the uploaded folder must already hold everything the model downloads.");
  return { status, name: name!, version: version!, gpu: `${gpu.count}× ${String(gpu.type).replace(/^NVIDIA_/, "")}`, regions, blockers, context, steps, notes };
}

// ---- the catalog -----------------------------------------------------------

function readModel(entry: any, d: any, g: any, t: any): Model {
  const id: string = entry.id;
  const dir = `${MODELS_DIR}/${id}`;
  const m = readManifest(id);
  const res = m.model?.resources ?? {};
  const readmePath = path.join(MODELS_ROOT, id, "README.md");
  const readme = existsSync(readmePath) ? readFileSync(readmePath, "utf8") : null;
  const hist = history(dir);
  const gpuCount = Number(res.gpu?.count ?? 0);
  const demo = path.join(MODELS_ROOT, id, "demo");
  const demoReadme = existsSync(path.join(demo, "README.md"));
  const envExample = path.join(demo, ".env.example");
  const envText = existsSync(envExample) ? readFileSync(envExample, "utf8") : "";
  const frontend: Frontend | null = existsSync(demo)
    ? {
        link: demoReadme ? "#files/demo/README.md" : "#files/demo",
        readme: demoReadme,
        apiKeySwitch: envText.includes("REACTOR_API_KEY"),
        modelNameVar: envText.includes("REACTOR_MODEL_NAME"),
      }
    : null;
  const files = listFiles(dir, hist.byPath);
  const run = planRun(id, entry, m, gpuCount, readme);
  return {
    id,
    path: dir,
    name: str(m.model?.name) ?? id,
    displayName: str(m.model?.["display-name"]),
    version: str(m.model?.version)?.replace(/^v/, "") ?? null,
    description: (str(m.model?.description) ?? "").trim(),
    resources: {
      gpuType: str(res.gpu?.type),
      gpuCount,
      cpuRequest: str(res.cpu?.request),
      cpuLimit: str(res.cpu?.limit),
      memRequest: str(res.memory?.request),
      memLimit: str(res.memory?.limit),
    },
    runtimeVersion: str(m.build?.runtime_version),
    entrypoint: str(m.runtime?.import),
    configFile: str(m.runtime?.config),
    buildRoute: existsSync(path.join(MODELS_ROOT, id, "Dockerfile")) ? "Dockerfile" : m.build ? "build block" : null,
    recording: Boolean(m.runtime?.recording?.enabled),
    audio: Boolean(entry.audio),
    domainId: d.id,
    domainTitle: d.title,
    domainIcon: d.icon ?? "box",
    embodiment: str(entry.embodiment),
    groupId: g.id,
    groupTitle: g.title,
    groupIcon: g.icon ?? "box",
    typeId: t.id,
    typeTitle: t.title,
    typeShort: t.short ?? t.title,
    typeIcon: t.icon ?? "box",
    lastCommit: hist.commits[0] ?? null,
    commits: hist.commits,
    readme,
    files,
    run,
    serve: planServe(id, entry, m, run, files, frontend),
    frontend,
  };
}

// A build reads the catalog once. The dev server re-reads it only when the
// repo changed (HEAD plus uncommitted changes, categories.yaml included),
// checked at most every couple of seconds.
let cached: Catalog | null = null;
let cachedSig = "";
let checkedAt = 0;
const DEV_CHECK_MS = 2000;

function repoSignature(): string {
  return git("rev-parse", "HEAD") + git("status", "--porcelain", "-uall");
}

export function loadCatalog(): Catalog {
  if (cached && !import.meta.env.DEV) return cached;
  if (cached && Date.now() - checkedAt < DEV_CHECK_MS) return cached;
  const sig = import.meta.env.DEV ? repoSignature() : "";
  checkedAt = Date.now();
  if (cached && sig === cachedSig) return cached;
  cachedSig = sig;
  const cfg = parse(readFileSync(path.join(SITE_DIR, "categories.yaml"), "utf8"));

  // Fail closed: every folder placed exactly once, every entry a real folder.
  const present = new Set(discoverModels());
  const placed = new Map<string, number>();
  for (const d of cfg.domains)
    for (const g of d.groups)
      for (const t of g.types)
        for (const e of t.models) placed.set(e.id, (placed.get(e.id) ?? 0) + 1);
  const problems = [
    ...[...present].filter((id) => !placed.has(id)).map((id) => `models/${id} is not placed in categories.yaml`),
    ...[...placed.keys()].filter((id) => !present.has(id)).map((id) => `categories.yaml lists ${id}, but models/${id}/reactor.yaml does not exist`),
    ...[...placed].filter(([, n]) => n > 1).map(([id]) => `categories.yaml lists ${id} more than once`),
  ];
  if (problems.length) throw new Error(`[gallery] ${problems.join("; ")}`);

  const models: Model[] = [];
  const domains: Domain[] = cfg.domains.map((d: any) => ({
    id: d.id,
    title: d.title,
    icon: d.icon ?? "box",
    summary: d.summary,
    groups: d.groups.map((g: any) => ({
      id: g.id,
      domainId: d.id,
      title: g.title,
      icon: g.icon ?? "box",
      summary: g.summary,
      types: g.types.map((t: any) => {
        const typeModels: Model[] = t.models.map((entry: any) => readModel(entry, d, g, t));
        models.push(...typeModels);
        return {
          id: t.id,
          title: t.title,
          short: t.short ?? t.title,
          icon: t.icon ?? "box",
          summary: t.summary,
          models: typeModels,
        };
      }),
    })),
  }));

  cached = { domains, groups: domains.flatMap((d) => d.groups), models };
  return cached;
}

export function rawContentType(file: string): string {
  return IMAGE_TYPES[path.extname(file).toLowerCase()] ?? "text/plain; charset=utf-8";
}

export function gpuLabel(r: Resources): string {
  if (!r.gpuType || r.gpuCount === 0) return "CPU only";
  return `${r.gpuCount}× ${r.gpuType.replace(/^NVIDIA_/, "")}`;
}

export function githubTree(dir: string, sub = ""): string {
  return `${GITHUB_REPO}/tree/${GITHUB_REF}/${dir}${sub ? "/" + sub : ""}`;
}

export function firstSentence(s: string): string {
  const flat = s.replace(/\s+/g, " ").trim();
  const m = flat.match(/^(.+?[.!?])(\s|$)/);
  return m ? m[1] : flat;
}

// A path inside the site, prefixed with its base (CATALOG_BASE when the site is
// served from a sub-path).
export function u(p: string): string {
  return import.meta.env.BASE_URL.replace(/\/$/, "") + p;
}

// What the top-bar search needs on every page.
export function searchIndex() {
  return loadCatalog().models.map((m) => ({ id: m.id, type: m.typeShort, icon: m.typeIcon, gpu: gpuLabel(m.resources) }));
}
