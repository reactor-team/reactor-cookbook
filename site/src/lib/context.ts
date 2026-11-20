// The build context `reactor build --remote` uploads for a model folder: its
// files, filtered by the folder's .dockerignore with Docker's rules (patterns
// are anchored at the folder root, `**` crosses directories, a later `!`
// pattern re-includes, and excluding a directory excludes everything in it).
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";

// The limits `reactor build --help` states for a remote build context.
export const REMOTE_LIMITS = {
  entries: 4096,
  fileBytes: 32 * 1024 * 1024,
  totalBytes: 256 * 1024 * 1024,
  compressedBytes: 64 * 1024 * 1024,
};

interface Rule {
  re: RegExp;
  negate: boolean;
}

function globToRegex(glob: string): RegExp {
  let out = "";
  for (let i = 0; i < glob.length; i++) {
    const c = glob[i];
    if (c === "*" && glob[i + 1] === "*") {
      // "**/" matches zero or more directories; a trailing "**" matches anything.
      if (glob[i + 2] === "/") {
        out += "(?:.*/)?";
        i += 2;
      } else {
        out += ".*";
        i += 1;
      }
    } else if (c === "*") out += "[^/]*";
    else if (c === "?") out += "[^/]";
    else if (c === "[") {
      const end = glob.indexOf("]", i);
      if (end < 0) out += "\\[";
      else {
        out += glob.slice(i, end + 1).replace(/^\[!/, "[^");
        i = end;
      }
    } else out += c.replace(/[.+^${}()|\\]/g, "\\$&");
  }
  return new RegExp(`^${out}$`);
}

function readRules(dir: string): Rule[] {
  const file = path.join(dir, ".dockerignore");
  if (!existsSync(file)) return [];
  return readFileSync(file, "utf8")
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith("#"))
    .map((l) => {
      const negate = l.startsWith("!");
      const p = path.posix.normalize((negate ? l.slice(1) : l).trim()).replace(/^\/+/, "").replace(/\/+$/, "");
      return { re: globToRegex(p), negate };
    });
}

// A path is excluded when the last rule matching it, or any of its parent
// directories, excludes it.
function excluded(rel: string, rules: Rule[]): boolean {
  const parts = rel.split("/");
  const prefixes = parts.map((_, i) => parts.slice(0, i + 1).join("/"));
  let out = false;
  for (const r of rules) if (prefixes.some((p) => r.re.test(p))) out = !r.negate;
  return out;
}

export interface ContextReport {
  entries: number; // files plus the directories that hold them
  totalBytes: number;
  largest: { path: string; size: number } | null;
  overLimit: string[]; // one line per limit exceeded, empty when it fits
  ignores: (rel: string) => boolean;
}

export function buildContext(dir: string, files: { path: string; size: number }[]): ContextReport {
  const rules = readRules(dir);
  const kept = files.filter((f) => !excluded(f.path, rules));
  const dirs = new Set<string>();
  for (const f of kept) {
    const parts = f.path.split("/");
    for (let i = 1; i < parts.length; i++) dirs.add(parts.slice(0, i).join("/"));
  }
  const totalBytes = kept.reduce((n, f) => n + f.size, 0);
  const largest = kept.reduce<{ path: string; size: number } | null>((m, f) => (!m || f.size > m.size ? { path: f.path, size: f.size } : m), null);
  const entries = kept.length + dirs.size;
  const mib = (n: number) => `${(n / 1024 / 1024).toFixed(1)} MiB`;
  const overLimit: string[] = [];
  if (entries > REMOTE_LIMITS.entries) overLimit.push(`${entries} entries (limit ${REMOTE_LIMITS.entries})`);
  if (totalBytes > REMOTE_LIMITS.totalBytes) overLimit.push(`${mib(totalBytes)} in total (limit 256 MiB)`);
  // Compression cannot grow the context, so only one over 64 MiB uncompressed
  // can miss the compressed limit; how much it shrinks depends on the files.
  if (totalBytes > REMOTE_LIMITS.compressedBytes && totalBytes <= REMOTE_LIMITS.totalBytes)
    overLimit.push(`${mib(totalBytes)} uncompressed, which may not compress under the 64 MiB limit`);
  for (const f of kept) if (f.size > REMOTE_LIMITS.fileBytes) overLimit.push(`${f.path} is ${mib(f.size)} (limit 32 MiB per file)`);
  return { entries, totalBytes, largest, overLimit, ignores: (rel) => excluded(rel, rules) };
}
