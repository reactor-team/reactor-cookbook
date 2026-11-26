"use client";

import type { Keys } from "./lib/keys";

export type ServerKeys = { reactor: boolean; anthropic: boolean; openai: boolean };

const ALL = ["reactor", "anthropic", "openai"] as const;
const LABELS = { reactor: ["Reactor", "rk_"], anthropic: ["Claude", "sk-ant-"], openai: ["OpenAI", "sk-"] } as const;

/**
 * The keys, in a collapsed dropdown. A key set on the server (.env, .env.local,
 * or the host's environment) shows as set, with no input. Only the keys the
 * server lacks get a field to paste one into.
 */
export function KeysBar({ keys, server, onChange }: { keys: Keys; server: ServerKeys | null; onChange: (k: Keys) => void }) {
  // Until the server answers, show nothing rather than flash fields it may not need.
  if (!server) return null;
  const onServer = ALL.filter((k) => server[k]);
  const missing = ALL.filter((k) => !server[k]);
  const names = (list: readonly (typeof ALL)[number][]) => list.map((k) => LABELS[k][0]).join(", ");

  // Every key is set on the server: nothing to paste, just say so.
  if (!missing.length) {
    return (
      <p className="keys-set mono">
        API keys <em>Reactor, Claude and OpenAI set on the server</em>
      </p>
    );
  }

  const needReactor = !server.reactor && !keys.reactor;
  const pasted = missing.filter((k) => keys[k]).length;
  const status = needReactor
    ? "Reactor key needed"
    : [onServer.length ? `${names(onServer)} set on the server` : "", pasted ? `${pasted} pasted` : ""].filter(Boolean).join(" · ") || "optional";

  return (
    // Collapsed unless the one required key is missing.
    <details className="keys-drop" open={needReactor ? true : undefined}>
      <summary className="mono">
        API keys <em>{status}</em>
      </summary>
      <div className="keys-bar" role="group" aria-label="API keys" style={{ "--cols": ALL.length } as React.CSSProperties}>
        {ALL.map((k) =>
          server[k] ? (
            <div key={k} className="key on-server">
              <span className="mono">
                {LABELS[k][0]}
                <em>set</em>
              </span>
              <span className="key-note">Set on the server</span>
            </div>
          ) : (
            <label key={k} className={`key${keys[k] ? " set" : ""}`}>
              <span className="mono">
                {LABELS[k][0]}
                <em>{keys[k] ? "pasted" : k === "reactor" ? "required" : "optional"}</em>
              </span>
              <input
                type="password"
                autoComplete="off"
                spellCheck={false}
                placeholder={`${LABELS[k][1]}…`}
                value={keys[k]}
                onChange={(e) => onChange({ ...keys, [k]: e.target.value.trim() })}
              />
            </label>
          ),
        )}
      </div>
    </details>
  );
}
