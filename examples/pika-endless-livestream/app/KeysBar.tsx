"use client";

import type { Keys } from "./lib/keys";

export type ServerKeys = { reactor: boolean; anthropic: boolean; openai: boolean };

/**
 * The keys, in a collapsed dropdown. A key in the server's environment says so,
 * and a pasted key overrides it. A hosted copy with every key set in its
 * environment shows no dropdown at all.
 */
export function KeysBar({ keys, server, onChange }: { keys: Keys; server: ServerKeys | null; onChange: (k: Keys) => void }) {
  const field = (name: "reactor" | "anthropic" | "openai", label: string, prefix: string) => {
    const fromServer = server?.[name];
    const set = Boolean(keys[name]);
    return (
      <label className={`key${set ? " set" : fromServer ? " server" : ""}`}>
        <span className="mono">
          {label}
          <em>{set ? "pasted" : fromServer ? ".env.local" : name === "reactor" ? "required" : "optional"}</em>
        </span>
        <input
          type="password"
          autoComplete="off"
          spellCheck={false}
          placeholder={fromServer ? "Using the server's key" : `${prefix}…`}
          value={keys[name]}
          onChange={(e) => onChange({ ...keys, [name]: e.target.value.trim() })}
        />
      </label>
    );
  };

  // Until the server answers, show nothing rather than flash fields it may not need.
  if (!server) return null;
  const all = ["reactor", "anthropic", "openai"] as const;
  const missing = all.filter((k) => !server[k]);
  if (!missing.length) return null;
  const labels = { reactor: ["Reactor", "rk_"], anthropic: ["Claude", "sk-ant-"], openai: ["OpenAI", "sk-"] } as const;
  const pasted = all.filter((k) => keys[k]).length;
  const status = missing.includes("reactor") && !keys.reactor ? "Reactor key needed" : pasted ? `${pasted} pasted` : "optional";
  return (
    // Collapsed unless the one required key is missing.
    <details className="keys-drop" open={missing.includes("reactor") && !keys.reactor ? true : undefined}>
      <summary className="mono">
        API keys <em>{status}</em>
      </summary>
      <div className="keys-bar" role="group" aria-label="API keys" style={{ "--cols": all.length } as React.CSSProperties}>
        {all.map((k) => (
          <span key={k} style={{ display: "contents" }}>{field(k, labels[k][0], labels[k][1])}</span>
        ))}
      </div>
    </details>
  );
}
