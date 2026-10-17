export type WatchSource = 'scene' | 'webcam' | 'feed';
export type RuleAction = 'mark' | 'chime' | 'stop' | 'direct';
export type LiveRule = {
  id: string;
  source: WatchSource;
  condition: string;
  action: RuleAction;
  direction: string;
  enabled: boolean;
};
export type Observation = {
  source: WatchSource;
  key: string;
  capturedAt: number;
  seconds: number;
  shotId: string;
  image?: string;
  text?: string;
};
export type RuleEvent = Observation & {
  id: string;
  ruleId: string;
  condition: string;
  action: RuleAction;
  direction: string;
  probability: number;
};
export type RuleState = { hits: number; fired: boolean; lastKey?: string; probability?: number };
export type DecisionQuestion = { id: string; condition: string };
export type DecisionAnswer = { id: string; probability: number | null };
export const RULE_LIMIT = 8;
export const TRIGGER_THRESHOLD = 0.85;

/** Two distinct observations must agree. Ambiguity breaks the streak; a fired rule needs explicit rearming. */
export function observeRule(state: RuleState, key: string, probability: number | null): { state: RuleState; fire: boolean } {
  if (state.fired || state.lastKey === key) return { state, fire: false };
  const hits = probability !== null && Number.isFinite(probability) && probability >= TRIGGER_THRESHOLD && probability <= 1 ? state.hits + 1 : 0;
  const fire = hits >= 2;
  return { state: { hits, fired: fire, lastKey: key, probability: probability ?? undefined }, fire };
}

export function parseRules(value: unknown): LiveRule[] {
  if (!Array.isArray(value) || value.length > RULE_LIMIT) throw new Error(`Use at most ${RULE_LIMIT} rules.`);
  const ids = new Set<string>();
  return value.map(item => {
    if (!item || typeof item !== 'object') throw new Error('Each rule must be an object.');
    const r = item as Record<string, unknown>;
    if (typeof r.id !== 'string' || !/^[\w-]{1,80}$/.test(r.id) || ids.has(r.id)) throw new Error('Rule IDs must be unique letters, digits, dashes or underscores.');
    ids.add(r.id);
    if (['quality', '__proto__', 'constructor', 'prototype'].includes(r.id)) throw new Error('That rule ID is reserved.');
    if (!['scene', 'webcam', 'feed'].includes(String(r.source))) throw new Error('Choose scene, webcam or feed as the source.');
    if (!['mark', 'chime', 'stop', 'direct'].includes(String(r.action))) throw new Error('Choose mark, chime, stop or direct as the action.');
    if (typeof r.condition !== 'string' || !r.condition.trim() || r.condition.length > 500) throw new Error('Each condition needs 1–500 characters.');
    if (typeof r.direction !== 'string' || r.direction.length > 500 || (r.action === 'direct' && !r.direction.trim())) throw new Error('Director actions need a direction of 1–500 characters.');
    if (typeof r.enabled !== 'boolean') throw new Error('Each rule needs an enabled boolean.');
    return { id: r.id, source: r.source as WatchSource, condition: r.condition.trim(), action: r.action as RuleAction, direction: r.direction.trim(), enabled: r.enabled };
  });
}

export function eventMemory(events: RuleEvent[]): string[] {
  return events.slice(-12).map(e => `[${e.source}, clip ${e.shotId}, ${e.seconds.toFixed(1)}s] Possible observation: ${e.condition} (${Math.round(e.probability * 100)}% model probability).`);
}

/** A feed adapter intentionally consumes text data only, never executes it. */
export function feedText(data: unknown): string {
  if (typeof data === 'string') return data.slice(0, 4000);
  if (!data || typeof data !== 'object') throw new Error('Feed must return text, { text }, or { items: [{ title, text }] }.');
  const value = data as { text?: unknown; items?: unknown };
  if (typeof value.text === 'string') return value.text.slice(0, 4000);
  if (Array.isArray(value.items)) return value.items.slice(0, 10).map(item => {
    if (!item || typeof item !== 'object') return '';
    const v = item as { title?: unknown; text?: unknown; content_text?: unknown };
    return [v.title, v.text ?? v.content_text].filter(x => typeof x === 'string').join(': ');
  }).join('\n').slice(0, 4000);
  throw new Error('Feed must return { text } or { items: [{ title, text }] }.');
}
