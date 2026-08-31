/**
 * Mock-mode only: derive procedural visual parameters from a timeline
 * entry's prompt text, so the placeholder canvas visibly changes with
 * the score exactly where the real model would receive `set_prompt`.
 *
 * This is a crude keyword sketch on purpose — in live mode the prompt
 * text goes to the world model verbatim and none of this runs.
 */

import type { MockVisualParams } from "./types";

const HUE_WORDS: [RegExp, number][] = [
  [/\b(amber|gold|golden|sun|dawn|warm|honey)\b/i, 40],
  [/\b(crimson|red|rose|ember|molten)\b/i, 10],
  [/\b(violet|purple|magenta|indigo)\b/i, 280],
  [/\b(blue|azure|ocean|sea|water|ice|frost)\b/i, 215],
  [/\b(teal|cyan|aqua)\b/i, 185],
  [/\b(green|forest|moss|aurora|emerald)\b/i, 140],
  [/\b(silver|pale|mist|moon|grey|gray|white)\b/i, 210],
  [/\b(night|black|dark|void)\b/i, 250],
];

const FAST_WORDS =
  /\b(storm|surge|surging|racing|burst|bursting|rapid|charged|electric|cascade|pouring|fast|wild|crash)\b/i;
const SLOW_WORDS =
  /\b(still|calm|slow|gentle|drift|drifting|quiet|breathing|serene|soft|imperceptibl)\b/i;
const BRIGHT_WORDS =
  /\b(radiant|brilliant|luminous|bright|glowing|blazing|light)\b/i;
const DARK_WORDS = /\b(night|dark|dim|shadow|dusk|twilight|black|deep)\b/i;

function hashHue(text: string): number {
  let h = 0;
  for (let i = 0; i < text.length; i++) h = (h * 31 + text.charCodeAt(i)) | 0;
  return Math.abs(h) % 360;
}

export function visualsFromPrompt(prompt: string): MockVisualParams {
  const hues: number[] = [];
  for (const [re, hue] of HUE_WORDS) {
    if (re.test(prompt)) hues.push(hue);
    if (hues.length >= 2) break;
  }
  const hueA = hues[0] ?? hashHue(prompt);
  const hueB = hues[1] ?? (hueA + 50) % 360;

  let speed = 0.35;
  if (FAST_WORDS.test(prompt)) speed = 0.9;
  else if (SLOW_WORDS.test(prompt)) speed = 0.15;

  let brightness = 0.5;
  if (BRIGHT_WORDS.test(prompt)) brightness += 0.3;
  if (DARK_WORDS.test(prompt)) brightness -= 0.25;

  return {
    hueA,
    hueB,
    speed,
    turbulence: speed * 0.8,
    brightness: Math.min(1, Math.max(0.1, brightness)),
  };
}
