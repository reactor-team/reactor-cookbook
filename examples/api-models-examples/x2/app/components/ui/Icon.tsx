import type { ReactElement, SVGProps } from "react";
import { cn } from "./ui";

export type IconName =
  | "play"
  | "pause"
  | "reset"
  | "power"
  | "chevron"
  | "x"
  | "download"
  | "check"
  | "plus"
  | "scissors"
  | "dot";

// Per-icon inner geometry. Stroke icons rely on the <svg> stroke; fill
// icons (play/pause/dot) set their own fill + clear the stroke.
const PATHS: Record<IconName, ReactElement> = {
  play: <polygon points="7 5 19 12 7 19" fill="currentColor" stroke="none" />,
  pause: (
    <g fill="currentColor" stroke="none">
      <rect x="7" y="5" width="3.5" height="14" rx="1" />
      <rect x="13.5" y="5" width="3.5" height="14" rx="1" />
    </g>
  ),
  reset: (
    <>
      <path d="M20 11.5A8 8 0 1 1 17 5.3" />
      <path d="M20 3.5v4h-4" />
    </>
  ),
  power: (
    <>
      <path d="M12 4v8" />
      <path d="M7.5 7a7 7 0 1 0 9 0" />
    </>
  ),
  chevron: <path d="M6 9l6 6 6-6" />,
  x: <path d="M6 6l12 12M18 6L6 18" />,
  download: (
    <>
      <path d="M12 4v10" />
      <path d="M8 11l4 4 4-4" />
      <path d="M5 19h14" />
    </>
  ),
  check: <path d="M4 12l5 5L20 6" />,
  plus: <path d="M12 5v14M5 12h14" />,
  scissors: (
    <>
      <circle cx="6" cy="6" r="3" />
      <circle cx="6" cy="18" r="3" />
      <path d="M8.5 8.5L20 20M8.5 15.5L20 4" />
    </>
  ),
  dot: <circle cx="12" cy="12" r="5" fill="currentColor" stroke="none" />,
};

interface IconProps extends Omit<SVGProps<SVGSVGElement>, "name"> {
  name: IconName;
  className?: string;
}

export function Icon({ name, className, ...rest }: IconProps) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
      className={cn("size-4", className)}
      {...rest}
    >
      {PATHS[name]}
    </svg>
  );
}
