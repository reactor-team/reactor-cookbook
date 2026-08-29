import type { ButtonHTMLAttributes } from "react";
import { cn, FOCUS_RING } from "./ui";
import { Icon, type IconName } from "./Icon";

interface IconButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  icon: IconName;
  label: string; // aria-label + title
  tone?: "default" | "danger";
}

export function IconButton({
  icon,
  label,
  tone = "default",
  className,
  ...rest
}: IconButtonProps) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      className={cn(
        "inline-flex size-8 shrink-0 items-center justify-center rounded-md border transition-colors disabled:cursor-not-allowed disabled:opacity-40",
        tone === "danger"
          ? "border-red-900/60 text-red-300 hover:bg-red-950/40"
          : "border-zinc-700 text-zinc-300 hover:bg-zinc-800",
        FOCUS_RING,
        className,
      )}
      {...rest}
    >
      <Icon name={icon} />
    </button>
  );
}
