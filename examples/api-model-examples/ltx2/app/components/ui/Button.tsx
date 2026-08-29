import type { ButtonHTMLAttributes, ReactNode } from "react";
import { cn, FOCUS_RING } from "./ui";

type Variant = "primary" | "secondary" | "ghost";
type Size = "sm" | "md";

const VARIANT: Record<Variant, string> = {
  primary:
    "bg-brand text-brand-fg font-medium hover:brightness-110 disabled:opacity-40",
  secondary:
    "border border-zinc-700 text-zinc-200 hover:bg-zinc-800 disabled:opacity-40",
  ghost: "text-zinc-400 hover:text-zinc-200 disabled:opacity-40",
};
const SIZE: Record<Size, string> = {
  sm: "px-3 py-1.5 text-xs",
  md: "px-3 py-2 text-sm",
};

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  leadingIcon?: ReactNode;
  children: ReactNode;
}

export function Button({
  variant = "primary",
  size = "md",
  leadingIcon,
  className,
  children,
  ...rest
}: ButtonProps) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-1.5 rounded-md tracking-tight transition disabled:cursor-not-allowed",
        VARIANT[variant],
        SIZE[size],
        FOCUS_RING,
        className,
      )}
      {...rest}
    >
      {leadingIcon}
      {children}
    </button>
  );
}
