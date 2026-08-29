import type { ButtonHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md";

const VARIANTS: Record<Variant, string> = {
  primary: "bg-primary text-primary-foreground hover:brightness-95",
  secondary:
    "border border-white/15 bg-white/10 text-white/80 hover:bg-white/15",
  ghost:
    "border border-white/10 text-white/60 hover:border-white/25 hover:text-white/90",
  danger:
    "border border-red-500/25 bg-red-500/15 text-red-300 hover:bg-red-500/25",
};

const SIZES: Record<Size, string> = {
  sm: "h-7 px-3 text-xs",
  md: "h-9 px-4 text-sm",
};

export function Button({
  variant = "primary",
  size = "md",
  className,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant;
  size?: Size;
}) {
  return (
    <button
      className={cn(
        "inline-flex select-none items-center justify-center gap-1.5 rounded-md font-medium transition disabled:pointer-events-none disabled:opacity-40",
        VARIANTS[variant],
        SIZES[size],
        className,
      )}
      {...props}
    />
  );
}
