import { cn, FOCUS_RING } from "./ui";

export interface SegmentOption<T extends string> {
  value: T;
  label: string;
  /** Optional muted suffix, e.g. " · soft". */
  hint?: string;
}

interface SegmentedToggleProps<T extends string> {
  options: SegmentOption<T>[];
  value: T;
  onChange: (next: T) => void;
  "aria-label"?: string;
}

export function SegmentedToggle<T extends string>({
  options,
  value,
  onChange,
  ...rest
}: SegmentedToggleProps<T>) {
  return (
    <div
      role="group"
      aria-label={rest["aria-label"]}
      className="flex overflow-hidden rounded-md border border-zinc-700"
    >
      {options.map((opt, i) => {
        const active = opt.value === value;
        return (
          <button
            key={opt.value}
            type="button"
            aria-pressed={active}
            onClick={() => onChange(opt.value)}
            className={cn(
              "flex-1 px-2.5 py-1.5 text-xs font-medium tracking-tight transition-colors",
              i > 0 && "border-l border-zinc-700",
              active
                ? "bg-zinc-700 text-zinc-100"
                : "text-zinc-400 hover:text-zinc-200",
              FOCUS_RING,
            )}
          >
            {opt.label}
            {opt.hint && <span className="text-zinc-500">{opt.hint}</span>}
          </button>
        );
      })}
    </div>
  );
}
