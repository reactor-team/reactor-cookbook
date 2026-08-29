import { cn, FOCUS_RING } from "./ui";

interface SwitchProps {
  checked: boolean;
  onChange: (next: boolean) => void;
  disabled?: boolean;
  "aria-label"?: string;
  "aria-labelledby"?: string;
}

/**
 * A binary on/off switch. Unlike a form checkbox it reads as a live control:
 * the track fills with the brand color when on and the thumb slides across, so
 * a state change is legible at a glance. `role="switch"` + `aria-checked` keep
 * it a proper toggle for assistive tech; the underlying <button> gives keyboard
 * activation for free.
 */
export function Switch({
  checked,
  onChange,
  disabled = false,
  ...rest
}: SwitchProps) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={rest["aria-label"]}
      aria-labelledby={rest["aria-labelledby"]}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={cn(
        "relative inline-flex h-5 w-9 shrink-0 items-center rounded-full transition-colors disabled:cursor-not-allowed disabled:opacity-40",
        checked ? "bg-brand" : "bg-zinc-700",
        FOCUS_RING,
      )}
    >
      <span
        className={cn(
          "inline-block h-4 w-4 rounded-full bg-white shadow transition-transform",
          checked ? "translate-x-[18px]" : "translate-x-0.5",
        )}
      />
    </button>
  );
}
