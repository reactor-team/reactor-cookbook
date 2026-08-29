import { cn, EYEBROW, FOCUS_RING, PANEL } from "./ui";

// Surfaces command_error messages from the model. The shell owns the
// message subscription and the auto-dismiss TTL; this component is purely
// presentational.
export function CommandError({
  message,
  onDismiss,
}: {
  message: string;
  onDismiss: () => void;
}) {
  return (
    <div
      className={cn(PANEL, "border-red-900/50 bg-red-950/20 p-3")}
      role="alert"
    >
      <div className="flex items-start justify-between gap-2">
        <span className={cn(EYEBROW, "text-red-500")}>Command failed</span>
        <button
          type="button"
          aria-label="Dismiss error"
          onClick={onDismiss}
          className={cn(
            "rounded text-red-400/60 transition hover:text-red-300",
            FOCUS_RING,
          )}
        >
          ✕
        </button>
      </div>
      <p className="mt-1 text-sm text-red-300">{message}</p>
    </div>
  );
}
