"use client";

import { useEffect, useRef, useState } from "react";

/**
 * Song upload → offline extraction. The file goes to POST /api/extract,
 * which runs the extractor pipeline (stem separation, rhythm, structure,
 * feature curves) as a server-side subprocess — the browser never
 * analyzes audio. Extraction takes minutes; this panel polls the job
 * and calls onExtracted when the new bundle is ready in the picker.
 */
export function UploadPanel({
  onExtracted,
  disabled,
}: {
  onExtracted: (songId: string) => void;
  disabled?: boolean;
}) {
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [logTail, setLogTail] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const busy = jobId !== null && status !== "done" && status !== "error";

  const upload = async (file: File | undefined | null) => {
    if (!file || disabled || busy) return;
    setError(null);
    setStatus("uploading");
    setLogTail([]);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch("/api/extract", { method: "POST", body: form });
      const data = (await res.json()) as { jobId?: string; error?: string };
      if (!res.ok || !data.jobId) throw new Error(data.error ?? `Upload failed: ${res.status}`);
      setJobId(data.jobId);
      setStatus("running");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
      setStatus(null);
      setJobId(null);
    }
  };

  // Poll the running job.
  useEffect(() => {
    if (!jobId || status === "done" || status === "error") return;
    const timer = setInterval(async () => {
      try {
        const res = await fetch(`/api/extract?job=${encodeURIComponent(jobId)}`);
        const data = (await res.json()) as {
          status?: string;
          songId?: string;
          logTail?: string[];
          error?: string;
        };
        if (!res.ok) throw new Error(data.error ?? `Status failed: ${res.status}`);
        setLogTail(data.logTail ?? []);
        if (data.status === "done") {
          setStatus("done");
          onExtracted(data.songId ?? "");
        } else if (data.status === "error") {
          setStatus("error");
          setError(data.error ?? "Extraction failed");
        }
      } catch {
        // Transient poll failure (dev-server recompile etc.) — keep trying.
      }
    }, 3000);
    return () => clearInterval(timer);
  }, [jobId, status, onExtracted]);

  return (
    <div className="w-full">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          void upload(Array.from(e.dataTransfer.files).find((f) => /\.(mp3|wav|flac|ogg|m4a)$/i.test(f.name)));
        }}
        onClick={() => !disabled && !busy && inputRef.current?.click()}
        className={`flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed p-6 text-center backdrop-blur-xl transition-colors ${
          dragOver
            ? "border-brand/60 bg-brand/10"
            : "border-white/15 bg-white/[0.03] hover:border-white/30 hover:bg-white/[0.06]"
        } ${disabled || busy ? "pointer-events-none opacity-60" : ""}`}
      >
        {!busy ? (
          <>
            <div className="text-2xl">♫</div>
            <p className="mt-2 text-sm text-zinc-200">
              Drop a new song here, or click to choose
            </p>
            <p className="mt-1 text-xs text-zinc-500">
              mp3 / wav / flac — runs the offline extractor (takes a few
              minutes, GPU-heavy)
            </p>
          </>
        ) : (
          <>
            <div className="h-5 w-5 animate-spin rounded-full border-2 border-white/15 border-t-brand" />
            <p className="mt-2 text-sm text-zinc-200">
              Extracting… stems, rhythm, structure, feature curves
            </p>
            {logTail.length > 0 && (
              <p className="mt-1 max-w-full truncate font-mono text-[10px] text-zinc-500">
                {logTail[logTail.length - 1]}
              </p>
            )}
          </>
        )}
      </div>

      {status === "done" && (
        <p className="mt-2 text-center text-xs text-emerald-300">
          Extraction complete — the song is now in the list above.
        </p>
      )}
      {error && (
        <p className="mt-2 text-center text-xs text-red-300">{error}</p>
      )}

      <input
        ref={inputRef}
        type="file"
        accept="audio/*,.mp3,.wav,.flac,.ogg,.m4a"
        className="hidden"
        onChange={(e) => void upload(e.target.files?.[0])}
      />
    </div>
  );
}
