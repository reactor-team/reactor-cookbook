import type { Metadata } from "next";
import "@reactor-team/ui/styles.css";
import "./globals.css";

export const metadata: Metadata = {
  title: "NaadaKai",
  description: "Turn a song into a generative world you can walk through.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-zinc-950 text-zinc-100 antialiased">
        {/* Fixed background image + dark overlay behind everything — this
            is what the glassmorphism panels are actually "glassing" over.
            z-0 + the content wrapper's z-10 pins stacking order explicitly
            rather than relying on DOM-order stacking-context guesses.

            The image itself is HEAVILY blurred (blur-[80px]) so a busy
            maximalist/brutalist source collapses into a soft neon color
            wash — the detail and any baked-in text/UI dissolve, leaving
            only mood, which reads well under the frosted panels. scale-125
            + overflow-hidden hides the transparent halo a big blur radius
            leaves at the element edges. Image lives at
            /images/world-bg.png; the gradient alone is a safe dark
            fallback if the file is missing (no broken-image flash). During
            live world playback the full-bleed video covers this entirely —
            it only shows on the picker/menu screens. */}
        <div className="fixed inset-0 z-0 overflow-hidden bg-[#050308]">
          <div
            className="absolute inset-0 scale-125 bg-cover bg-center"
            style={{ backgroundImage: "url(/images/world-bg.png)" }}
          />
          <div className="absolute inset-0 bg-gradient-to-b from-black/40 via-black/55 to-black/85" />
        </div>
        <div className="relative z-10">{children}</div>
      </body>
    </html>
  );
}
