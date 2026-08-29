import type { Metadata } from "next";
import "@reactor-team/ui/styles.css";
import "./globals.css";

export const metadata: Metadata = {
  title: "Lingbot Interactive",
  description:
    "Real-time interactive world model with Reactor + Lingbot — drive a scene with WASD",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-zinc-950 text-zinc-100 antialiased">
        {children}
      </body>
    </html>
  );
}
