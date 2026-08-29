import type { Metadata } from "next";
import "@reactor-team/ui/styles.css";
import "./globals.css";

export const metadata: Metadata = {
  title: "Helios Interactive",
  description: "Real-time interactive video generation with Reactor + Helios",
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
