import type { Metadata } from "next";
import "@reactor-team/ui/styles.css";
import "./globals.css";

export const metadata: Metadata = {
  title: "XMAX X2",
  description: "Real-time video-to-video editing with Reactor + XMAX X2",
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
