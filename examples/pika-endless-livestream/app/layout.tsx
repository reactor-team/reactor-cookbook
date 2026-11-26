import type { Metadata } from "next";
import { Host_Grotesk, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";

// Reactor's brand faces are Aeonik (licensed, so not shipped here) with Host
// Grotesk as its fallback, and IBM Plex Mono for labels.
const sans = Host_Grotesk({ subsets: ["latin"], weight: ["400", "500", "700"], variable: "--font-sans" });
const mono = IBM_Plex_Mono({ subsets: ["latin"], weight: ["400", "500"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: "Pika Endless Livestream",
  description: "An endless live video stream on Pika H3 Flex, powered by Reactor.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable}`}>
      <body>{children}</body>
    </html>
  );
}
