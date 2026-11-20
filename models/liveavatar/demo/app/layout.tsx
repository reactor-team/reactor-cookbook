import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "LiveKit avatar agent",
  description: "A voice agent that speaks through a talking avatar served by Reactor.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
