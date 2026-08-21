import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Reactor World Model Arcade",
  description: "Seven controller-first worlds in one arcade cabinet.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
