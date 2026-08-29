import type { Metadata } from "next";
import "@reactor-team/ui/styles.css";
import "./globals.css";

export const metadata: Metadata = {
  title: "LTX",
  description:
    "Real-time streaming talking-avatar generation with Reactor + LTX",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
