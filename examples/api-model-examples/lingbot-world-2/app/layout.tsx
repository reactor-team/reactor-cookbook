import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LingBot World 2",
  description:
    "Steer a generated world with WASD + arrows from a starting image and prompt.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  // The app is dark-only; setting the class here (rather than from a
  // client effect) lets server-rendered pages like <SetupRequired />
  // pick up the dark theme tokens too.
  return (
    <html lang="en" className="dark">
      <body className="antialiased font-sans">{children}</body>
    </html>
  );
}
