import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "HappyOyster",
  description:
    "Build a world from a prompt, then travel it live, WASD in Adventure, text in Directing.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  // Dark-only: setting the class here (not from a client effect) lets
  // server-rendered pages like <SetupRequired /> pick up the dark tokens too.
  return (
    <html lang="en" className="dark">
      <body className="antialiased font-sans">{children}</body>
    </html>
  );
}
