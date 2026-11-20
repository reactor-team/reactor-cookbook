import type { Metadata } from 'next';
import './globals.css';
export const metadata: Metadata = { title: 'Storyboard Previs — Reactor', description: 'A storyboard that generates itself: consistent cast, dialogue, and continuation with FastH3 Omni.', icons: { icon: '/brand/symbol.svg' } };
export default function RootLayout({ children }: { children: React.ReactNode }) { return <html lang="en"><body>{children}</body></html>; }
