import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // ffmpeg-static resolves its binary path via `path.join(__dirname, ...)`.
  // If webpack bundles it into .next/server/vendor-chunks, __dirname points
  // there instead of node_modules/ffmpeg-static, so the binary can't be found.
  // Keeping it external makes Next require() it from node_modules at runtime.
  serverExternalPackages: ["ffmpeg-static"],
};

export default nextConfig;
