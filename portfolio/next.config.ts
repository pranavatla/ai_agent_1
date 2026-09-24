import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Static export: atla.in is served from S3 + CloudFront, so there is no image server.
  output: "export",
  images: {
    unoptimized: true,
    remotePatterns: [
      { protocol: "https", hostname: "picsum.photos" },
      { protocol: "https", hostname: "images.unsplash.com" },
    ],
  },
};

export default nextConfig;
