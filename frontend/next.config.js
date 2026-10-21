/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
  // served from the root of biodreamer.ciyalabs.com; set NEXT_PUBLIC_BASE_PATH to host it under a sub-path
  basePath: process.env.NEXT_PUBLIC_BASE_PATH || "",
  images: {
    unoptimized: true,
  },
  trailingSlash: true,
};

module.exports = nextConfig;
