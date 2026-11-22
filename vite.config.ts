import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Project site: https://fuerostic.github.io/ai-mastery/
export default defineConfig({
  base: "/ai-mastery/",
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
  },
  preview: {
    host: "127.0.0.1",
    port: 4173,
    strictPort: true,
  },
});
