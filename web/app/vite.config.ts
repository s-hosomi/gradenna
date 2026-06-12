import { defineConfig } from "vite";
import path from "path";

export default defineConfig({
  base: "./",
  resolve: {
    alias: {
      "@wasm": path.resolve(__dirname, "../wasm-kernel/pkg"),
    },
  },
  server: {
    fs: {
      allow: [
        path.resolve(__dirname, ".."),
        path.resolve(__dirname, "../wasm-kernel"),
      ],
    },
  },
  build: {
    target: "es2020",
    rollupOptions: {
      output: {
        manualChunks: {
          three: ["three"],
        },
      },
    },
  },
  optimizeDeps: {
    exclude: [],
  },
});
