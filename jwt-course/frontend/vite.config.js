import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

export default defineConfig({
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:55110",
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
    port: 5317
  },
  plugins: [svelte()],
  preview: { port: 5317 },
});

