import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8080",
      "/healthz": "http://127.0.0.1:8080",
      "/readyz": "http://127.0.0.1:8080",
      "/stackz": "http://127.0.0.1:8080",
      "/annotation": "http://127.0.0.1:8080",
      "/benchmark": "http://127.0.0.1:8080",
      "/prompt": "http://127.0.0.1:8080",
      "/dashboard": {
        target: "http://127.0.0.1:8080",
        ws: true,
      },
      "/projects": "http://127.0.0.1:8080",
    },
  },
});
