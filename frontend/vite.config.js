import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteStaticCopy } from "vite-plugin-static-copy";

// Cesium ships workers, widgets and assets that must be served as static files.
const cesiumSource = "node_modules/cesium/Build/Cesium";
const cesiumBaseUrl = "cesiumStatic";
const backend = process.env.GEO_VLA_BACKEND || "http://localhost:8000";

export default defineConfig({
  define: { CESIUM_BASE_URL: JSON.stringify(`/${cesiumBaseUrl}`) },
  plugins: [
    react(),
    viteStaticCopy({
      targets: [
        ...["ThirdParty", "Workers", "Assets", "Widgets"].map((dir) => ({
          src: `${cesiumSource}/${dir}`,
          dest: cesiumBaseUrl,
          rename: { stripBase: 4 }, // node_modules/cesium/Build/Cesium/X → cesiumStatic/X
        })),
        // MapLibre v6's ES-module worker (+ the chunk it imports) must be served as files.
        ...["maplibre-gl-worker.mjs", "maplibre-gl-shared.mjs"].map((file) => ({
          src: `node_modules/maplibre-gl/dist/${file}`,
          dest: "maplibre",
          rename: { stripBase: true },
        })),
      ],
    }),
  ],
  build: { chunkSizeWarningLimit: 5000 }, // Cesium is large by nature and lazy-loaded
  server: {
    proxy: { "/query": backend, "/health": backend, "/tools": backend },
  },
});
