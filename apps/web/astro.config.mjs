// @ts-check
import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import node from '@astrojs/node';
import tailwindcss from '@tailwindcss/vite';

/**
 * Server output: pages are rendered on request so the dashboard always reflects
 * live pipeline state. Interactive pieces (the 3D cortical viewer, tables,
 * timelines) are React islands hydrated on the client.
 */
export default defineConfig({
  output: 'server',
  adapter: node({ mode: 'standalone' }),
  integrations: [react()],
  vite: {
    plugins: [tailwindcss()],
    server: {
      proxy: {
        // In dev the browser talks to Astro; /api is forwarded to FastAPI so
        // there is no CORS surface and no hard-coded origin in the client.
        '/api': {
          target: process.env.API_INTERNAL_BASE?.replace(/\/api$/, '') ?? 'http://localhost:8000',
          changeOrigin: true,
        },
      },
    },
  },
  server: { host: true, port: 4321 },
  devToolbar: { enabled: false },
});
