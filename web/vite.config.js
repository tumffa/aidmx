import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: './',              // ensure relative asset paths in dist
  server: {
    port: 5173,
    strictPort: true
  },
  build: {
    outDir: 'dist',        // default, explicit for clarity
    emptyOutDir: true
  }
})