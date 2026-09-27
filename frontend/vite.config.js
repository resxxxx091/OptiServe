import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

const target = process.env.VITE_PYTHON_API_URL || 'http://localhost:8000'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      '/api/python': {
        target,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/python/, '')
      },
      // /docs 能反代过去，但 Swagger 自己按根路径拉规范，不补这条它拿到的是 SPA 的 index.html
      '/openapi.json': { target, changeOrigin: true }
    }
  }
})
