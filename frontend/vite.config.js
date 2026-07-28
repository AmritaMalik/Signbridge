import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  root: './',
  server: {
    port: 3000,
    proxy: {
      '/predict_landmarks': 'http://127.0.0.1:8000',
      '/predict_phrase': 'http://127.0.0.1:8000',
      '/predict_word': 'http://127.0.0.1:8000',
      '/predict_sentence': 'http://127.0.0.1:8000',
      '/predict': 'http://127.0.0.1:8000'
    }
  },
  build: {
    outDir: '../static',
    emptyOutDir: false
  }
});
