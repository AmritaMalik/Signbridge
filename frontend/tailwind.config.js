/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        space: '#050814',
        card: 'rgba(13, 18, 38, 0.75)',
        cyan: {
          400: '#06b6d4',
          500: '#00e5ff',
        },
        purple: {
          500: '#a855f7',
          600: '#9333ea',
        },
        emerald: {
          400: '#34d399',
          500: '#10b981',
        }
      },
      fontFamily: {
        sans: ['Outfit', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      }
    },
  },
  plugins: [],
}
