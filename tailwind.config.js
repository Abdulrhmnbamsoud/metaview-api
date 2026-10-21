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
        gold: {
          50: '#fbf8ee',
          100: '#f5efd5',
          200: '#ebdcab',
          300: '#dfc377',
          400: '#d4af37',
          500: '#c5a059',
          600: '#a8813a',
          700: '#87612e',
          800: '#6f4e29',
          900: '#5c4125',
        },
        dark: {
          bg: '#050508',
          card: '#0c0e14',
          subcard: '#12151e',
          border: 'rgba(197, 160, 89, 0.12)',
          borderHover: 'rgba(197, 160, 89, 0.35)',
        }
      },
      fontFamily: {
        sans: ['Cairo', 'Outfit', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'spin-slow': 'spin 12s linear infinite',
      }
    },
  },
  plugins: [],
}
