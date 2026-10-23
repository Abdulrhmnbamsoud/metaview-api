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
        // MetaView command-center design tokens
        surface: {
          DEFAULT: '#0b1326',
          canvas: '#0A0E1A',
          raised: '#0F172A',
          overlay: '#1E293B',
          bright: '#31394d',
          lowest: '#060e20',
          low: '#131b2e',
          container: '#171f33',
          high: '#222a3d',
          highest: '#2d3449',
        },
        brand: { DEFAULT: '#4cd7f6', container: '#06b6d4', on: '#003640', light: '#acedff' },
        ink: { DEFAULT: '#dae2fd', variant: '#bcc9cd', outline: '#869397' },
        edge: { subtle: '#1E293B', strong: '#334155' },
        threat: { critical: '#EF4444', high: '#F97316', medium: '#F59E0B', low: '#10B981' },
        epistemic: { fact: '#10B981', inference: '#06B6D4', hypothesis: '#F59E0B', unknown: '#64748B' },
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
        sans: ['Geist', 'Cairo', 'Outfit', 'sans-serif'],
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
