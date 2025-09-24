/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        negative: '#ef4444',
        neutral: '#6b7280',
        positive: '#22c55e',
      },
      borderRadius: {
        '2xl': '1rem',
      },
      fontFamily: {
        inter: ['Inter', 'ui-sans-serif', 'system-ui'],
      },
    },
    container: { center: true, screens: { xl: '1200px' } },
  },
  plugins: [],
}

