# Multilingual Sentiment (XLM-R) UI

Premium single-page experience for running multilingual sentiment predictions against your local XLM-R backend.

## Requirements
- Node.js >= 20.11
- npm
- Backend API running at `${VITE_API_BASE}` (defaults to `http://127.0.0.1:7860`) with `/predict`, `/predict/batch`, and `/operating-point` routes.

## Setup
```bash
npm install
```

## Development
```bash
npm run dev
```
Then open http://127.0.0.1:5173.

### Operating point & policy modes
- The front end loads calibrated thresholds from `D:\sentiment classifier\releases\v0_2_calibrated\operating_point.json`.
- Toggle between modes with the segmented control:
  - **Balanced** → `mode="balanced"`
  - **High-precision (NEG)** → `mode="high_precision_negative"`
- The backend honours the mode via `predict_with_mode` and returns the copy used in the Remarks panel.

### Result view highlights
- Calibrated confidence bar + per-class bars obey the palette:
  - Positive `#22C55E`, Neutral `#F59E0B`, Negative `#EF4444`, Info `#06B6D4`
- Token attributions render as pill chips (max 5) with hover tooltips.
- “Copy remarks” button copies the auto-generated analyst summary.
- Animations use Framer Motion (fade-up 220 ms, bar grow 350 ms) and respect `prefers-reduced-motion`.

### Dev assertions
When `import.meta.env.DEV` is true the UI fires console assertions against the API to ensure:
- A clearly negative sample scores with confidence > 0.6 in balanced mode.
- Borderline neutral text is not marked negative in high-precision mode.
- A positive sample keeps a high (>80%) probability bar.

## Scripts reference
The original scaffold commands are kept for reference:
```bash
npm create vite@latest sentiment-ui -- --template react-ts
cd sentiment-ui
npm i framer-motion papaparse pdfjs-dist clsx
npm i -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
npm run dev
```
