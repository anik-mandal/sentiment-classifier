# Sentiment Classifier

This repo includes the backend Python scripts and web UI source code. Heavy, generated, and private files are ignored via .gitignore.

## What's Included
- Backend Python scripts (in root): app.py, check.py, evaluation.py, export_logits.py, predict_cli.py, train_tfidf_svm.py, train_xlmr.py, train_xlmr_weighted.py, tune_bias_and_eval.py
- Requirements: requirements.txt, requirements_ui.txt
- Frontend app: web/ (excluding node_modules, build output, and env secrets)
- data/README.txt (excluding CSVs)

## What's Excluded
- data/*.csv, models/, releases/, runtime/
- venv/, __pycache__/, *.pyc, *.bak
- web/node_modules/, web/dist/, web/.env*
- installers/logs/OS files

## How to Run

### Backend
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

### Frontend (web)
```bash
cd web
npm install
npm run dev
```
