# Sentiment Classifier � Upload Guide

<<<<<<< HEAD
This repo should include only source code. Heavy/generated/local files are ignored via .gitignore files.

What is included
- Backend Python scripts (in root): app.py, check.py, evaluation.py, export_logits.py, predict_cli.py, train_tfidf_svm.py, train_xlmr.py, train_xlmr_weighted.py, tune_bias_and_eval.py
- Requirements: requirements.txt, requirements_ui.txt
- Frontend app: web/ (but without node_modules, build output, or env secrets)
- data/README.txt (but not CSVs)

What is excluded automatically
- data/*.csv, models/, releases/, runtime/
- venv/, __pycache__/, *.pyc, *.bak
- web/node_modules/, web/dist/ (or build/), web/.env*
- installers/logs/OS junk

Step-by-step: Upload only the right files
1) Create GitHub repo
   - On GitHub, click New repository, keep it empty (no README/template).

2) Initialize Git in the root folder
   - Open Terminal in: D:\sentiment classifier
   - Run:
     git init
     git config user.name "Your Name"
     git config user.email "you@example.com"

3) Confirm ignore rules work
   - Files like data/train.csv, models/, venv/, web/node_modules/ should NOT appear staged.
   - Run:
     git status

4) Stage and commit the allowed files
   - Run:
     git add .
     git status   (double-check only source is added)
     git commit -m "Initial commit: backend + web source"

5) Push to GitHub
   - Replace URL with your repo:
     git branch -M main
     git remote add origin https://github.com/<your-username>/<your-repo>.git
     git push -u origin main

How to run (local rebuild)
Backend
- python -m venv venv
- venv\Scripts\activate
- pip install -r requirements.txt
- python app.py   (or the correct start command)

Frontend (web)
- cd web
- npm install
- npm run dev   (or: npm run build && npm run preview)
=======
This repo includes the backend Python scripts and the web UI source only. Heavy, private, and generated files are ignored via .gitignore.

How to run
- Backend
  - python -m venv venv
  - venv\Scripts\activate
  - pip install -r requirements.txt
  - python app.py
- Frontend (web)
  - cd web
  - npm install
  - npm run dev

Notes
- Do not commit data/*.csv, models/, venv/, or web/node_modules/.
>>>>>>> temp-branch
