# Sentiment Classifier – Upload Guide

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
