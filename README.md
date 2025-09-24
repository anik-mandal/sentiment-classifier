What we’ve done so far 
•	Multilingual classifier trained on XLM‑RoBERTa with class‑weighted loss (handles English/Hindi/Hinglish).
•	Exported logits for calibration: logits_val.npy & logits_test.npy
•	Planned multi‑input inference: single text, CSV batch, and PDF ingestion 
•	Now: the first version of the app is built (screenshots incoming) with minimal, colorful UI.
What the current app does (v0)
•	Loads the latest checkpoint from the xlmr_weighted directory and runs predictions.
•	Accepts single sentence, CSV, and PDF; shows label + confidence and basic summaries.
•	Generates simple word clouds; designed to stay lightweight for local use.
