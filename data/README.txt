
DATASET BUILD SUMMARY
=====================
Root: /mnt/data
Output: /mnt/data/corpus_out

Files read (if existed):
- banglabooktrain.csv → split=train, lang=bn
- banglabookvalidation.csv → split=val, lang=bn
- banglabooktest.csv → split=test, lang=bn
- BnSentMix.csv → split=train, lang=bn
- hinglish.csv → split=train, lang=hi-en
- train/val/test_text(.txt) + *_labels(.txt) → split as named, lang=en

Canonical columns: id, text, label, lang, source, split
Label space: negative / neutral / positive
Dedup: md5 on normalized text
Lengths kept: 3..2000 chars

Exports:
- corpus_out/corpus_all.csv (everything merged)
- corpus_out/train.csv
- corpus_out/val.csv
- corpus_out/test.csv

Next: use these CSVs directly with scikit-learn or HuggingFace trainers.
