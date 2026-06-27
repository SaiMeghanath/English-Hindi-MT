# English–Hindi Dataset Processing and Translation

This repository contains my submission for **Assessment 1** (Dataset Processing and Analysis) and **Assessment 2** (Translation with LLM).

## Repository Structure

```
.
│   ├── process_dataset.py          # Dataset cloning, filtering, word-count analysis
│   └── english_hindi_cleaned.xlsx  # Final cleaned dataset (output of Assignment 1)
├── assignment2/
│   ├── translate_and_score.py      # LLM-based translation + BLEU/chrF/TER scoring
│   ├── translation_output.xlsx     # English sentences + model-generated Hindi translations
│   └── scores.txt                  # BLEU, chrF, TER scores
└── README.md
```

---

## Assignment 1: English–Hindi Dataset Processing and Analysis

**Dataset:** [`ainlpml/english-hindi`](https://huggingface.co/datasets/ainlpml/english-hindi) (Hugging Face, gated access — requires accepting dataset terms and a personal access token).

**Pipeline:**
1. Downloaded `eng.txt` and `hin.txt` (10,000 line-aligned English–Hindi sentence pairs).
2. Computed word counts for every sentence in both languages.
3. Filtered to keep only sentence pairs where **both** the English and Hindi sentence had a word count between **5 and 50**.
4. Calculated the word count difference (English − Hindi) for each pair and retained only pairs where the difference was between **-10 and +10**.
5. Exported the cleaned dataset to Excel with columns: `English Sentences`, `Hindi Sentences`, `Word Count (English)`, `Word Count (Hindi)`.

**Results:**

| Stage | Rows |
|---|---|
| Raw dataset loaded | 10,000 |
| After word-count range filter (5–50, both languages) | 8,788 |
| After word-count difference filter (-10 to +10) | 8,214 |

### How to run

```bash
pip install huggingface_hub pandas openpyxl --break-system-packages
```

Set your Hugging Face access token (required since the dataset is gated — accept the dataset terms on the HF page first, then generate a token under HF Settings → Access Tokens):

```powershell
$env:HF_TOKEN = "your_hf_token_here"
```

Run:

```bash
python assignment1/process_dataset.py
```

Output: `english_hindi_cleaned.xlsx`

---

## Assignment 2: Translation with LLM

**Model used:** [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M) — an open, freely available multilingual translation model (no API key required).

**Pipeline:**
1. Randomly sampled 100 English sentences from the Assignment 1 cleaned dataset (fixed random seed for reproducibility).
2. Translated each English sentence into Hindi using the NLLB model.
3. Used the existing reference Hindi sentences (from Assignment 1's parallel dataset) as ground truth.
4. Computed **BLEU**, **chrF**, and **TER** scores comparing model translations against the reference Hindi sentences, using `sacrebleu`.
5. Saved the translation output to Excel (`English Sentence`, `Model-Generated Hindi Translation`) and the scores to a text file.

### How to run

```bash
pip install transformers torch sentencepiece pandas openpyxl sacrebleu --break-system-packages
```

Ensure `english_hindi_cleaned.xlsx` (from Assignment 1) is in the same directory, then run:

```bash
python assignment2/translate_and_score.py
```

Outputs:
- `translation_output.xlsx`
- `scores.txt`

**Note:** the NLLB model (~2.4 GB) is downloaded automatically on first run. No GPU required, though translation will be faster with one.

---

## Author

Sai Meghanath
Final-semester MCA (Artificial Intelligence), Amrita Vishwa Vidyapeetham
