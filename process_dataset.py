"""
Assessment 1: English-Hindi Dataset Processing and Analysis
Dataset: ainlpml/english-hindi (gated) -> eng.txt + hin.txt, line-aligned parallel text files

Pipeline: download -> pair lines -> word count -> filter -> diff filter -> export to Excel
"""

import os
import re
import pandas as pd
from huggingface_hub import hf_hub_download

# -----------------------------
# CONFIG
# -----------------------------
REPO_ID = "ainlpml/english-hindi"
ENG_FILE = "eng.txt"
HIN_FILE = "hin.txt"

MIN_ROWS_REQUIRED = 10000
WORD_COUNT_MIN = 5
WORD_COUNT_MAX = 50
DIFF_MIN = -10
DIFF_MAX = 10

OUTPUT_PATH = "/mnt/user-data/outputs/english_hindi_cleaned.xlsx"

HF_TOKEN = os.environ.get("HF_TOKEN")  # set this env var before running, or paste token below


def word_count(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(re.findall(r"\S+", text.strip()))


def download_files():
    print(f"Downloading {ENG_FILE} and {HIN_FILE} from {REPO_ID} ...")
    eng_path = hf_hub_download(repo_id=REPO_ID, filename=ENG_FILE, repo_type="dataset", token=HF_TOKEN)
    hin_path = hf_hub_download(repo_id=REPO_ID, filename=HIN_FILE, repo_type="dataset", token=HF_TOKEN)
    print(f"Downloaded:\n  {eng_path}\n  {hin_path}")
    return eng_path, hin_path


def load_pairs(eng_path, hin_path):
    with open(eng_path, "r", encoding="utf-8") as f:
        eng_lines = [line.strip() for line in f.readlines()]
    with open(hin_path, "r", encoding="utf-8") as f:
        hin_lines = [line.strip() for line in f.readlines()]

    print(f"eng.txt lines: {len(eng_lines)} | hin.txt lines: {len(hin_lines)}")

    n = min(len(eng_lines), len(hin_lines))
    if len(eng_lines) != len(hin_lines):
        print(f"WARNING: line count mismatch! Truncating both to {n} lines.")

    df = pd.DataFrame({"English": eng_lines[:n], "Hindi": hin_lines[:n]})
    df = df[(df["English"] != "") & (df["Hindi"] != "")]
    df = df.reset_index(drop=True)
    return df


def main():
    eng_path, hin_path = download_files()
    df = load_pairs(eng_path, hin_path)
    print(f"Loaded {len(df)} clean sentence pairs.")

    if len(df) < MIN_ROWS_REQUIRED:
        print(f"WARNING: only {len(df)} rows available, below required {MIN_ROWS_REQUIRED}.")

    # 3. Word Count Analysis
    df["Word Count (English)"] = df["English"].apply(word_count)
    df["Word Count (Hindi)"] = df["Hindi"].apply(word_count)

    df = df[
        df["Word Count (English)"].between(WORD_COUNT_MIN, WORD_COUNT_MAX)
        & df["Word Count (Hindi)"].between(WORD_COUNT_MIN, WORD_COUNT_MAX)
    ]
    print(f"After word-count range filter (5-50 both languages): {len(df)} rows")

    # 4. Word Count Difference Calculation
    df["Word Count Difference"] = df["Word Count (English)"] - df["Word Count (Hindi)"]
    df = df[df["Word Count Difference"].between(DIFF_MIN, DIFF_MAX)]
    print(f"After word-count difference filter (-10 to +10): {len(df)} rows")

    df = df.reset_index(drop=True)

    # 5. Final Output
    final_df = df.rename(columns={"English": "English Sentences", "Hindi": "Hindi Sentences"})
    final_df = final_df[
        ["English Sentences", "Hindi Sentences", "Word Count (English)", "Word Count (Hindi)", "Word Count Difference"]
    ]

    final_df.to_excel(OUTPUT_PATH, index=False, engine="openpyxl")
    print(f"Saved cleaned dataset with {len(final_df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
