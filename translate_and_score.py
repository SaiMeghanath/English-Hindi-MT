"""
Assessment 2: Translation with LLM
-----------------------------------
1. Selects 100 English sentences from the Assignment 1 cleaned dataset
   (english_hindi_cleaned.xlsx, produced by process_dataset.py).
2. Runs English -> Hindi translation using facebook/nllb-200-distilled-600M
   (a free, open Hugging Face MT model - no API key/token required).
3. Computes BLEU, chrF, and TER against the reference Hindi sentences
   already present in the Assignment 1 dataset (used as ground truth).
4. Saves:
   - translation_output.xlsx   (English sentence, Model-generated Hindi translation)
   - scores.txt                (BLEU / chrF / TER scores)

Run:
    pip install transformers torch sentencepiece pandas openpyxl sacrebleu --break-system-packages
    python translate_and_score.py
"""

import pandas as pd
import sacrebleu
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# -----------------------------
# CONFIG
# -----------------------------
INPUT_XLSX = "english_hindi_cleaned.xlsx"   # output of Assignment 1
N_SENTENCES = 100
MODEL_NAME = "facebook/nllb-200-distilled-600M"

# NLLB language codes (FLORES-200 codes)
SRC_LANG = "eng_Latn"
TGT_LANG = "hin_Deva"

OUTPUT_XLSX = "translation_output.xlsx"
SCORES_TXT = "scores.txt"


def load_sentences():
    df = pd.read_excel(INPUT_XLSX)
    df = df.sample(n=N_SENTENCES, random_state=42).reset_index(drop=True)  # fixed seed for reproducibility
    return df


def load_model():
    print(f"Loading model: {MODEL_NAME} (this may take a few minutes on first run) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tokenizer, model


def translate_batch(tokenizer, model, sentences, batch_size=8):
    translations = []
    tokenizer.src_lang = SRC_LANG
    tgt_lang_id = tokenizer.convert_tokens_to_ids(TGT_LANG)

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        generated = model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_id,
            max_length=128,
        )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        translations.extend(decoded)
        print(f"Translated {min(i + batch_size, len(sentences))}/{len(sentences)} sentences...")

    return translations


def compute_scores(hypotheses, references):
    # sacrebleu expects references as list of lists (multiple refs supported; we have 1 ref per sentence)
    refs = [references]

    bleu = sacrebleu.corpus_bleu(hypotheses, refs)
    chrf = sacrebleu.corpus_chrf(hypotheses, refs)
    ter = sacrebleu.corpus_ter(hypotheses, refs)

    return bleu, chrf, ter


def main():
    df = load_sentences()
    english_sentences = df["English Sentences"].tolist()
    reference_hindi = df["Hindi Sentences"].tolist()

    tokenizer, model = load_model()
    translations = translate_batch(tokenizer, model, english_sentences)

    # Save Excel output: Column A = English, Column B = Model-generated Hindi translation
    out_df = pd.DataFrame({
        "Original English Sentence": english_sentences,
        "Model-Generated Hindi Translation": translations,
    })
    out_df.to_excel(OUTPUT_XLSX, index=False, engine="openpyxl")
    print(f"Saved translations to {OUTPUT_XLSX}")

    # Compute BLEU / chrF / TER against reference Hindi sentences
    bleu, chrf, ter = compute_scores(translations, reference_hindi)

    with open(SCORES_TXT, "w", encoding="utf-8") as f:
        f.write("Assessment 2: Translation Evaluation Scores\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Number of sentences evaluated: {len(translations)}\n\n")
        f.write(f"BLEU score : {bleu.score:.2f}\n")
        f.write(f"chrF score : {chrf.score:.2f}\n")
        f.write(f"TER score  : {ter.score:.2f}\n")

    print(f"\nBLEU: {bleu.score:.2f} | chrF: {chrf.score:.2f} | TER: {ter.score:.2f}")
    print(f"Saved scores to {SCORES_TXT}")


if __name__ == "__main__":
    main()