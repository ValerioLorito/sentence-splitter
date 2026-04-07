import argparse
import os
import glob
import re
import torch
import nltk
import spacy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from transformers import AutoTokenizer, AutoModelForTokenClassification

from data_loader import load_prepare_and_augment_data

def get_clean_len(text):
    """Returns the number of characters excluding all spaces and newlines."""
    return len(re.sub(r'\s+', '', text))

def get_predictions_for_file(file_path, model, tokenizer, device, lang, spacy_nlp):
    """Extracts predictions from BERT, NLTK, and spaCy ensuring perfect alignment."""
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    clean_text_for_baselines = raw_text.replace("<EOS>", "")

    try:
        chunks_words, chunks_labels = load_prepare_and_augment_data(file_path, dirty_prob=0.0)
    except TypeError:
        chunks_words, chunks_labels = load_prepare_and_augment_data(file_path, max_chunk_size=150, dirty_prob=0.0)
    
    if not chunks_words:
        return None, None, None, None, None
        
    all_true = []
    all_words = []
    bert_preds = []
    
    # --- 1. BERT PREDICTIONS ---
    for words, labels in zip(chunks_words, chunks_labels):
        encoding = tokenizer(
            words, is_split_into_words=True, return_tensors="pt", truncation=True, padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).squeeze().cpu().tolist()
            
        if type(preds) == int:
            preds = [preds]
            
        word_ids = encoding.word_ids(batch_index=0)
        word_preds = [0] * len(words)
        for idx, w_idx in enumerate(word_ids):
            if w_idx is not None:
                if preds[idx] == 1:
                    word_preds[w_idx] = 1
                    
        all_true.extend(labels)
        bert_preds.extend(word_preds)
        all_words.extend(words)

    # --- 2. NLTK PREDICTIONS ---
    nltk_sentences = nltk.sent_tokenize(clean_text_for_baselines, language=lang)
    nltk_preds = [0] * len(all_words)
    current_word_idx = 0
    
    for sent in nltk_sentences:
        chars_needed = get_clean_len(sent)
        chars_consumed = 0
        last_idx = current_word_idx
        while chars_consumed < chars_needed and current_word_idx < len(all_words):
            word_chars = get_clean_len(all_words[current_word_idx])
            chars_consumed += word_chars
            if word_chars > 0:
                last_idx = current_word_idx
            current_word_idx += 1
        if last_idx < len(nltk_preds):
            nltk_preds[last_idx] = 1

    # --- 3. SPACY PREDICTIONS ---
    spacy_nlp.max_length = len(clean_text_for_baselines) + 10000 
    doc = spacy_nlp(clean_text_for_baselines)
    spacy_sentences = [sent.text for sent in doc.sents]
    
    spacy_preds = [0] * len(all_words)
    current_word_idx = 0
    
    for sent in spacy_sentences:
        chars_needed = get_clean_len(sent)
        chars_consumed = 0
        last_idx = current_word_idx
        while chars_consumed < chars_needed and current_word_idx < len(all_words):
            word_chars = get_clean_len(all_words[current_word_idx])
            chars_consumed += word_chars
            if word_chars > 0:
                last_idx = current_word_idx
            current_word_idx += 1
        if last_idx < len(spacy_preds):
            spacy_preds[last_idx] = 1

    return all_true, all_words, bert_preds, nltk_preds, spacy_preds

def extract_errors(true_lbls, preds, words, ds_name):
    fp, fn = [], []
    for j in range(len(words)):
        t_label = true_lbls[j]
        p_label = preds[j]
        
        if t_label != p_label:
            start = max(0, j - 4)
            end = min(len(words), j + 5)
            context = f"[{ds_name}] " + " ".join(words[start:j]) + f" >>> {words[j]} <<< " + " ".join(words[j+1:end])
            
            if p_label == 1 and t_label == 0:
                fp.append(context)
            elif p_label == 0 and t_label == 1:
                fn.append(context)
    return fp, fn

def main():
    parser = argparse.ArgumentParser(description="Error Analysis: BERT vs Baselines")
    parser.add_argument('--run', type=str, required=True, help='Name of the run directory')
    parser.add_argument('--lang', type=str, choices=['italian', 'english'], required=True, help='Language of the dataset')
    args = parser.parse_args()

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(SCRIPT_DIR)
    RUN_DIR = os.path.join(BASE_DIR, 'runs', args.run)
    MODEL_PATH = os.path.join(RUN_DIR, 'final_model')

    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Final model not found in {MODEL_PATH}")
        return

    DATA_DIR = os.path.join(BASE_DIR, 'data')

    print(f"🧠 Loading NLTK Punkt model...")
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    # ---------------------------------------------------------
    # DYNAMIC LANGUAGE SETUP
    # ---------------------------------------------------------
    if args.lang == 'italian':
        model_name = "dbmdz/bert-base-italian-xxl-cased"
        target_datasets = ["UD_ITALIAN-ISDT", "UD_ITALIAN-MARKIT", "UD_ITALIAN-PARTUT", "UD_ITALIAN-VIT"]
        spacy_model_name = "it_core_news_sm"
    elif args.lang == 'english':
        model_name = "bert-base-cased"
        target_datasets = ["UD_English-EWT", "UD_English-GUM", "UD_English-ParTUT", "UD_English-PUD"]
        spacy_model_name = "en_core_web_sm"

    print(f"🧠 Loading spaCy model '{spacy_model_name}'...")
    try:
        spacy_nlp = spacy.load(spacy_model_name)
    except OSError:
        print(f"\n❌ ERROR: spaCy model '{spacy_model_name}' not found.")
        print(f"Run this command to download it, then restart the script:")
        print(f"python -m spacy download {spacy_model_name}\n")
        return

    print(f"🧠 Loading tokenizer and BERT model for {args.lang.upper()}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    global_true, global_bert, global_nltk, global_spacy = [], [], [], []
    
    bert_fp, bert_fn = [], []
    nltk_fp, nltk_fn = [], []
    spacy_fp, spacy_fn = [], []

    print("\n🔍 Starting global prediction analysis on all DEV sets...")

    # ---------------------------------------------------------
    # ITERATING OVER ALL DATASETS
    # ---------------------------------------------------------
    for ds_name in target_datasets:
        dev_file_pattern = os.path.join(DATA_DIR, ds_name, "*-dev.sent_split")
        dev_files = glob.glob(dev_file_pattern)
        
        if not dev_files:
            print(f"  ⚠️ Skipped {ds_name}: Dev file not found.")
            continue
            
        dev_file = dev_files[0]
        print(f"  -> Extracting and predicting chunks from {ds_name}...")
        
        t_lbls, words, b_preds, n_preds, s_preds = get_predictions_for_file(
            dev_file, model, tokenizer, device, args.lang, spacy_nlp
        )
        
        if t_lbls is None:
            continue
            
        global_true.extend(t_lbls)
        global_bert.extend(b_preds)
        global_nltk.extend(n_preds)
        global_spacy.extend(s_preds)
        
        # Extract context errors for each model
        bfp, bfn = extract_errors(t_lbls, b_preds, words, ds_name)
        nfp, nfn = extract_errors(t_lbls, n_preds, words, ds_name)
        sfp, sfn = extract_errors(t_lbls, s_preds, words, ds_name)
        
        bert_fp.extend(bfp); bert_fn.extend(bfn)
        nltk_fp.extend(nfp); nltk_fn.extend(nfn)
        spacy_fp.extend(sfp); spacy_fn.extend(sfn)

    # ---------------------------------------------------------
    # CONFUSION MATRICES PLOT (1x3 Subplots)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    models_data = [
        ("NLTK (Punkt)", global_nltk),
        (f"spaCy ({spacy_model_name[:2]})", global_spacy),
        ("BERT Model", global_bert)
    ]

    for ax, (m_name, m_preds) in zip(axes, models_data):
        cm = confusion_matrix(global_true, m_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NON-EOS", "EOS"])
        disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
        ax.set_title(m_name, fontsize=14, pad=10)
        ax.set_xlabel("Model Prediction", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)

    plt.suptitle(f"Global Confusion Matrices - {args.lang.upper()} DEV SETS", fontsize=18, y=1.05)
    plt.tight_layout()
    
    output_img_path = os.path.join(RUN_DIR, 'confusion_matrix_comparison.png')
    plt.savefig(output_img_path, dpi=300, bbox_inches='tight')
    print(f"\n🖼️  Comparative Confusion Matrices saved in: {output_img_path}")

    # ---------------------------------------------------------
    # ERROR LOG SAVING
    # ---------------------------------------------------------
    errors_log_path = os.path.join(RUN_DIR, 'error_analysis_log.txt')
    with open(errors_log_path, 'w', encoding='utf-8') as f:
        f.write(f"=========================================================================\n")
        f.write(f" 🤖 BERT MODEL ERRORS (FP: {len(bert_fp)} | FN: {len(bert_fn)})\n")
        f.write(f"=========================================================================\n")
        f.write(f"\n🔴 FALSE POSITIVES:\n")
        for text in bert_fp: f.write(f"- {text}\n")
        f.write(f"\n🟡 FALSE NEGATIVES:\n")
        for text in bert_fn: f.write(f"- {text}\n")
        f.write("\n\n")
        
        f.write(f"=========================================================================\n")
        f.write(f" 📚 NLTK ERRORS (FP: {len(nltk_fp)} | FN: {len(nltk_fn)})\n")
        f.write(f"=========================================================================\n")
        f.write(f"\n🔴 FALSE POSITIVES:\n")
        for text in nltk_fp: f.write(f"- {text}\n")
        f.write(f"\n🟡 FALSE NEGATIVES:\n")
        for text in nltk_fn: f.write(f"- {text}\n")
        f.write("\n\n")

        f.write(f"=========================================================================\n")
        f.write(f" 🌐 SPACY ERRORS (FP: {len(spacy_fp)} | FN: {len(spacy_fn)})\n")
        f.write(f"=========================================================================\n")
        f.write(f"\n🔴 FALSE POSITIVES:\n")
        for text in spacy_fp: f.write(f"- {text}\n")
        f.write(f"\n🟡 FALSE NEGATIVES:\n")
        for text in spacy_fn: f.write(f"- {text}\n")

    print(f"📄 Comparative Error analysis log saved in: {errors_log_path}")
    print(f"📊 Total tokens accurately analyzed: {len(global_true)}")

if __name__ == "__main__":
    main()