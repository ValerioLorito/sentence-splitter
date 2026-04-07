import argparse
import os
import glob
import re
import json
import torch
import nltk
import spacy
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from data_loader import load_prepare_and_augment_data

def get_clean_len(text):
    """Returns the number of characters excluding all spaces and newlines."""
    return len(re.sub(r'\s+', '', text))

def get_metrics_dict(true_labels, preds):
    """Calculates metrics and returns them as a dictionary for JSON dumping."""
    p, r, f1, _ = precision_recall_fscore_support(true_labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(true_labels, preds)
    return {
        "f1_score": round(f1, 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "accuracy": round(acc, 4)
    }

def print_formatted_metrics(metrics_dict, model_name):
    """Prints metrics in a formatted string."""
    print(f"  [{model_name:<16}] -> F1: {metrics_dict['f1_score']:.4f} | Prec: {metrics_dict['precision']:.4f} | Rec: {metrics_dict['recall']:.4f} | Acc: {metrics_dict['accuracy']:.4f}")

def evaluate_all(file_path, model, tokenizer, device, lang, spacy_nlp):
    # Read the original raw text to give NLTK and spaCy a fair, perfectly formatted input
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    clean_text_for_baselines = raw_text.replace("<EOS>", "")

    # 1. Base loading from the data loader
    try:
        chunks_words, chunks_labels = load_prepare_and_augment_data(file_path, dirty_prob=0.0)
    except TypeError:
        chunks_words, chunks_labels = load_prepare_and_augment_data(file_path, max_chunk_size=150, dirty_prob=0.0)
    
    if not chunks_words:
        return None, None, None, None
        
    all_true = []
    all_words = []
    bert_preds = []
    
    # 2. Iterate dynamically on the sentence-aware chunks for BERT
    for words, labels in zip(chunks_words, chunks_labels):
        encoding = tokenizer(
            words, 
            is_split_into_words=True, 
            return_tensors="pt", 
            truncation=True, 
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).squeeze().cpu().tolist()
            
        # Edge case: if a chunk is too short, predictions might be an int instead of a list
        if type(preds) == int:
            preds = [preds]
            
        word_ids = encoding.word_ids(batch_index=0)
        
        # Token to Word Alignment
        word_preds = [0] * len(words)
        for idx, w_idx in enumerate(word_ids):
            if w_idx is not None:
                if preds[idx] == 1:
                    word_preds[w_idx] = 1
                    
        all_true.extend(labels)
        bert_preds.extend(word_preds)
        all_words.extend(words)

    # 3. Dynamic NLTK Prediction using the natural raw text
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

    # 4. Dynamic spaCy Prediction using the natural raw text
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

    return all_true, bert_preds, nltk_preds, spacy_preds

def main():
    parser = argparse.ArgumentParser(description="Final Evaluation: BERT vs NLTK vs spaCy")
    parser.add_argument("--run", type=str, required=True, help="Name of the local run folder OR Hugging Face Repo ID (e.g. user/repo)")
    parser.add_argument("--lang", type=str, choices=['italian', 'english'], required=True, help="Language of the dataset")
    args = parser.parse_args()

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(SCRIPT_DIR) 
    DATA_DIR = os.path.join(BASE_DIR, 'data')

    # ---------------------------------------------------------
    # SMART MODEL PATH DETECTION (Local vs Cloud)
    # ---------------------------------------------------------
    is_hf_model = "/" in args.run

    if is_hf_model:
        print(f"☁️ Hugging Face Cloud Model detected: {args.run}")
        model_path = args.run
        # Create a safe folder name for saving results locally
        safe_repo_name = args.run.replace("/", "_")
        RUN_DIR = os.path.join(BASE_DIR, 'runs', f'eval_{safe_repo_name}')
        os.makedirs(RUN_DIR, exist_ok=True)
    else:
        print(f"💻 Local Run detected: {args.run}")
        RUN_DIR = os.path.join(BASE_DIR, 'runs', args.run)
        if not os.path.exists(RUN_DIR):
            print(f"❌ ERROR: Local run not found at: {RUN_DIR}")
            return
        model_path = os.path.join(RUN_DIR, 'final_model')

    # Create the methodologically sound test_results directory
    TEST_RESULTS_DIR = os.path.join(RUN_DIR, 'test_results')
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    # DYNAMIC LANGUAGE SETUP
    if args.lang == 'italian':
        target_datasets = ["UD_ITALIAN-ISDT", "UD_ITALIAN-MARKIT", "UD_ITALIAN-PARTUT", "UD_ITALIAN-VIT"]
        spacy_model_name = "it_core_news_sm"
    elif args.lang == 'english':
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

    print(f"🤖 Loading Tokenizer and BERT model from: {model_path}")
    print(f"🌍 Language set to: {args.lang.upper()}")
    
    # Loads from Hugging Face OR Local path dynamically!
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("\n" + "="*75)
    print(f" 🏆 ULTIMATE SHOWDOWN: BERT vs NLTK vs spaCy ON TEST SETS ({args.lang.upper()})")
    print("="*75)

    # Data structures for JSON and Plotting
    final_json_data = {
        "language": args.lang,
        "run_name": args.run,
        "datasets_metrics": {}
    }
    
    dataset_names_for_plot = []
    f1_scores_plot = {'NLTK': [], 'spaCy': [], 'BERT': []}

    for ds_name in target_datasets:
        test_file_pattern = os.path.join(DATA_DIR, ds_name, "*-test.sent_split")
        test_files = glob.glob(test_file_pattern)
        
        if not test_files:
            print(f"\n⚠️ Skipped {ds_name}: Test file not found.")
            continue
            
        test_file = test_files[0]
        
        true_lbls, bert_preds, nltk_preds, spacy_preds = evaluate_all(
            test_file, model, tokenizer, device, args.lang, spacy_nlp
        )
        
        if true_lbls is None:
            continue
            
        print(f"\n📊 {os.path.basename(test_file).upper()}")
        print("-" * 75)
        
        # Calculate metrics
        nltk_metrics = get_metrics_dict(true_lbls, nltk_preds)
        spacy_metrics = get_metrics_dict(true_lbls, spacy_preds)
        bert_metrics = get_metrics_dict(true_lbls, bert_preds)
        
        # Print metrics
        print_formatted_metrics(nltk_metrics, "NLTK (Punkt)")
        print_formatted_metrics(spacy_metrics, f"spaCy ({spacy_model_name[:2]})")
        print_formatted_metrics(bert_metrics, "BERT Model")
        
        # Store in JSON dictionary
        final_json_data["datasets_metrics"][ds_name] = {
            "NLTK": nltk_metrics,
            "spaCy": spacy_metrics,
            "BERT": bert_metrics
        }
        
        # Store for plotting
        short_ds_name = ds_name.split('-')[-1] # Extracts ISDT, VIT, EWT, etc.
        dataset_names_for_plot.append(short_ds_name)
        f1_scores_plot['NLTK'].append(nltk_metrics["f1_score"])
        f1_scores_plot['spaCy'].append(spacy_metrics["f1_score"])
        f1_scores_plot['BERT'].append(bert_metrics["f1_score"])

    # ---------------------------------------------------------
    # 1. SAVE JSON RESULTS
    # ---------------------------------------------------------
    json_path = os.path.join(TEST_RESULTS_DIR, 'test_metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_json_data, f, indent=4)
    print(f"\n📄 Test metrics successfully saved in JSON format: {json_path}")

    # ---------------------------------------------------------
    # 2. GENERATE F1-SCORE BAR CHART FOR TEST SETS
    # ---------------------------------------------------------
    if dataset_names_for_plot:
        print(f"🖼️  Generating Final Test F1-Score Bar Chart...")
        x = np.arange(len(dataset_names_for_plot))  
        width = 0.25  

        fig_bar, ax_bar = plt.subplots(figsize=(12, 6))
        
        rects1 = ax_bar.bar(x - width, f1_scores_plot['NLTK'], width, label='NLTK (Punkt)', color='#a8d0e6')
        rects2 = ax_bar.bar(x, f1_scores_plot['spaCy'], width, label=f'spaCy', color='#f8e9a1')
        rects3 = ax_bar.bar(x + width, f1_scores_plot['BERT'], width, label='BERT (Our Model)', color='#f76c6c')

        ax_bar.set_ylabel('F1-Score', fontsize=12)
        ax_bar.set_title(f'FINAL TEST EVALUATION: F1-Score Comparison ({args.lang.upper()})', fontsize=14, pad=15)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(dataset_names_for_plot, fontsize=11)
        
        ax_bar.legend(loc='lower right')
        ax_bar.set_ylim([0.0, 1.1]) 

        ax_bar.bar_label(rects1, padding=3, fmt='%.3f', fontsize=9)
        ax_bar.bar_label(rects2, padding=3, fmt='%.3f', fontsize=9)
        ax_bar.bar_label(rects3, padding=3, fmt='%.3f', fontsize=9, weight='bold')

        fig_bar.tight_layout()
        bar_img_path = os.path.join(TEST_RESULTS_DIR, 'f1_test_comparison_barchart.png')
        fig_bar.savefig(bar_img_path, dpi=300, bbox_inches='tight')
        print(f"✅ Final Test F1-Score Bar Chart saved in: {bar_img_path}")

if __name__ == "__main__":
    main()