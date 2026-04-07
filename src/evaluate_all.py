import argparse
import os
import glob
import re
import torch
import nltk
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from data_loader import load_prepare_and_augment_data

def get_clean_len(text):
    """Returns the number of characters excluding all spaces and newlines."""
    return len(re.sub(r'\s+', '', text))

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

def print_metrics(true_labels, preds, model_name):
    p, r, f1, _ = precision_recall_fscore_support(true_labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(true_labels, preds)
    print(f"  [{model_name}] -> F1: {f1:.4f} | Prec: {p:.4f} | Rec: {r:.4f} | Acc: {acc:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Final Evaluation: BERT vs NLTK vs spaCy")
    parser.add_argument("--run", type=str, required=True, help="Name of the run to evaluate")
    parser.add_argument("--lang", type=str, choices=['italian', 'english'], required=True, help="Language of the dataset")
    args = parser.parse_args()

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(SCRIPT_DIR) 
    RUN_DIR = os.path.join(BASE_DIR, 'runs', args.run)
    DATA_DIR = os.path.join(BASE_DIR, 'data')

    if not os.path.exists(RUN_DIR):
        print(f"❌ ERROR: Run not found at: {RUN_DIR}")
        return

    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    # DYNAMIC LANGUAGE SETUP
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

    model_path = os.path.join(RUN_DIR, 'final_model')
    print(f"🤖 Loading BERT model from: {args.run}")
    print(f"🌍 Language set to: {args.lang.upper()}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("\n" + "="*75)
    print(f" 🏆 ULTIMATE SHOWDOWN: BERT vs NLTK vs spaCy ON TEST SETS ({args.lang.upper()})")
    print("="*75)

    for ds_name in target_datasets:
        test_file_pattern = os.path.join(DATA_DIR, ds_name, "*-test.sent_split")
        test_files = glob.glob(test_file_pattern)
        
        if not test_files:
            print(f"\n⚠️ Skipped {ds_name}: Test file not found.")
            continue
            
        # We iterate over all found test files for safety
        for test_file in test_files:
            true_lbls, bert_preds, nltk_preds, spacy_preds = evaluate_all(
                test_file, model, tokenizer, device, args.lang, spacy_nlp
            )
            
            if true_lbls is None:
                continue
                
            print(f"\n📊 {os.path.basename(test_file).upper()}")
            print("-" * 75)
            print_metrics(true_lbls, nltk_preds, "NLTK (Punkt) ")
            print_metrics(true_lbls, spacy_preds, f"spaCy ({spacy_model_name[:2]})")
            print_metrics(true_lbls, bert_preds, "BERT Model   ")

if __name__ == "__main__":
    main()