import argparse
import os
import glob
import datetime
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    AutoTokenizer
)

# Import functions from our previous files
from data_loader import load_prepare_and_augment_data
from transformer_utils import tokenize_and_align_labels

def compute_metrics(eval_preds): 
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    true_labels = labels[labels != -100]
    true_predictions = predictions[labels != -100]

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, true_predictions, average='binary', zero_division=0
    )
    acc = accuracy_score(true_labels, true_predictions)

    return {
        'accuracy': acc,
        'f1': f1,            
        'precision': precision,
        'recall': recall
    }

def main():
    parser = argparse.ArgumentParser(description="Training Sentence Splitter BERT")
    parser.add_argument("--lang", type=str, choices=['italian', 'english'], required=True, help="Language to train (italian or english)")
    args = parser.parse_args()

    # =========================================================
    # SHARED HYPERPARAMETERS CONFIGURATION
    # =========================================================
    NUM_EPOCHS = 10
    LEARNING_RATE = 3e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0
    BATCH_SIZE = 16
    # =========================================================

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(SCRIPT_DIR)
    DATA_DIR = os.path.join(BASE_DIR, 'data')

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_DIR = os.path.join(BASE_DIR, 'runs', f'run_{args.lang}_{timestamp}')
    os.makedirs(RUN_DIR, exist_ok=True)

    CHECKPOINT_DIR = os.path.join(RUN_DIR, 'checkpoints')
    FINAL_MODEL_DIR = os.path.join(RUN_DIR, 'final_model')

    # ---------------------------------------------------------
    # LANGUAGE-SPECIFIC SETTINGS
    # ---------------------------------------------------------
    if args.lang == 'italian':
        model_name = "dbmdz/bert-base-italian-xxl-cased"
        datasets = [
            "UD_ITALIAN-ISDT", 
            "UD_ITALIAN-MARKIT", 
            "UD_ITALIAN-PARTUT",
            "UD_ITALIAN-VIT"  
        ]
        TRAIN_DIRTY_PROB = 0.15 
        
    elif args.lang == 'english':
        model_name = "bert-base-cased"
        datasets = [
            "UD_English-EWT", 
            "UD_English-GUM", 
            "UD_English-ParTUT",
            "UD_English-PUD"
        ]
        TRAIN_DIRTY_PROB = 0.2

    print(f"🌍 Starting Training for language: {args.lang.upper()}")
    print(f"🤖 Base model: {model_name}")
    print(f"🌪️  Dirty Probability set to: {TRAIN_DIRTY_PROB}")
    print(f"📁 Run directory: {RUN_DIR}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loading and preparing data...")

    train_words = []
    train_labels = []
    dev_words = []
    dev_labels = []

    train_files = []
    dev_files = []
    
    for ds_name in datasets:
        ds_path = os.path.join(DATA_DIR, ds_name)
        train_files.extend(glob.glob(os.path.join(ds_path, "*-train.sent_split")))
        dev_files.extend(glob.glob(os.path.join(ds_path, "*-dev.sent_split")))
    
    for file_path in train_files:
        print(f"  -> Train: Reading {os.path.basename(file_path)}...")
        w, l = load_prepare_and_augment_data(file_path, dirty_prob=TRAIN_DIRTY_PROB)
        train_words.extend(w)
        train_labels.extend(l)

    for file_path in dev_files:
        print(f"  -> Dev: Reading {os.path.basename(file_path)}...")
        w, l = load_prepare_and_augment_data(file_path, dirty_prob=0.0)
        dev_words.extend(w)
        dev_labels.extend(l)

    print(f"\n📊 Total training chunks: {len(train_words)}")
    print(f"📊 Total dev chunks: {len(dev_words)}")        

    print("Tokenization and alignment...")
    train_dataset = tokenize_and_align_labels(train_words, train_labels, model_name)
    dev_dataset = tokenize_and_align_labels(dev_words, dev_labels, model_name)

    print("Initializing the Model...")
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=2, 
        id2label={0: "O", 1: "EOS"},
        label2id={"O": 0, "EOS": 1}
    )
    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        eval_strategy="epoch",    
        save_strategy="epoch",         
        learning_rate=LEARNING_RATE,             
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,             
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        load_best_model_at_end=True,    
        metric_for_best_model="f1",   
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("🚀 Training in progress...")
    trainer.train()
    
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR) 

    print("Evaluating on Dev Set...")
    eval_results = trainer.evaluate()
    
    # ---------------------------------------------------------
    # METRICS AND CONFIGURATION SAVING
    # ---------------------------------------------------------
    metrics_path = os.path.join(RUN_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(eval_results, f, indent=4)

    config_data = {
        "lang": args.lang,
        "base_model": model_name,
        "datasets": datasets,
        "hyperparameters": {
            "dirty_prob_train": TRAIN_DIRTY_PROB,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "batch_size": BATCH_SIZE
        }
    }
    
    config_path = os.path.join(RUN_DIR, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=4)

    print(f"✅ Training completed. Model, metrics, and configuration saved in:\n{RUN_DIR}")

if __name__ == "__main__":
    main()