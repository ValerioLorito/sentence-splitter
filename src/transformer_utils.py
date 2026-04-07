# src/transformer_utils.py
import torch
from transformers import AutoTokenizer

class SentenceSplitDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        # Receives data already processed by the tokenizer
        self.encodings = encodings

    def __getitem__(self, idx):
        # Returns a single chunk ready for the model (as tensors)
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

def tokenize_and_align_labels(tokens_chunks, labels_chunks, model_name="dbmdz/bert-base-italian-xxl-cased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    tokenized_inputs = tokenizer(
        tokens_chunks,
        is_split_into_words=True,
        truncation=True,
        padding='max_length',
        max_length=256,
        return_tensors="pt"
    )

    aligned_labels = []
    for i, label_chunk in enumerate(labels_chunks):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100) # Ignore special tokens like [CLS] and [SEP]
            else:
                label_ids.append(label_chunk[word_idx])
                
        aligned_labels.append(label_ids)

    tokenized_inputs["labels"] = aligned_labels
    
    # Returns the Dataset object ready for training
    return SentenceSplitDataset(tokenized_inputs)