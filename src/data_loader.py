import re
import random
from transformers import AutoTokenizer

def load_prepare_and_augment_data(file_path, max_chunk_size=200, dirty_prob=0.05):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    sentences = raw_text.split('<EOS>')
    tokenizer_re = re.compile(r'\w+|[^\w\s]|\n')
    
    chunked_tokens = []
    chunked_labels = []
    
    current_tokens = []
    current_labels = []
    
    # 1. Standard punctuation targets
    standard_punct = {'.', '!', '?', ':', ';'}
    
    # 2. Idiosyncratic targets discovered via analyze_datasets.py
    weird_endings = ['"', '”', ']', ')', '...', '-', '—']

    for i, sentence in enumerate(sentences):
        if not sentence.strip():
            continue
            
        sent_tokens = tokenizer_re.findall(sentence)
        if not sent_tokens:
            continue
        
        # ---------------------------------------------------------
        # 1. DATA AUGMENTATION
        # ---------------------------------------------------------
        # Apply augmentation if the sentence ends with ANY known token
        if sent_tokens[-1] in standard_punct or sent_tokens[-1] in weird_endings:
            if random.random() < dirty_prob:
                action = random.random()
                
                if action < 0.50:
                    # 50% chance: Drop the token entirely (forces learning on [WORD] boundaries)
                    sent_tokens.pop()
                elif action < 0.75:
                    # 25% chance: Swap standard ending with brackets or quotes
                    sent_tokens[-1] = random.choice(['"', '”', ']', ')', "'"])
                else:
                    # 25% chance: Swap with unconventional punctuation or dashes
                    sent_tokens[-1] = random.choice(['...', '-', '—', '!', '?', ';'])
                    
                # Safeguard in case popping emptied the sentence
                if not sent_tokens: 
                    continue
        
        sent_labels = [0] * len(sent_tokens)
        
        # Set EOS label to the last token of the sentence
        if i < len(sentences) - 1:
            sent_labels[-1] = 1
            
        # ---------------------------------------------------------
        # 2. CHUNKING (Sentence-Aware)
        # ---------------------------------------------------------
        if len(current_tokens) + len(sent_tokens) > max_chunk_size:
            if current_tokens:
                chunked_tokens.append(current_tokens)
                chunked_labels.append(current_labels)
                
            current_tokens = sent_tokens
            current_labels = sent_labels
        else:
            current_tokens.extend(sent_tokens)
            current_labels.extend(sent_labels)
            
    if current_tokens:
        chunked_tokens.append(current_tokens)
        chunked_labels.append(current_labels)
        
    return chunked_tokens, chunked_labels