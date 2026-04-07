import glob
import os
import re
from collections import defaultdict

def analyze_eos_dynamically(file_path):
    if not os.path.exists(file_path):
        print(f"❌ ERROR: File not found in -> {file_path}")
        return None

    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    total_eos = raw_text.count("<EOS>")
    
    # Advanced Tokenizer: captures <EOS>, words, sequences of periods (like ...), and single punctuations
    tokens = re.findall(r'<EOS>|\w+|\.+|[^\w\s]', raw_text)
    
    stats = defaultdict(lambda: {'eos': 0, 'no_eos': 0})
    
    for i in range(len(tokens)):
        tok = tokens[i]
        
        # Skip counting the <EOS> token itself as a preceding target
        if tok == '<EOS>':
            continue
        
        # Check if the next token is an <EOS>
        is_eos = False
        if i + 1 < len(tokens) and tokens[i+1] == '<EOS>':
            is_eos = True
            
        # Group all alphanumeric tokens under a single [WORD] category
        if tok.isalnum() or tok == '_':
            target_key = '[WORD]'
        else:
            target_key = tok
            
        if is_eos:
            stats[target_key]['eos'] += 1
        else:
            stats[target_key]['no_eos'] += 1

    return stats, total_eos

def print_dynamic_stats(stats, total_eos, dataset_name, valid_targets):
    print(f"\n{'='*65}")
    print(f"📊 DATASET ANALYSIS: {dataset_name.upper()}")
    print(f"   Total <EOS> found: {total_eos}")
    print(f"{'='*65}")
    
    if stats is None:
        print("File not found!")
        return

    print(f"{'Token':<10} | {'With <EOS>':<12} | {'Without <EOS>':<13} | {'% with <EOS>':<12}")
    print("-" * 65)
    
    for punct in valid_targets:
        data = stats.get(punct, {'eos': 0, 'no_eos': 0})
        eos_count = data['eos']
        no_eos_count = data['no_eos']
        total = eos_count + no_eos_count
        
        if total == 0:
            continue
        
        perc = (eos_count / total * 100) if total > 0 else 0.0
        
        print(f"{punct:<10} | {eos_count:<12} | {no_eos_count:<13} | {perc:.1f}%")

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

    if not os.path.exists(DATA_DIR):
        print(f"❌ ERROR: Data directory not found at {DATA_DIR}")
        return

    dataset_folders = [f.path for f in os.scandir(DATA_DIR) if f.is_dir()]
    
    all_results = {}
    global_valid_targets = set()

    print("🔍 Processing datasets to extract dynamic metrics...")
    
    for folder in sorted(dataset_folders):
        dataset_name = os.path.basename(folder)
        
        train_files = glob.glob(os.path.join(folder, "*-train.sent_split"))
        dev_files = glob.glob(os.path.join(folder, "*-dev.sent_split"))
        test_files = glob.glob(os.path.join(folder, "*-test.sent_split"))

        files = {}
        if train_files: files["Train"] = train_files[0]
        if dev_files: files["Dev"] = dev_files[0]
        if test_files: files["Test"] = test_files[0]

        if not files:
            continue
            
        all_results[dataset_name] = {}

        for split_name in ["Train", "Dev", "Test"]:
            if split_name in files:
                result = analyze_eos_dynamically(files[split_name])
                if result:
                    stats, total_eos = result
                    all_results[dataset_name][split_name] = (stats, total_eos)
                    
                    for key, counts in stats.items():
                        if counts['eos'] > 0:
                            global_valid_targets.add(key)

    sorted_targets = sorted(list(global_valid_targets))
    
    if '[WORD]' in sorted_targets:
        sorted_targets.remove('[WORD]')
        sorted_targets.insert(0, '[WORD]')

    for dataset_name in sorted(all_results.keys()):
        print(f"\n{'#'*70}")
        print(f"Dataset Overviews: {dataset_name.upper()}")
        print(f"{'#'*70}")
        
        for split_name in ["Train", "Dev", "Test"]:
            if split_name in all_results[dataset_name]:
                stats, total_eos = all_results[dataset_name][split_name]
                print_dynamic_stats(stats, total_eos, f"{dataset_name} - {split_name}", sorted_targets)

if __name__ == "__main__":
    main()