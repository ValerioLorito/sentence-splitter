import glob
import os
import re
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def analyze_eos_dynamically(file_path):
    if not os.path.exists(file_path):
        print(f"❌ ERROR: File not found in -> {file_path}")
        return None

    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    total_eos = raw_text.count("<EOS>")
    
    tokens = re.findall(r'<EOS>|\w+|\.+|[^\w\s]', raw_text)
    stats = defaultdict(lambda: {'eos': 0, 'no_eos': 0})
    
    for i in range(len(tokens)):
        tok = tokens[i]
        
        if tok == '<EOS>':
            continue
        
        is_eos = False
        if i + 1 < len(tokens) and tokens[i+1] == '<EOS>':
            is_eos = True
            
        if tok.isalnum() or tok == '_':
            target_key = '[WORD]'
        else:
            target_key = tok
            
        if is_eos:
            stats[target_key]['eos'] += 1
        else:
            stats[target_key]['no_eos'] += 1

    return dict(stats), total_eos

def format_table_for_txt(stats, total_eos, dataset_name, valid_targets):
    """Generates a nicely formatted Markdown-style table as a string."""
    lines = []
    lines.append(f"\n{'='*75}")
    lines.append(f"📊 DATASET: {dataset_name.upper()} | Total <EOS>: {total_eos}")
    lines.append(f"{'='*75}")
    lines.append(f"| {'Token':<12} | {'With <EOS>':<12} | {'Without <EOS>':<13} | {'% with <EOS>':<12} |")
    lines.append("-" * 62)
    
    for punct in valid_targets:
        data = stats.get(punct, {'eos': 0, 'no_eos': 0})
        eos_count = data['eos']
        no_eos_count = data['no_eos']
        total = eos_count + no_eos_count
        
        if total == 0:
            continue
        
        perc = (eos_count / total * 100) if total > 0 else 0.0
        lines.append(f"| {punct:<12} | {eos_count:<12} | {no_eos_count:<13} | {perc:>5.1f}%      |")
    
    lines.append("-" * 62 + "\n")
    return "\n".join(lines)

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
    
    ANALYSIS_DIR = os.path.join(SCRIPT_DIR, 'dataset_analysis')
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

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
                    all_results[dataset_name][split_name] = {"stats": stats, "total_eos": total_eos}
                    
                    for key, counts in stats.items():
                        if counts['eos'] > 0:
                            global_valid_targets.add(key)

    sorted_targets = sorted(list(global_valid_targets))
    if '[WORD]' in sorted_targets:
        sorted_targets.remove('[WORD]')
        sorted_targets.insert(0, '[WORD]')

    # ---------------------------------------------------------
    # 1. TEXT REPORT GENERATION (Human Readable)
    # ---------------------------------------------------------
    report_content = []
    report_content.append("# SENTENCE SPLITTER DATASET METRICS REPORT\n")
    
    for dataset_name in sorted(all_results.keys()):
        for split_name in ["Train", "Dev", "Test"]:
            if split_name in all_results[dataset_name]:
                stats = all_results[dataset_name][split_name]["stats"]
                total_eos = all_results[dataset_name][split_name]["total_eos"]
                
                # Print to terminal
                table_str = format_table_for_txt(stats, total_eos, f"{dataset_name} - {split_name}", sorted_targets)
                print(table_str)
                # Save for txt file
                report_content.append(table_str)

    txt_path = os.path.join(ANALYSIS_DIR, 'dataset_metrics_report.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_content))
    print(f"\n📄 Readable Text Report saved in: {txt_path}")

    # Keep JSON for code persistence
    json_path = os.path.join(ANALYSIS_DIR, 'dataset_metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)

    # ---------------------------------------------------------
    # 2. GENERATE GROUPED VISUALIZATION CHART (Data Augmentation Logic)
    # ---------------------------------------------------------
    print(f"🖼️  Generating Grouped <EOS> distribution Bar Chart...")
    
    standard_punct = {'.', '!', '?', ':', ';'}
    
    dataset_totals = {}
    grouped_counts = defaultdict(lambda: {'Standard Punctuation': 0, 'Weird Endings': 0, '[WORD]': 0})
    
    for ds_name, splits in all_results.items():
        total_ds_eos = 0
        for split_name, data in splits.items():
            total_ds_eos += data["total_eos"]
            for target, counts in data["stats"].items():
                if counts['eos'] > 0:
                    if target == '[WORD]':
                        grouped_counts[ds_name]['[WORD]'] += counts['eos']
                    elif target in standard_punct:
                        grouped_counts[ds_name]['Standard Punctuation'] += counts['eos']
                    else:
                        grouped_counts[ds_name]['Weird Endings'] += counts['eos']
                        
        dataset_totals[ds_name] = total_ds_eos

    datasets_list = sorted(list(dataset_totals.keys()))
    groups = ['Standard Punctuation', 'Weird Endings', '[WORD]']
    x = np.arange(len(datasets_list))
    width = 0.25 

    fig_bar, ax_bar = plt.subplots(figsize=(14, 7))
    colors = ['#4a90e2', '#f5a623', '#d0021b'] 
    
    for i, group in enumerate(groups):
        percentages = []
        for ds in datasets_list:
            if dataset_totals[ds] > 0:
                perc = (grouped_counts[ds][group] / dataset_totals[ds]) * 100
            else:
                perc = 0
            percentages.append(perc)
        
        offset = (i - 1) * width 
        ax_bar.bar(x + offset, percentages, width, label=group, color=colors[i])

    ax_bar.set_ylabel('% of total <EOS> boundaries', fontsize=12)
    ax_bar.set_title('Distribution of <EOS> Triggers', fontsize=15, pad=15)
    ax_bar.set_xticks(x)
    
    clean_names = [ds.replace('UD_', '').replace('Italian-', 'IT_').replace('English-', 'EN_') for ds in datasets_list]
    ax_bar.set_xticklabels(clean_names, fontsize=11, rotation=45, ha="right")
    
    ax_bar.legend(title="Token Category", loc='upper right')
    ax_bar.grid(axis='y', linestyle='--', alpha=0.7)

    fig_bar.tight_layout()
    bar_img_path = os.path.join(ANALYSIS_DIR, 'eos_grouped_distribution.png')
    fig_bar.savefig(bar_img_path, dpi=300, bbox_inches='tight')
    
    print(f"✅ Grouped Distribution Chart successfully saved in: {bar_img_path}")

if __name__ == "__main__":
    main()