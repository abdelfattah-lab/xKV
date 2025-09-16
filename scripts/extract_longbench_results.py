#!/usr/bin/env python3
"""
Elegant script to extract and pivot experiment results from log files.
Order: NarrativeQA, Qasper, MultifieldQA, HotpotQA, 2WikiMQA, Musique,
       Gov Report, QMSum, MultiNews, TriviaQA, Samsum, PassageRetrieval,
       LCC, RepoBench
"""

import re
import pandas as pd
import argparse
import glob
import os

# Configuration
# LongBench datasets to display and their desired column order
DATASET_ORDER = [
    'narrativeqa',
    'qasper',
    'multifieldqa_en',
    'hotpotqa',
    '2wikimqa',
    'musique',
    'gov_report',
    'qmsum',
    'multi_news',
    'triviaqa',
    'samsum',
    'passage_retrieval_en',
    'lcc',
    'repobench-p',
]

DATASET_NAMES = {
    'narrativeqa': 'NarrativeQA',
    'qasper': 'Qasper',
    'multifieldqa_en': 'MultifieldQA',
    'hotpotqa': 'HotpotQA',
    '2wikimqa': '2WikiMQA',
    'musique': 'Musique',
    'gov_report': 'Gov Report',
    'qmsum': 'QMSum',
    'multi_news': 'MultiNews',
    'triviaqa': 'TriviaQA',
    'samsum': 'Samsum',
    'passage_retrieval_en': 'PassageRetrieval',
    'lcc': 'LCC',
    'repobench-p': 'RepoBench',
}

# Experiment ordering
EXPERIMENT_ORDER = ['full', 'MiniCache', 'xKV', 'StreamingLLM', 'SnapKV', 'PyramidKV', 'KIVI', 'Quest']
EXPERIMENT_PATTERNS = {
    'Enabled xKV: True': 'xKV',
    'streamingllm': 'StreamingLLM',
    'snapkv': 'SnapKV',
    'minicache': 'MiniCache',
    'pyramidkv': 'PyramidKV', 
    'kivi': 'KIVI',
    'quest': 'Quest'
}

# Sort xKV variants by effective k (k divided by the xKV factor), then by factor, k, v
def sort_xkv_variants(labels):
    patt_with_factor = re.compile(r'^xKV_(\d+)_k(\d+)_v(\d+)$', re.IGNORECASE)
    items = []
    for lbl in labels:
        m = patt_with_factor.match(lbl)
        if not m:
            continue
        factor = int(m.group(1))
        k = int(m.group(2))
        v = int(m.group(3))
        eff_k = k / max(1, factor)
        items.append((eff_k, factor, k, v, lbl))
    # Sort by effective k DESC, then factor ASC, then k, v, label
    items.sort(key=lambda x: (-x[0], x[1], x[2], x[3], x[4]))
    return [it[-1] for it in items]

def extract_tables(log_file: str) -> list:
    """Extract all result tables from log file."""
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    return re.findall(r'\| model.*?\| mean.*?\|.*?\n', content, re.DOTALL)

def parse_table(table_text: str) -> pd.DataFrame:
    """Parse table text to DataFrame."""
    lines = [line for line in table_text.strip().split('\n') 
             if line.strip() and not line.strip().startswith('|:')]
    
    rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.split('|')[1:-1]]
        rows.append(cells)
    
    if rows:
        df = pd.DataFrame(rows[1:], columns=rows[0])
        return df[df['model'] != 'mean']  # Filter out mean rows
    return pd.DataFrame()


def identify_experiment(log_content: str, table_index: int) -> str:
    """Identify experiment type from log context."""
    lines = log_content.split('\n')
    table_count = 0
    
    for i, line in enumerate(lines):
        if '| model' in line and 'dataset' in line and 'baseline' in line:
            if table_count == table_index:
                # Search backwards for experiment markers, prioritizing more recent ones
                experiment_found = None
                
                # Look for experiment markers in the preceding lines
                for j in range(i-1, max(0, i-100), -1):  # Extended search range
                    current_line = lines[j].lower()
                    
                    # Check for xKV enabled status first (but continue checking for other methods if disabled)
                    if 'enabled xkv: true' in current_line:
                        # Prefer to find factor + k,v from nearby file naming
                        factor_match = None
                        for k in range(max(0, j-50), min(len(lines), j+50)):
                            nm = re.search(r'xkv[_-](\d+)_k(\d+)_v(\d+)', lines[k].lower())
                            if nm:
                                num, kk, vv = nm.groups()
                                experiment_found = f'xKV_{num}_k{kk}_v{vv}'
                                factor_match = True
                                break
                        if not factor_match:
                            # Fall back to plain 'xKV' to avoid no-factor variant labels
                            experiment_found = 'xKV'
                        break
                    # Note: Don't break on 'enabled xkv: false' as it might be another method
                    
                    # Check for other experiment patterns
                    for pattern, name in EXPERIMENT_PATTERNS.items():
                        if pattern.lower() in current_line:
                            experiment_found = name
                            break
                    if experiment_found:
                        break

                    # Check for method names in file paths or configurations
                    if 'streamingllm' in current_line:
                        experiment_found = 'StreamingLLM'
                        break
                    elif 'snapkv' in current_line:
                        experiment_found = 'SnapKV'
                        break
                    elif 'minicache' in current_line:
                        experiment_found = 'MiniCache'
                        break
                    elif 'pyramidkv' in current_line:
                        experiment_found = 'PyramidKV'
                        break
                    elif 'kivi' in current_line:
                        experiment_found = 'KIVI'
                        break
                    elif 'quest' in current_line:
                        experiment_found = 'Quest'
                        break
                    
                    # Check for xKV file naming patterns (support hyphen or underscore)
                    xkv_match = re.search(r'xkv[_-](\d+)_k(\d+)_v(\d+)', current_line)
                    if xkv_match:
                        num, k, v = xkv_match.groups()
                        experiment_found = f'xKV_{num}_k{k}_v{v}'
                        break
                
                # If no specific method found but xKV is disabled, check if it's truly baseline
                if not experiment_found:
                    # Look for any explicit xKV disabled flag in the context
                    for j in range(max(0, i-100), min(len(lines), i+1)):
                        if 'enabled xkv: false' in lines[j].lower():
                            # Double check that no other method is enabled
                            has_other_method = False
                            for k in range(max(0, j-20), min(len(lines), j+20)):
                                check_line = lines[k].lower()
                                if any(method in check_line for method in ['snapkv', 'streamingllm', 'pyramidkv', 'minicache', 'kivi', 'quest']):
                                    if ('enabled' in check_line and 'true' in check_line) or 'kv_type' in check_line:
                                        has_other_method = True
                                        break
                            if not has_other_method:
                                experiment_found = 'Baseline'
                            break
                
                return experiment_found or f'Experiment_{table_index + 1}'
            table_count += 1
    return f'Experiment_{table_index + 1}'

def format_dataset(name: str) -> str:
    """Format dataset name for display."""
    for key, display_name in DATASET_NAMES.items():
        if key in name.lower():
            return display_name
    return name

def get_dataset_sort_key(dataset: str) -> int:
    """Get sort key for dataset ordering."""
    for i, key in enumerate(DATASET_ORDER):
        if key in dataset.lower():
            return i
    return 999

def get_experiment_sort_key(experiment: str) -> int:
    """Get sort key for experiment ordering."""
    # Handle xKV variants (xKV_1_k96_v144, etc.)
    base_experiment = experiment
    if experiment.startswith('xKV'):
        base_experiment = 'xKV'
    lowered = base_experiment.lower()
    for i, exp in enumerate(EXPERIMENT_ORDER):
        if exp.lower() in lowered:
            return i
    
    # If not found in predefined order, put at end
    return 999


def main():
    parser = argparse.ArgumentParser(description='Extract LongBench experiment results from log files')
    parser.add_argument('dir', default='logs/', help='Log file directory (default: logs/)')
    args = parser.parse_args()
    
    log_files = glob.glob(os.path.join(args.dir, '*.log'))
    if not log_files:
        print(f"No log files found in {args.dir}")
        return
        
    print(f"Found {len(log_files)} log files: {log_files}")
    
    # Collect all data from all log files
    all_combined_data = []
    
    for log_file in log_files:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                log_content = f.read()
            
            tables = extract_tables(log_file)
            print(f"Processing {log_file}: Found {len(tables)} result tables")
            
            # Process all tables from this file
            for idx, table_text in enumerate(tables):
                df = parse_table(table_text)
                if not df.empty:
                    base_name = os.path.basename(log_file)
                    df['experiment'] = base_name[:-4]  # Remove .log extension
                    # df['experiment'] = identify_experiment(log_content, idx)
                    df['dataset_formatted'] = df['dataset'].apply(format_dataset)
                    df['sort_key'] = df['dataset'].apply(get_dataset_sort_key)
                    df['experiment_sort_key'] = df['experiment'].apply(get_experiment_sort_key)
                    df['source_file'] = log_file
                    all_combined_data.append(df)
        except Exception as e:
            print(f"Error processing {log_file}: {e}")
    
    if not all_combined_data:
        print("No valid tables found in any log files")
        return
    
    # Combine all data
    combined = pd.concat(all_combined_data, ignore_index=True)
    combined['baseline'] = pd.to_numeric(combined['baseline'], errors='coerce')
    # Convert to percentage (multiply by 100)
    combined['baseline'] = combined['baseline'] * 100
    combined = combined.sort_values(['experiment_sort_key', 'sort_key'])
    
    # Check for and report duplicates
    duplicates = combined.duplicated(subset=['experiment', 'dataset_formatted'], keep=False)
    if duplicates.any():
        print(f"\n📋 Found {duplicates.sum()} duplicate experiment-dataset combinations:")
        dup_data = combined[duplicates].sort_values(['experiment', 'dataset_formatted', 'baseline'])
        
        # Group by experiment and dataset to check if values are the same
        has_different_results = False
        for (exp, dataset), group in dup_data.groupby(['experiment', 'dataset_formatted']):
            unique_scores = group['baseline'].unique()
            if len(unique_scores) != 1:
                print(f"⚠️  {exp} - {dataset}: {len(group)} DIFFERENT results: {unique_scores}")
                has_different_results = True
                # Show source files for debugging
                source_files = group['source_file'].unique()
                print(f"   Source files: {', '.join(source_files)}")
        
        if has_different_results:
            print("⚠️  WARNING: Some experiments have different results for the same dataset!")
        print()
    
    # Handle duplicates by taking the last occurrence (most recent)
    combined = combined.drop_duplicates(subset=['experiment', 'dataset_formatted'], keep='last')
    
    # Create output DataFrame
    output_df = combined[['experiment', 'dataset_formatted', 'baseline']].copy()
    output_df.columns = ['Experiment', 'Dataset', 'Score']
    
    # Create and display table
    table = output_df.pivot(index='Experiment', columns='Dataset', values='Score')
    table = table.reindex(columns=[DATASET_NAMES[k] for k in DATASET_ORDER if DATASET_NAMES[k] in table.columns])
    
    # Sort experiments by predefined order and then by detected experiment_sort_key
    experiment_order: list[str] = []
    # Always collect xKV variants present in the table (labels like xKV_1_k96_v144)
    xkv_variants_present = [idx for idx in table.index if idx.lower().startswith('xkv') and idx != 'xKV']
    xkv_variants_sorted = sort_xkv_variants(xkv_variants_present)

    for exp in EXPERIMENT_ORDER:
        # Add the base method row if present
        if exp in table.index:
            experiment_order.append(exp)
        # Insert xKV variants at the xKV slot regardless of whether a plain 'xKV' row exists
        if exp == 'xKV' and xkv_variants_sorted:
            experiment_order.extend([v for v in xkv_variants_sorted if v not in experiment_order])

    # Build desired order from the combined dataframe we already computed
    remaining = [idx for idx in table.index if idx not in experiment_order]
    if remaining:
        # Map experiment to its min sort key
        exp_order_map = (
            combined[['experiment', 'experiment_sort_key']]
            .drop_duplicates()
            .groupby('experiment')['experiment_sort_key']
            .min()
            .to_dict()
        )
        remaining_sorted = sorted(remaining, key=lambda e: (exp_order_map.get(e, 999), e))
        experiment_order.extend(remaining_sorted)

    table = table.reindex(index=experiment_order)
    
    # Remove rows that are completely NaN (no data for any dataset)
    table = table.dropna(how='all')
    
    print(f"\n=== LongBench Results Table (%) ===")
    print(table.to_string(float_format='%.2f'))
    
    table_file = "longbench_results.csv"
    # Save CSV rounded to 3 decimals
    table_rounded = table.round(3)
    table_rounded.to_csv(table_file, float_format='%.3f')
    print(f"Table saved to {table_file}")

if __name__ == "__main__":
    main()
