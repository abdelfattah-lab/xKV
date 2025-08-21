#!/usr/bin/env python3
"""
Elegant script to extract and pivot RULER experiment results from log files.
Order: NIAH Single 1, NIAH Single 2, NIAH MultiKey 1, NIAH MultiKey 2, NIAH MultiQuery, NIAH MultiValue, VT, FWE, QA 1, QA 2
"""

import re
import pandas as pd
import argparse
import glob
import os

# Configuration for RULER datasets
DATASET_ORDER = ['niah_single_1', 'niah_single_2', 'niah_multikey_1', 'niah_multikey_2', 'niah_multiquery', 'niah_multivalue', 'qa_1', 'qa_2', 'vt', 'fwe']
DATASET_NAMES = {
    'niah_single_1': 'N-S1',
    'niah_single_2': 'N-S2',
    'niah_multikey_1': 'N-MK1',
    'niah_multikey_2': 'N-MK2',
    'niah_multiquery': 'N-MQ',
    'niah_multivalue': 'N-MV',
    'qa_1': 'QA-1',
    'qa_2': 'QA-2',
    'vt': 'VT',
    'fwe': 'FWE'
}

# Experiment ordering - Baseline first, then EXPERIMENT_PATTERNS order
EXPERIMENT_ORDER = ['Baseline', 'xKV', 'MiniCache', 'StreamingLLM', 'SnapKV', 'PyramidKV', 'KIVI', 'Quest']

EXPERIMENT_PATTERNS = {
    'Enabled xKV: True': 'xKV',
    'minicache': 'MiniCache',
    'streamingllm': 'StreamingLLM',
    'snapkv': 'SnapKV',
    'pyramidkv': 'PyramidKV', 
    'kivi': 'KIVI',
    'quest': 'Quest'
}

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
                        # Look for specific xKV configuration
                        for k in range(max(0, j-20), min(len(lines), j+20)):
                            config_line = lines[k]
                            k_match = re.search(r'rank_k[=:\s]+(\d+)', config_line)
                            v_match = re.search(r'rank_v[=:\s]+(\d+)', config_line)
                            if k_match and v_match:
                                k_val = k_match.group(1)
                                v_val = v_match.group(1)
                                experiment_found = f'xKV_k{k_val}_v{v_val}'
                                break
                        if not experiment_found:
                            experiment_found = 'xKV'
                        break
                    
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
                    
                    # Check for xKV file naming patterns
                    xkv_match = re.search(r'xkv-(\d+)_k(\d+)_v(\d+)', current_line)
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
    # Extract the ruler dataset name from ruler/dataset_name format
    if 'ruler/' in name:
        dataset_key = name.split('ruler/')[-1]
        return DATASET_NAMES.get(dataset_key, dataset_key)
    return name

def get_dataset_sort_key(dataset: str) -> int:
    """Get sort key for dataset ordering."""
    if 'ruler/' in dataset:
        dataset_key = dataset.split('ruler/')[-1]
        for i, key in enumerate(DATASET_ORDER):
            if key == dataset_key:
                return i
    return 999

def get_experiment_sort_key(experiment: str) -> int:
    """Get sort key for experiment ordering."""
    # Handle xKV variants (xKV_1_k96_v144, etc.)
    base_experiment = experiment
    if experiment.startswith('xKV'):
        base_experiment = 'xKV'
    
    for i, exp in enumerate(EXPERIMENT_ORDER):
        if base_experiment == exp:
            return i
    
    # If not found in predefined order, put at end
    return 999

def process_log_file(log_file: str):
    """Process single log file and extract results."""
    print(f"\n{'='*60}")
    print(f"Processing: {log_file}")
    print(f"{'='*60}")
    
    base_name = os.path.splitext(log_file)[0]
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        tables = extract_tables(log_file)
        print(f"Found {len(tables)} result tables")
        
        # Process all tables
        all_data = []
        for idx, table_text in enumerate(tables):
            df = parse_table(table_text)
            if not df.empty:
                df['experiment'] = identify_experiment(log_content, idx)
                df['dataset_formatted'] = df['dataset'].apply(format_dataset)
                df['sort_key'] = df['dataset'].apply(get_dataset_sort_key)
                df['experiment_sort_key'] = df['experiment'].apply(get_experiment_sort_key)
                all_data.append(df)
        
        if not all_data:
            print("No valid tables found")
            return
            
        # Combine and clean data
        combined = pd.concat(all_data, ignore_index=True)
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
        
        # Sort experiments by predefined order
        experiment_order = []
        for exp in EXPERIMENT_ORDER:
            if exp in table.index:
                experiment_order.append(exp)
            # Also add xKV variants right after xKV
            if exp == 'xKV':
                xkv_variants = [idx for idx in table.index if idx.startswith('xKV') and idx != 'xKV']
                experiment_order.extend(sorted(xkv_variants))
        
        # Add any remaining experiments not in base order
        for exp in sorted(table.index):
            if exp not in experiment_order:
                experiment_order.append(exp)
        table = table.reindex(index=experiment_order)
        
        # Remove rows that are completely NaN (no data for any dataset)
        table = table.dropna(how='all')
        
        print(f"\n=== RULER Results Table (%) ===")
        print(table.to_string(float_format='%.2f'))
        
        table_file = f"{base_name}.csv"
        table.to_csv(table_file)
        print(f"Table saved to {table_file}")
            
    except Exception as e:
        print(f"Error processing {log_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Extract RULER experiment results from log files')
    parser.add_argument('--pattern', default='logs/*.log', help='Log file pattern (default: logs/*.log)')
    args = parser.parse_args()
    
    log_files = glob.glob(args.pattern)
    if not log_files:
        print(f"No log files found matching: {args.pattern}")
        return
        
    print(f"Found {len(log_files)} log files: {log_files}")
    
    # Collect all data from all log files
    all_combined_data = []
    
    for log_file in log_files:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                log_content = f.read()
            
            tables = extract_tables(log_file)
            print(f"\nProcessing {log_file}: Found {len(tables)} result tables")
            
            # Process all tables from this file
            for idx, table_text in enumerate(tables):
                df = parse_table(table_text)
                if not df.empty:
                    df['experiment'] = identify_experiment(log_content, idx)
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
    
    # Sort experiments by predefined order
    experiment_order = []
    for exp in EXPERIMENT_ORDER:
        if exp in table.index:
            experiment_order.append(exp)
        # Also add xKV variants right after xKV
        if exp == 'xKV':
            xkv_variants = [idx for idx in table.index if idx.startswith('xKV') and idx != 'xKV']
            experiment_order.extend(sorted(xkv_variants))
    
    # Add any remaining experiments not in base order
    for exp in sorted(table.index):
        if exp not in experiment_order:
            experiment_order.append(exp)
    table = table.reindex(index=experiment_order)
    
    # Remove rows that are completely NaN (no data for any dataset)
    table = table.dropna(how='all')
    
    print(f"\n=== RULER Results Table (%) ===")
    print(table.to_string(float_format='%.2f'))
    
    table_file = "ruler_results.csv"
    # Save CSV without rounding (keep original precision)
    table.to_csv(table_file)
    print(f"Table saved to {table_file}")

if __name__ == "__main__":
    main()
