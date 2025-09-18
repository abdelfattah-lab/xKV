#!/usr/bin/env python3
"""
Unified script to extract and pivot experiment results from log files.

Autodetects RULER vs LongBench from parsed tables and writes one CSV per
benchmark present (ruler_results.csv and/or longbench_results.csv).

Usage:
  python scripts/extract_results.py logs/

Outputs:
  ruler_results.csv and/or longbench_results.csv with scores rounded to 3 decimals.
"""

import re
import os
import glob
import argparse
import pandas as pd
from typing import Dict, List, Tuple

# Common experiment ordering
EXPERIMENT_ORDER = ['full', 'MiniCache', 'xKV', 'StreamingLLM', 'SnapKV', 'PyramidKV', 'KIVI', 'Quest']
BUDGETED_METHODS = ['StreamingLLM', 'SnapKV', 'PyramidKV']

# RULER
RULER_DATASET_ORDER: List[str] = [
    'niah_single_1', 'niah_single_2', 'niah_single_3',
    'niah_multikey_1', 'niah_multikey_2', 'niah_multiquery', 'niah_multivalue',
    'qa_1', 'qa_2', 'vt', 'fwe',
]
RULER_DATASET_NAMES: Dict[str, str] = {
    'niah_single_1': 'N-S1', 'niah_single_2': 'N-S2', 'niah_single_3': 'N-S3',
    'niah_multikey_1': 'N-MK1', 'niah_multikey_2': 'N-MK2', 'niah_multiquery': 'N-MQ',
    'niah_multivalue': 'N-MV', 'qa_1': 'QA-1', 'qa_2': 'QA-2', 'vt': 'VT', 'fwe': 'FWE',
}

# LongBench
LONGBENCH_DATASET_ORDER: List[str] = [
    'narrativeqa', 'qasper', 'multifieldqa_en', 'hotpotqa', '2wikimqa', 'musique',
    'gov_report', 'qmsum', 'multi_news', 'trec', 'triviaqa', 'samsum',
    'passage_count', 'passage_retrieval_en', 'lcc', 'repobench-p',
]
LONGBENCH_DATASET_NAMES: Dict[str, str] = {
    'narrativeqa': 'NarrativeQA', 'qasper': 'Qasper', 'multifieldqa_en': 'MultifieldQA',
    'hotpotqa': 'HotpotQA', '2wikimqa': '2WikiMQA', 'musique': 'Musique',
    'gov_report': 'Gov Report', 'qmsum': 'QMSum', 'multi_news': 'MultiNews',
    'trec': 'TREC', 'triviaqa': 'TriviaQA', 'samsum': 'Samsum',
    'passage_count': 'PassageCount', 'passage_retrieval_en': 'PassageRetrieval',
    'lcc': 'LCC', 'repobench-p': 'RepoBench',
}

def find_dataset_key_by_substring(name: str, keys: List[str]) -> str | None:
    lower = name.lower()
    for key in keys:
        if key in lower:
            return key
    return None

def format_dataset(name: str, order: List[str], names: Dict[str, str]) -> str:
    key = find_dataset_key_by_substring(name, order)
    if key is None:
        return name
    return names.get(key, key)

def dataset_sort_key(name: str, order: List[str]) -> int:
    key = find_dataset_key_by_substring(name, order)
    if key is None:
        return 999
    try:
        return order.index(key)
    except ValueError:
        return 999


def extract_tables(log_file: str) -> List[str]:
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    return re.findall(r'\| model.*?\| mean.*?\|.*?\n', content, re.DOTALL)


def parse_table(table_text: str) -> pd.DataFrame:
    lines = [line for line in table_text.strip().split('\n') if line.strip() and not line.strip().startswith('|:')]
    rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.split('|')[1:-1]]
        rows.append(cells)
    if rows:
        df = pd.DataFrame(rows[1:], columns=rows[0])
        return df[df['model'] != 'mean']
    return pd.DataFrame()

def sort_xkv_variants(labels: List[str]) -> List[str]:
    patt_with_factor = re.compile(r'^xKV[-_](\d+)[-_]k(\d+)[-_]v(\d+)$', re.IGNORECASE)
    items: List[Tuple[float, int, int, int, str]] = []
    for lbl in labels:
        m = patt_with_factor.match(lbl)
        if not m:
            continue
        factor = int(m.group(1))
        k = int(m.group(2))
        v = int(m.group(3))
        eff_k = k / max(1, factor)
        items.append((eff_k, factor, k, v, lbl))
    items.sort(key=lambda x: (-x[0], x[1], x[2], x[3], x[4]))
    return [it[-1] for it in items]


def sort_minicache_variants(labels: List[str]) -> List[str]:
    """Sort MiniCache variants by the first numeric value in the label (desc).

    Examples handled: 'minicache_16-27', 'MiniCache-32', 'minicache32_foo'.
    If no number is found, place it after numbered ones (tie-break by label).
    """
    items: List[Tuple[float | None, str]] = []
    for lbl in labels:
        if lbl.lower() == 'minicache':
            # Baseline label is handled separately by EXPERIMENT_ORDER
            continue
        m = re.search(r'minicache[^0-9]*([0-9]+(?:\.[0-9]+)?)', lbl, flags=re.IGNORECASE)
        val: float | None
        if m:
            try:
                val = float(m.group(1))
            except ValueError:
                val = None
        else:
            val = None
        items.append((val, lbl))
    # Sort by numeric value desc; unknowns (None) go last; tie-break lexicographically
    items.sort(key=lambda t: (-t[0] if t[0] is not None else float('inf'), t[1]))
    return [lbl for _, lbl in items]


def sort_kivi_variants(labels: List[str]) -> List[str]:
    """Sort KIVI variants like 'kivi-gs128' by gs value ascending (small first)."""
    items: List[Tuple[float | None, str]] = []
    for lbl in labels:
        if lbl.lower() == 'kivi':
            continue
        m = re.search(r'kivi[^a-z0-9]?gs([0-9]+(?:\.[0-9]+)?)', lbl, flags=re.IGNORECASE)
        val: float | None
        if m:
            try:
                val = float(m.group(1))
            except ValueError:
                val = None
        else:
            val = None
        items.append((val, lbl))
    # Sort by numeric value asc; unknowns (None) go last; tie-break lexicographically
    items.sort(key=lambda t: (t[0] if t[0] is not None else float('inf'), t[1]))
    return [lbl for _, lbl in items]


def get_experiment_sort_key(experiment: str) -> int:
    base_experiment = 'xKV' if experiment.startswith('xKV') else experiment
    lowered = base_experiment.lower()
    for i, exp in enumerate(EXPERIMENT_ORDER):
        if exp.lower() in lowered:
            return i
    return 999


# Benchmark-specific config

def build_config(benchmark: str) -> Tuple[List[str], Dict[str, str], str, str]:
    bench = benchmark.lower()
    if bench == 'ruler':
        dataset_order = RULER_DATASET_ORDER
        dataset_names = RULER_DATASET_NAMES

        title = 'RULER Results Table (%)'
        return dataset_order, dataset_names, title
    elif bench == 'longbench':
        dataset_order = LONGBENCH_DATASET_ORDER
        dataset_names = LONGBENCH_DATASET_NAMES

        title = 'LongBench Results Table (%)'
        return dataset_order, dataset_names, title
    else:
        raise ValueError("benchmark must be 'ruler' or 'longbench'")


def build_and_save_table(bench: str, frames: List[pd.DataFrame]):
    dataset_order, dataset_names, title = build_config(bench)
    if frames and 'source_file' in frames[0].columns:
        first_log = frames[0]['source_file'].iloc[0]
        log_dir = os.path.dirname(first_log)
        csv_name = os.path.basename(log_dir) + '.csv'
    else:
        csv_name = 'results.csv'

    combined = pd.concat(frames, ignore_index=True)
    combined['baseline'] = pd.to_numeric(combined['baseline'], errors='coerce')
    combined['baseline'] = combined['baseline'] * 100
    combined = combined.sort_values(['experiment_sort_key', 'sort_key'])

    duplicates = combined.duplicated(subset=['experiment', 'dataset_formatted'], keep=False)
    if duplicates.any():
        print(f"\nFound {duplicates.sum()} duplicate experiment-dataset combinations ({bench}):")
        dup_data = combined[duplicates].sort_values(['experiment', 'dataset_formatted', 'baseline'])
        for (exp, dataset), group in dup_data.groupby(['experiment', 'dataset_formatted']):
            unique_scores = group['baseline'].unique()
            if len(unique_scores) != 1:
                srcs = ', '.join(group['source_file'].unique())
                print(f"  {exp} - {dataset}: {len(group)} results: {unique_scores} | Sources: {srcs}")

    combined = combined.drop_duplicates(subset=['experiment', 'dataset_formatted'], keep='last')

    output_df = combined[['experiment', 'dataset_formatted', 'baseline']].copy()
    output_df.columns = ['Experiment', 'Dataset', 'Score']
    table = output_df.pivot(index='Experiment', columns='Dataset', values='Score')

    # Dynamic dataset columns by observed sort_key, then name
    cols = list(table.columns)
    order_map = (
        combined[['dataset_formatted', 'sort_key']]
        .drop_duplicates()
        .groupby('dataset_formatted')['sort_key']
        .min()
        .to_dict()
    )
    cols_sorted = sorted(cols, key=lambda c: (order_map.get(c, 999), c))
    table = table.reindex(columns=cols_sorted)

    # Build experiment order with xKV variants and budgeted methods
    experiment_order: List[str] = []
    xkv_variants_present = [idx for idx in table.index if idx.lower().startswith('xkv') and idx != 'xKV']
    xkv_variants_sorted = sort_xkv_variants(xkv_variants_present)

    budgeted_labels: List[Tuple[float, int, str]] = []
    for label in table.index:
        for m_i, mth in enumerate(BUDGETED_METHODS):
            if label.lower().startswith(mth.lower() + '-'):
                m = re.search(r'-([0-9]+(?:\.[0-9]+)?)(k)?$', label, re.IGNORECASE)
                if m:
                    val = float(m.group(1))
                    if m.group(2):
                        val *= 1000.0
                else:
                    val = 0.0
                budgeted_labels.append((val, m_i, label))
                break
    budgeted_labels.sort(key=lambda t: (-t[0], t[1], t[2]))

    for exp in EXPERIMENT_ORDER:
        if exp in table.index:
            experiment_order.append(exp)
        # Insert MiniCache filename-based variants right after the MiniCache slot
        if exp == 'MiniCache':
            mini_labels = [idx for idx in table.index if idx.lower().startswith('minicache')]
            if mini_labels:
                mini_sorted = sort_minicache_variants(mini_labels)
                for lbl in mini_sorted:
                    if lbl not in experiment_order:
                        experiment_order.append(lbl)
        if exp == 'xKV' and xkv_variants_sorted:
            experiment_order.extend([v for v in xkv_variants_sorted if v not in experiment_order])
        if exp == 'KIVI':
            kivi_labels = [idx for idx in table.index if idx.lower().startswith('kivi')]
            if kivi_labels:
                kivi_sorted = sort_kivi_variants(kivi_labels)
                for lbl in kivi_sorted:
                    if lbl not in experiment_order:
                        experiment_order.append(lbl)
        if exp == 'StreamingLLM' and budgeted_labels:
            for _, _, lbl in budgeted_labels:
                if lbl not in experiment_order and lbl in table.index:
                    experiment_order.append(lbl)

    remaining = [idx for idx in table.index if idx not in experiment_order]
    if remaining:
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
    table = table.dropna(how='all')

    print(f"\n=== {title} ===")
    print(table.to_string(float_format='%.2f'))

    table_rounded = table.round(3)
    table_rounded.to_csv(csv_name, float_format='%.3f')
    print(f"Table saved to {csv_name}")


def main():
    parser = argparse.ArgumentParser(description='Extract experiment results from log files (autodetect RULER/LongBench)')
    parser.add_argument('dir', help='Log file directory')
    args = parser.parse_args()

    log_files = glob.glob(os.path.join(args.dir, '*.log'))
    if not log_files:
        print(f"No log files found in {args.dir}")
        return

    print(f"Found {len(log_files)} log files: {log_files}")

    combined_by_bench: Dict[str, List[pd.DataFrame]] = {'ruler': [], 'longbench': []}

    for log_file in log_files:
        try:
            tables = extract_tables(log_file)
            print(f"Processing {log_file}: Found {len(tables)} result tables")
            for idx, table_text in enumerate(tables):
                df = parse_table(table_text)
                if df.empty:
                    continue

                datasets = [str(x) for x in df['dataset'].tolist()]
                is_ruler = any('ruler/' in s for s in datasets)
                # Derive LongBench keys from config to avoid duplication
                lb_keys, _, _ = build_config('longbench')
                is_longbench = any(any(k in s.lower() for k in lb_keys) for s in datasets) if not is_ruler else False

                bench = None
                if is_ruler:
                    bench = 'ruler'
                elif is_longbench:
                    bench = 'longbench'
                else:
                    continue

                dataset_order, dataset_names, _ = build_config(bench)

                # Experiment label is the raw filename stem (before .log)
                df['experiment'] = os.path.splitext(os.path.basename(log_file))[0]
                df['dataset_formatted'] = df['dataset'].apply(lambda n: format_dataset(n, dataset_order, dataset_names))
                df['sort_key'] = df['dataset'].apply(lambda n: dataset_sort_key(n, dataset_order))
                df['experiment_sort_key'] = df['experiment'].apply(get_experiment_sort_key)
                df['source_file'] = log_file
                combined_by_bench[bench].append(df)
        except Exception as e:
            print(f"Error processing {log_file}: {e}")

    any_output = False
    for bench, frames in combined_by_bench.items():
        if frames:
            any_output = True
            build_and_save_table(bench, frames)

    if not any_output:
        print("No valid tables found for RULER or LongBench")


if __name__ == '__main__':
    main()
