import os
import json
import csv
from glob import glob
import numpy as np
import subprocess
import re

def extract_xkv_index(filename):
    match = re.search(r"xKV-(\d+)", filename)
    return int(match.group(1)) if match else float("inf")

def process_jsonl_file(input_dir):
    jsonl_files = glob(os.path.join(input_dir, "**", "*multiturn*.jsonl"), recursive=True)

    # Field
    rows = []

    for jsonl_file in jsonl_files:
        try:
            with open(jsonl_file, "r") as f:
                lines = f.readlines()
            if not lines:
                continue

            # Use the last line to extract the correct answers
            last_data = json.loads(lines[-1])
            correct = last_data.get("correct", [])
            if len(correct) < 5:
                continue

            correct_np = np.array(correct)
            rounds = [correct_np[i::5] for i in range(5)]
            round_scores = [r.mean() if len(r) > 0 else 0.0 for r in rounds]
            overall = correct_np.mean()

            rows.append({
                "file": os.path.relpath(jsonl_file, input_dir),
                "round_1": round_scores[0],
                "round_2": round_scores[1],
                "round_3": round_scores[2],
                "round_4": round_scores[3],
                "round_5": round_scores[4],
                "overall_avg": overall
            })

        except Exception as e:
            print(f"Error in {jsonl_file}: {e}")
            
    return rows


csv_fields = ["file", "round_1", "round_2", "round_3", "round_4", "round_5", "overall_avg"]

# Set input directory and output CSV file
input_dir = "temporary/Meta-Llama-3.1-8B-Instruct/ruler"
output_csv = "ruler_multiturn_results.csv"

results = process_jsonl_file(input_dir)
results.sort(key=lambda r: r["file"])
results.sort(key=lambda row: extract_xkv_index(row["file"]))

# Write results to CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields)
    writer.writeheader()
    writer.writerows(results)

print(f"Summary written to: {output_csv}")
subprocess.run(["code", output_csv])