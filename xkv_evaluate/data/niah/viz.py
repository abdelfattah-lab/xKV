################################################################################
# Copyright (c) Microsoft Corporation. and affiliates
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2024 ByteDance Ltd. and/or its affiliates.
################################################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os
import json
from collections import Counter


from argparse import ArgumentParser, Namespace
import json, os

def parse_args() -> Namespace:
    def str_to_list(arg):
        return arg.split(',')
    p = ArgumentParser()
    p.add_argument("--path", type=str, default='./archive/Llama-3-8B-Instruct-Gradient-1048k/niah_131072_96_full.jsonl')
    return p.parse_args()


def summary(jsonl_file_path: str):
    datas = []

    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Parse each line as a JSON object and append to the list
            data = json.loads(line.strip())
            # loop k, if k has a list, take [0]
            for key in data:
                if isinstance(data[key], list):
                    data[key] = data[key][0]
            datas.append(data)
    
    res = Counter()
    for ii in datas:
        res[(ii["context_length"], ii["depth_percent"])] += ii["correct"] == True
        if ii["correct"] is False:
            print(ii["pred"])
    sorted(res.items())
    with open("tmp.json", "w") as json_file:
        json.dump(datas, json_file)
    return res

def plot_needle_viz(
    res_file="tmp.json",
    model_name="Llama-3-8B-Instruct-1M",
    min_context=1024,
    max_context=1000000,
    mode="ours",
    output_path="figures/",
):
    def get_context_size(x, is_128k: bool = False):
        if is_128k:
            return f"{round(x / 1024)}K"
        if x > 990000:
            return f"{round(x / 1000000)}M"
        if x <= 10000:
            return "10K" if x > 5000 else "1K"
        if round(x / 1000) == 128:
            return "128K"
        return f"{round(x / 10000)* 10}K"

    plt.rc("axes", titlesize=25)  # fontsize of the title
    plt.rc("axes", labelsize=25)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=20)  # fontsize of the x tick labels
    plt.rc("ytick", labelsize=20)  # fontsize of the y tick labels
    plt.rc("legend", fontsize=20)  # fontsize of the legend

    df = pd.read_json(res_file)
    os.system(f"rm {res_file}")
    accuracy_df = df.groupby(["context_length", "depth_percent"])["correct"].mean()
    accuracy_df = accuracy_df
    accuracy_df = accuracy_df.reset_index()
    accuracy_df = accuracy_df.rename(
        columns={
            "correct": "Score",
            "context_length": "Context Length",
            "depth_percent": "Document Depth",
        }
    )

    pivot_table = pd.pivot_table(
        accuracy_df,
        values="Score",
        index=["Document Depth", "Context Length"],
        aggfunc="mean",
    ).reset_index()  # This will aggregate
    pivot_table = pivot_table.pivot(
        index="Document Depth", columns="Context Length", values="Score"
    )  # This will turn into a proper pivot

    # Create a custom colormap. Go to https://coolors.co/ and pick cool colors
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"]
    )

    # Create the heatmap with better aesthetics
    plt.figure(figsize=(14, 7))  # Can adjust these dimensions as needed
    ax = sns.heatmap(
        pivot_table,
        # annot=True,
        fmt="g",
        cmap=cmap,
        # cbar_kws={'label': 'Score'},
        vmin=0,
        vmax=1,
    )

    min_context_str = f"{min_context // 1000}K" if min_context >= 1000 else min_context
    max_context_str = f"{max_context // 1000}K" if max_context >= 1000 else max_context

    # More aesthetics
    if mode.lower() == "ours":
        name = " w/ ShadowKV"
    elif mode.lower() == "streamllm":
        name = " w/ StreamingLLM"
    elif mode.lower() == "infllm":
        name = " w/ InfLLM"
    else:
        name = ""
    if "Yi" in model_name:
        context = "200K"
    elif any(key in model_name for key in ["Phi", "Qwen2"]):
        context = "128K"
    else:
        context = "1M"
    plt.title(
        f"Needle in A Haystack {model_name}{name}"
    )  # Adds a title
    plt.xlabel("Context Length")  # X-axis label
    plt.ylabel("Depth Percent (%)")  # Y-axis label

    # Centering x-ticks
    xtick_labels = pivot_table.columns.values
    xtick_labels = [get_context_size(x, context == "128K") for x in xtick_labels]
    ax.set_xticks(np.arange(len(xtick_labels)) + 0.5, minor=False)
    ax.set_xticklabels(xtick_labels)

    # Drawing white grid lines
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color("white")
        spine.set_linewidth(1)

    # Iterate over the number of pairs of gridlines you want
    for i in range(pivot_table.shape[0]):
        ax.axhline(i, color="white", lw=1)
    for i in range(pivot_table.shape[1]):
        ax.axvline(i, color="white", lw=1)

    # Ensure the ticks are horizontal and prevent overlap
    plt.xticks(rotation=60)
    plt.yticks(rotation=0)

    # Fit everything neatly into the figure area
    plt.tight_layout()

    # Save and Show the plot
    save_path = os.path.join(
        output_path,
        f"needle_viz_{model_name}_{mode}_{min_context_str}_{max_context_str}.pdf",
    )
    plt.savefig(save_path, dpi=1000)
    print(f"Needle plot saved to {save_path}.")
    plt.show()

if __name__ == '__main__':
    args = parse_args()
    res = summary(args.path)
    plot_needle_viz()