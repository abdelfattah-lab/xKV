################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

# Unified eval entry for xKV and xKV-SR methods.
#
# xKV / baseline methods (HF model.generate + patch):
#   python evaluate/eval_acc.py --model_name_or_path ... --flash2 --method xkv [--rank_k ... --rank_v ...]
#   python evaluate/eval_acc.py --model_name_or_path ... --flash2 --method kivi
#
# xKV-SR methods (custom LLM engine):
#   python evaluate/eval_acc.py --model_name_or_path ... --method xkv_sr \
#       --sparse_budget 2048 --layer_group_size 2 --rank_k 192 --rank_v 288 --chunk_size 8

import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

import warnings
warnings.filterwarnings("ignore")

import torch
import gc
from termcolor import colored
from argparse import ArgumentParser, Namespace

import torch.distributed as dist
import datetime
import json

import numpy as np
import random

# xKV-SR method names (routed to custom LLM engine)
XKV_SR_METHODS = {
    'shadowkv',
    'xk_sr', 'xkv_sr',
}


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


class DistConfig:
    def __init__(self, is_distributed, rank, world_size, device, master_process):
        self.is_distributed = is_distributed
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.master_process = master_process


def init_dist():
    rank = int(os.environ.get("RANK", -1))
    is_distributed = rank != -1
    if is_distributed:
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{rank}"
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=60 * 90), device_id=torch.device(device))
        master_process = rank == 0
    else:
        device = "cuda:0"
        world_size = 1
        master_process = True

    if master_process:
        print(colored(f"[Dist init] world_size={world_size}", 'cyan'))

    return DistConfig(is_distributed, rank, world_size, device, master_process)


def parse_args() -> Namespace:
    def str_to_list(arg):
        return arg.split(',')

    p = ArgumentParser()
    from utils import add_common_args
    add_common_args(p)

    p.add_argument("--dataset_name", type=str_to_list, default=["ruler/niah_single_1"])
    p.add_argument("--num_samples", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--datalen", type=int, default=64 * 1024)
    p.add_argument("--result_dir", type=str, default="results")
    p.add_argument("--use_chat_template", action="store_true")

    # method routing
    p.add_argument("--method", type=str, default=None,
                   choices=[None, "dense", "xkv", "streamingllm", "snapkv", "pyramidkv", "kivi", "quest",
                            "shadowkv", "xk_sr", "xkv_sr"],
                   help="Compression method. None/dense = baseline; xkv = xKV SVD pipeline; "
                        "xk_sr/xkv_sr/shadowkv = xKV-SR custom engine.")
    p.add_argument("--inference_mode", type=str, default="single_turn",
                   choices=["single_turn", "multi_turn"])

    # xKV-SR hyperparams (--rank_k, --rank_v, --layer_group_size already in add_common_args)
    p.add_argument("--sparse_budget", type=int, default=2048)
    p.add_argument("--chunk_size", type=int, default=8)
    p.add_argument("--rank", type=int, default=160)
    p.add_argument("--fake_svd", action="store_true")
    p.add_argument("--minference_sr", action="store_true", default=False)

    return p.parse_args()


# ---------------------------------------------------------------------------
# xKV-SR pipeline (custom LLM engine)
# ---------------------------------------------------------------------------

def run_xkv_sr(args, dist_config):
    from xKV_SR.models import choose_model_class
    from evaluator import Evaluator
    from evaluate.data.dataset import Dataset

    model_name = args.model_name_or_path
    evaluator = Evaluator(dist_config)

    if dist_config.master_process:
        print(colored(f"[xKV-SR] method={args.method}  datasets={args.dataset_name}", 'cyan'))

    LLM = choose_model_class(model_name)
    llm = LLM(
        model_name=model_name,
        batch_size=args.batch_size,
        device=dist_config.device,
        max_length=args.datalen + 2048,
        attn_mode=args.method,
        dtype=torch.bfloat16,
        sparse_budget=args.sparse_budget,
        chunk_size=args.chunk_size,
        rank=args.rank,
        minference=args.minference_sr,
        rank_k=args.rank_k,
        rank_v=args.rank_v,
        group_size=args.layer_group_size,
        fake_svd=args.fake_svd,
    )

    if dist_config.master_process:
        llm.print_kv_stats()

    for dataset_name in args.dataset_name:
        dataset = Dataset(
            dataset_name, llm.tokenizer, args.datalen, args.num_samples,
            dist_config.rank, dist_config.world_size,
            inference_mode=args.inference_mode,
        )
        archive_path = (
            f"temporary/{model_name.split('/')[-1]}/"
            f"{dataset_name}_{args.datalen}_{args.method}"
            f"_b{args.sparse_budget}_c{args.chunk_size}"
            f"_x{args.layer_group_size}_r{args.rank}_k{args.rank_k}_v{args.rank_v}.jsonl"
        )
        evaluator.test(llm, llm.tokenizer, dataset, archive_path, setting=args.method)

        df = evaluator.summarize()
        if dist_config.master_process:
            result = df[df["dataset"] == dataset_name]
            per_dataset_stats = result.to_dict(orient="records")[0]
            print(colored(f"Results for {dataset_name}: {per_dataset_stats}", 'cyan'))

            benchmark_name = dataset_name.split('/')[-2]
            raw_model_name = model_name.split('/')[-1]
            os.makedirs(os.path.join(args.result_dir, benchmark_name), exist_ok=True)
            with open(os.path.join(args.result_dir, f"{benchmark_name}/{raw_model_name}.json"), "a") as f:
                meta_data_to_log = {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "args": vars(args),
                }
                meta_data_to_log.update(per_dataset_stats)
                json.dump(meta_data_to_log, f)
                f.write("\n")

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    evaluator.summarize()


def _xkv_setting(args):
    """Return setting key for result logging."""
    m = args.method
    if m == "xkv":
        if args.layer_merge_impl == "slerp":
            return f"minicache_l{args.start_layer_idx}_{args.end_layer_idx}"
        elif args.kv_bits < 16:
            return f"xkv{args.layer_group_size}_k{args.rank_k}_v{args.rank_v}_{args.kv_bits}bit"
        else:
            return f"xkv{args.layer_group_size}_k{args.rank_k}_v{args.rank_v}"
    elif m in ("snapkv", "pyramidkv", "kivi", "streamingllm", "quest"):
        return m
    else:
        return "dense"


# ---------------------------------------------------------------------------
# xKV upstream pipeline (HF model.generate + patch)
# ---------------------------------------------------------------------------

def run_xkv(args, dist_config):
    from evaluator import Evaluator
    from data.dataset import Dataset

    model_name = args.model_name_or_path
    evaluator = Evaluator(dist_config)

    if dist_config.master_process:
        os.makedirs("temporary", exist_ok=True)
        print(colored(f"[xKV] method={args.method} data_names: {args.dataset_name}", 'cyan'))

    from utils import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, use_flash_attn2=args.flash2)

    if args.method == "xkv":
        from utils import apply_kv_compress_patch
        model = apply_kv_compress_patch(model, args)
    elif args.method == "streamingllm":
        from minference import MInference
        model = MInference("dense", model_name, kv_type="streamingllm",
                           attn_kwargs={"n_local": 8064, "n_init": 128})(model)
    elif args.method == "snapkv":
        from minference import MInference
        model = MInference("dense", model_name, kv_type="snapkv",
                           attn_kwargs={"max_capacity_prompt": 8192})(model)
    elif args.method == "pyramidkv":
        from minference import MInference
        model = MInference("dense", model_name, kv_type="pyramidkv",
                           attn_kwargs={"max_capacity_prompt": 8192})(model)
    elif args.method == "kivi":
        from minference import MInference
        model = MInference("dense", model_name, kv_type="kivi",
                           attn_kwargs={"bits": 2, "group_size": 128, "residual_length": 128})(model)
    elif args.method == "quest":
        from minference import MInference
        model = MInference("dense", model_name, kv_type="quest",
                           attn_kwargs={"chunk_size": 16, "token_budget": 4096})(model)
    # else: dense baseline, no patch

    for dataset_name in args.dataset_name:
        dataset = Dataset(dataset_name, tokenizer, args.datalen, args.num_samples,
                          evaluator.dist_config.rank, evaluator.dist_config.world_size,
                          use_chat_template=args.use_chat_template)
        setting = _xkv_setting(args)
        archive_path = os.path.join("temporary", model_name.split('/')[-1], f"{dataset_name}_{args.datalen}_{setting}.jsonl")
        evaluator.test(model, tokenizer, dataset, archive_path, setting=setting)

        df = evaluator.summarize()
        if dist_config.master_process:
            df.reset_index(drop=True)
            result = df[df["dataset"] == dataset_name]
            per_dataset_stats = result.to_dict(orient="records")[0]
            print(colored(f"Results for {dataset_name}: {per_dataset_stats}", 'cyan'))

            benchmark_name = dataset_name.split('/')[-2]
            raw_model_name = model_name.split('/')[-1]
            os.makedirs(os.path.join(args.result_dir, f"{benchmark_name}"), exist_ok=True)
            with open(os.path.join(args.result_dir, f"{benchmark_name}/{raw_model_name}.json"), "a") as f:
                meta_data_to_log = {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "args": vars(args),
                }
                meta_data_to_log.update(per_dataset_stats)
                json.dump(meta_data_to_log, f)
                f.write("\n")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    evaluator.summarize(shown_avg=True)


if __name__ == '__main__':
    args = parse_args()
    seed_everything(42)
    dist_config = init_dist()

    if args.method in XKV_SR_METHODS:
        run_xkv_sr(args, dist_config)
    else:
        run_xkv(args, dist_config)

    if dist_config.is_distributed:
        dist.destroy_process_group()
