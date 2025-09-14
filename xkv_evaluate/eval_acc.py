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

# OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 8 test/eval_acc.py --datalen 131072 --method shadowKV --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multikey_3,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/cwe,ruler/fwe,ruler/qa_1,ruler/qa_2" --sparse_budget 896 --rank 160 --chunk_size 8

import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

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
        dist.init_process_group(backend="nccl",timeout=datetime.timedelta(seconds=60*90), device_id=torch.device(device))

        master_process = (
            rank == 0
        )
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
    p.add_argument("--datalen", type=int, default=64*1024, help="The length of the context.")
    p.add_argument("--result_dir", type=str, default="results")
    return p.parse_args()

if __name__ == '__main__':

    args = parse_args()
    model_name = args.model_name_or_path
    dataset_names = args.dataset_name
    num_samples = args.num_samples
    datalen = args.datalen

    seed_everything(42)
    dist_config = init_dist()
    
    if dist_config.master_process:
        os.makedirs("temporary", exist_ok=True)

    from evaluator import Evaluator
    from data.dataset import Dataset
    
    evaluator = Evaluator(dist_config)
    
    if dist_config.master_process:
        print(colored(f"data_names: {dataset_names}", 'cyan'))
    
    from utils import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, use_flash_attn2=args.flash2)
    
    if dist_config.master_process:
        print("Enabled xKV: {}".format(args.xKV))
        
    if args.xKV:
        from utils import apply_kv_compress_patch
        model = apply_kv_compress_patch(model, args)

    # Determine budget based on dataset type (8k for RULER, 1k for LongBench)
    is_ruler = any(dataset.startswith('ruler/') for dataset in dataset_names)
    budget = args.budget
    
    # Configuration mapping for MInference methods
    minference_configs = {
        'streamingllm': {
            'kv_type': 'streamingllm',
            'attn_kwargs': {"n_local": budget - 32, "n_init": 32}
        },
        'snapKV': {
            'kv_type': 'snapkv',
            'attn_kwargs': {"max_capacity_prompt": budget}
        },
        'pyramidkv': {
            'kv_type': 'pyramidkv',
            'attn_kwargs': {"max_capacity_prompt": budget}
        },
        'kivi': {
            'kv_type': 'kivi',
            'attn_kwargs': {"bits": 2, "group_size": 128, "residual_length": 128}
        },
        'quest': {
            'kv_type': 'quest',
            'attn_kwargs': {"chunk_size": 16, "token_budget": budget // 2}
        }
    }
    
    # Apply MInference patch based on arguments
    for method, config in minference_configs.items():
        if getattr(args, method, False):
            from minference import MInference
            attn_kwargs = config['attn_kwargs']
            if method == 'kivi':
                group_size = attn_kwargs.get('group_size', 'N/A')
                residual_length = attn_kwargs.get('residual_length', 'N/A')
                if dist_config.master_process:
                    print(colored(f"Applying {method.upper()} with group_size: {group_size}, residual_length: {residual_length}...", "green"))
            elif method == 'streamingllm':
                n_init = attn_kwargs.get('n_init', 'N/A')
                n_local = attn_kwargs.get('n_local', 'N/A')
                if dist_config.master_process:
                    print(colored(f"Applying {method.upper()} with n_init: {n_init}, n_local: {n_local}...", "green"))
            else:
                budget = (attn_kwargs.get('max_capacity_prompt') or 
                          attn_kwargs.get('token_budget') or 
                          attn_kwargs.get('n_local', 'N/A'))
                dataset_type = 'RULER' if is_ruler else 'LongBench'
                if dist_config.master_process:
                    print(colored(f"Applying {method.upper()} with budget: {budget} (dataset type: {dataset_type})...", "green"))
            minference_patch = MInference(
                attn_type="dense",
                model_name=model_name,
                kv_type=config['kv_type'],
                attn_kwargs=config['attn_kwargs']
            )
            model = minference_patch(model)
            if dist_config.master_process:
                print(colored(f"{method.upper()} applied successfully", "green"))
            break
    
    for dataset_name in dataset_names:
        dataset = Dataset(dataset_name, tokenizer, datalen, num_samples, evaluator.dist_config.rank, evaluator.dist_config.world_size)
        archive_path = os.path.join("temporary", model_name.split('/')[-1])
        if args.xKV:
            if args.layer_merge_impl == "slerp":
                file_name = f"{dataset_name}_{datalen}_minicache_{args.start_layer_idx}_to_{args.end_layer_idx}.jsonl"
            else:
                if args.kv_bits < 16:
                    file_name = f"{dataset_name}_{datalen}_xKV-{args.layer_group_size}_k{args.rank_k}_v{args.rank_v}_{args.kv_bits}bits_gs{args.group_size}.jsonl"
                else:
                    file_name = f"{dataset_name}_{datalen}_xKV-{args.layer_group_size}_k{args.rank_k}_v{args.rank_v}.jsonl"
        elif args.snapKV:
            file_name = f"{dataset_name}_{datalen}_snapKV.jsonl"
        elif args.pyramidkv:
            file_name = f"{dataset_name}_{datalen}_pyramidkv.jsonl"
        elif args.kivi:
            file_name = f"{dataset_name}_{datalen}_kivi.jsonl"
        elif args.streamingllm:
            file_name = f"{dataset_name}_{datalen}_streamingllm.jsonl"
        elif args.quest:
            file_name = f"{dataset_name}_{datalen}_quest.jsonl"
        else:
            file_name = f"{dataset_name}_{datalen}.jsonl"
        archive_path = os.path.join(archive_path, file_name)
        evaluator.test(model, tokenizer, dataset, archive_path)
        
        stats = evaluator.all_stats[-1]
        benchmark_name = dataset_name.split('/')[-2]
        raw_model_name = model_name.split('/')[-1]
        # Log the results of each datasets
        
        df = evaluator.summarize()
        
        if dist_config.master_process:
            df.reset_index(drop=True)
            result = df[df["dataset"] == dataset_name]
            per_dataset_stats = result.to_dict(orient="records")[0]
            if dist_config.master_process:
                print(colored(f"Results for {dataset_name}: {per_dataset_stats}", 'cyan'))

        if dist_config.master_process:
            os.makedirs(os.path.join(args.result_dir, f"{benchmark_name}"), exist_ok=True)
            with open(os.path.join(args.result_dir, f"{benchmark_name}/{raw_model_name}.json"), "a") as f:
                meta_data_to_log = {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "args": vars(args)
                }
                meta_data_to_log.update(per_dataset_stats)
                json.dump(meta_data_to_log, f)
                f.write("\n")
    
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    evaluator.summarize(shown_avg=True)
    if dist_config.is_distributed:
        dist.destroy_process_group()