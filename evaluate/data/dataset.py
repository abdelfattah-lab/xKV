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

from datasets import load_dataset
from termcolor import colored
import random
import numpy as np
from collections import Counter

# RULER
from .metrics import (
    needle_score, 
    string_match_part, 
    multi_number, 
    multi_words, 
    normalize_answer, 
    rouge_score,
    retrieval_score,
    code_sim_score,
)

# NIAH
from .utils import generate_random_number, read_context_files, create_contexts, NIAH_TEMPLATE, RANDOM_NEEDLE_CITIES, LONG_BENCH_TEMPLATE

#### LONG_BENCH ####

def f1_score_longbench(prediction, ground_truth):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def qa_f1_score_longbench(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score_longbench(prediction_tokens, ground_truth_tokens)



METRICS_FN = {
    'niah': needle_score,
    'multi': multi_number,
    'vt': multi_words,
    'cwe': multi_words,
    'fwe': multi_words,
    'qa': string_match_part,
    
    "long_bench/narrativeqa": qa_f1_score_longbench,
    "long_bench/qasper": qa_f1_score_longbench,
    "long_bench/multifieldqa_en": qa_f1_score_longbench,
    "long_bench/hotpotqa": qa_f1_score_longbench,
    "long_bench/2wikimqa": qa_f1_score_longbench,
    "long_bench/musique": qa_f1_score_longbench,
    "long_bench/gov_report": rouge_score,
    "long_bench/qmsum": rouge_score,
    "long_bench/multi_news": rouge_score,
    "long_bench/triviaqa": qa_f1_score_longbench,
    "long_bench/samsum": rouge_score,
    "long_bench/passage_retrieval_en": retrieval_score,
    "long_bench/lcc": code_sim_score,
    "long_bench/repobench-p": code_sim_score,
}

GEN_LEN = {
    'niah': 64,
    'vt': 30,
    'cwe': 120,
    'fwe': 50,
    'qa': 32,
    
    "long_bench/narrativeqa": 128,
    "long_bench/qasper": 128,
    "long_bench/multifieldqa_en": 64,
    "long_bench/multifieldqa_zh": 64,
    "long_bench/hotpotqa": 32,
    "long_bench/2wikimqa": 32,
    "long_bench/musique": 32,
    "long_bench/dureader": 128,
    "long_bench/gov_report": 512,
    "long_bench/qmsum": 512,
    "long_bench/multi_news": 512,
    "long_bench/vcsum": 512,
    "long_bench/trec": 64,
    "long_bench/triviaqa": 32,
    "long_bench/samsum": 128,
    "long_bench/passage_count": 32,
    "long_bench/passage_retrieval_en": 32,
    "long_bench/passage_retrieval_zh": 32,
    "long_bench/lcc": 64,
    "long_bench/repobench-p": 64
}

DATADIR = {
    'ruler': 'evaluate/data/ruler/data',
    'niah': 'evaluate/data/niah/data',
}

Templates = {
    'base': "{ctx}",
    'llama-3': "<|start_header_id|>system<|end_header_id|>You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>{ctx}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    'yi': "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n{ctx}<|im_end|>\n<|im_start|>assistant\n",
    'glm': "<|system|>\nYou are a helpful assistant\n<|user|> \n{ctx}<|assistant|>\n",
    'lwm': "You are a helpful assistant.\nUSER: {ctx}\nASSISTANT: Answer: ",
    'qwen': "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n{ctx}<|im_end|>\n<|im_start|>assistant\n",
    'phi': "<|system|>\nYou are a helpful assistant<|end|>\n<|user|>\n{ctx}<|end|>\n<|assistant|>\n",
    "deepseek": "<｜begin▁of▁sentence｜>User: {task_template}\n\nAssistant:",
}

class Dataset:
    def __init__(self, dataset_name, tokenizer, datalen, num_samples, rank=0, world_size=1, use_chat_template=False):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.datalen = datalen
        self.num_samples = num_samples
        self.rank = rank
        self.world_size = world_size
        self.use_chat_template = use_chat_template
        self.is_sharded = False

        if dataset_name == 'niah':
            self.tokenized_prompts, self.gt, self.ctx_len, self.depth_pct = self.get_dataset()
        elif 'long_bench' in dataset_name:
            self.tokenized_prompts, self.gt, self.classes = self.get_dataset()
        elif 'multiturn' in dataset_name:
            self.tokenized_prompts, self.tokenized_queries, self.gt = self.get_dataset()
        else:
            self.tokenized_prompts, self.gt = self.get_dataset()
        
        self.num_samples = len(self.tokenized_prompts)
        self.gen_len = self.get_gen_len()
        self.metric = self.get_metric()

    def __str__(self) -> str:
        return f"Dataset: {self.dataset_name}, Num Samples: {self.num_samples}, Gen Len: {self.gen_len}, DataLen: {self.datalen}"

    def __repr__(self) -> str:
        return f"Dataset: {self.dataset_name}, Num Samples: {self.num_samples}, Gen Len: {self.gen_len}, DataLen: {self.datalen}"

    def __len__(self) -> int:
        return self.num_samples

    def shard(self, rank, world_size):
        if world_size > 1:
            shard_size = self.num_samples // world_size
            start = rank * shard_size
            end = start + shard_size if rank != world_size - 1 else self.num_samples
            shard_tokenized_prompts, shard_gt = self.tokenized_prompts[start:end], self.gt[start:end]
            if 'multiturn' in self.dataset_name:
                shard_tokenized_queries = self.tokenized_queries[start:end]
                self.tokenized_queries = shard_tokenized_queries
            self.tokenized_prompts = shard_tokenized_prompts
            self.gt = shard_gt
            self.num_samples = len(shard_tokenized_prompts)

        self.is_sharded = True

    def get_gen_len(self):
        if 'niah' == self.dataset_name:
            return 10
        elif 'niah' in self.dataset_name:
            return 128
        elif 'vt' in self.dataset_name:
            return 30
        elif 'cwe' in self.dataset_name:
            return 120
        elif 'fwe' in self.dataset_name:
            return 50
        elif 'qa' in self.dataset_name:
            return 32
        elif 'long_bench' in self.dataset_name:
            return GEN_LEN[self.dataset_name]
        else:
            raise Exception("Gen len not found")

    def __getitem__(self, idx):
        if 'persona' in self.dataset_name:
            return self.tokenized_prompts[idx], self.queries[idx], self.gt[idx]
        return self.tokenized_prompts[idx], self.gt[idx]

    def get_metric(self):
        if 'long_bench' in self.dataset_name and self.dataset_name in METRICS_FN:
            return METRICS_FN[self.dataset_name]
        elif 'multiquery' in self.dataset_name or 'multivalue' in self.dataset_name:
            return METRICS_FN['multi']
        elif 'niah' in self.dataset_name:
            return METRICS_FN['niah']
        elif 'vt' in self.dataset_name:
            return METRICS_FN['vt']
        elif 'cwe' in self.dataset_name:
            return METRICS_FN['cwe']
        elif 'fwe' in self.dataset_name:
            return METRICS_FN['fwe']
        elif 'qa' in self.dataset_name:
            return METRICS_FN['qa']
        else:
            raise Exception("Metric not found")

    def get_dataset(self):
        if 'ruler' in self.dataset_name: # ruler/xxx
            task = self.dataset_name.split('/')[-1]
            assert self.datalen in [8*1024, 16*1024, 32*1024, 64*1024, 128*1024, 256*1024], "Only support datalen of 16k, 32k, 64k, 128k"

            if 'llama-3' in self.tokenizer.name_or_path.lower():
                model_dir = 'llama-3'
            elif 'yi' in self.tokenizer.name_or_path.lower():
                model_dir = 'yi'
            elif 'lwm' in self.tokenizer.name_or_path.lower():
                model_dir = 'lwm'
            elif 'glm' in self.tokenizer.name_or_path.lower():
                model_dir = 'glm'
            elif 'qwen' in self.tokenizer.name_or_path.lower():
                model_dir = 'qwen'
            elif 'phi' in self.tokenizer.name_or_path.lower():
                model_dir = 'phi'
            elif 'deepseek' in self.tokenizer.name_or_path.lower():
                model_dir = 'deepseek'
            else:
                raise Exception("Model not found", self.tokenizer.name_or_path)

            dataset = load_dataset("json", data_files=f'{DATADIR["ruler"]}/{model_dir}/{self.datalen}/{task}/validation.jsonl', split='train')
            if self.num_samples > 0:
                self.num_samples = min(self.num_samples, len(dataset))
            else:
                self.num_samples = len(dataset)
            tokenized_prompts = []
            tokenized_queries = []
            gt = []

            if 'multiturn' in self.dataset_name:
                for i in range(self.num_samples):
                    input_text = dataset[i]['input']
                    # Combine the first query with the input text and tokenize them jointly
                    first_query = dataset[i]['queries'][0]
                    combined_text = input_text + first_query
                    input_ids = self.tokenizer(combined_text, return_tensors="pt", add_special_tokens=False)
                    tokenized_prompts.append(input_ids)
                    
                    # Process remaining queries (if any)
                    tokenized_query_list = []
                    for query in dataset[i]['queries'][1:]:
                        query_ids = self.tokenizer(query, return_tensors="pt", add_special_tokens=False)
                        tokenized_query_list.append(query_ids)
                    
                    tokenized_queries.append(tokenized_query_list)
                    gt.append(dataset[i]['answers'])
                return tokenized_prompts, tokenized_queries, gt
            else:
                for i in range(self.num_samples):
                    input_text = dataset[i]['input']
                    input_ids = self.tokenizer(input_text, return_tensors="pt", add_special_tokens=False)
                    tokenized_prompts.append(input_ids)
                    gt.append(dataset[i]['outputs'])

                return tokenized_prompts, gt
        
        elif 'long_bench' in self.dataset_name:
            task = self.dataset_name.split('/')[-1]
            dataset = load_dataset('THUDM/LongBench', task, split='test', trust_remote_code=True)
            
            if self.num_samples > 0:
                self.num_samples = min(self.num_samples, len(dataset))
            else:
                self.num_samples = len(dataset)
            tokenized_prompts = []
            gt = []
            classes = []

            for i in range(len(dataset)):
                if self.use_chat_template:
                    if 'llama-3' in self.tokenizer.name_or_path.lower():
                        model_template = Templates['llama-3'].format(ctx=LONG_BENCH_TEMPLATE[task])
                    elif 'yi' in self.tokenizer.name_or_path.lower():
                        model_template = Templates['yi'].format(ctx=LONG_BENCH_TEMPLATE[task])
                    elif 'glm' in self.tokenizer.name_or_path.lower():
                        model_template = Templates['glm'].format(ctx=LONG_BENCH_TEMPLATE[task])
                    elif 'qwen' in self.tokenizer.name_or_path.lower():
                        model_template = Templates['qwen'].format(ctx=LONG_BENCH_TEMPLATE[task])
                    elif 'phi' in self.tokenizer.name_or_path.lower():
                        model_template = Templates['phi'].format(ctx=LONG_BENCH_TEMPLATE[task])
                    elif "deepseek" in self.tokenizer.name_or_path.lower():
                        model_template = Templates['deepseek'].format(task_template=LONG_BENCH_TEMPLATE[task])
                    else:
                        raise Exception("Model not found for chat template", self.tokenizer.name_or_path)
                else:
                    model_template = LONG_BENCH_TEMPLATE[task]

                input_text = model_template.format(**dataset[i])
                # import pdb; pdb.set_trace()
                #breakpoint()
                # input_ids = truncate_by_tokens(input_text, self.tokenizer, self.datalen)
                input_ids = self.tokenizer(input_text, return_tensors="pt")

                #if input_ids.shape[-1] <= self.datalen and input_ids.shape[-1] > 4096:
                tokenized_prompts.append(input_ids)
                gt.append(dataset[i]['answers'])
                classes.append(dataset[i]['all_classes'])
            return tokenized_prompts, gt, classes
        else:
            raise ValueError(f"Dataset {self.dataset_name} not found, please choose in ruler, persona, infini_bench, needle, niah, long_bench")