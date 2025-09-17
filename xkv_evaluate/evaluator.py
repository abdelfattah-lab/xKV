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

import os
import torch
from termcolor import colored
from tqdm import tqdm
import torch.distributed as dist
import pandas as pd
import json
import datetime

from data.dataset import Dataset


class Evaluator:
    def __init__(self, dist_config):

        self.dist_config = dist_config

        # init final report
        self.all_stats = []

    def test(self, llm, tokenizer, dataset: Dataset, output_path: str, setting: str = 'baseline'):

        # mkdir if not exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if self.dist_config.master_process:
            print(colored(f"[Test] {llm.config.name_or_path} on {dataset.dataset_name}, results saved to {output_path}", 'green'))

        if dataset.is_sharded == False:
            dataset.shard(self.dist_config.rank, self.dist_config.world_size)

        bsz = 1
        scores = []
        preds = []

        # clear the file
        open(output_path, 'w').close()
        if self.dist_config.is_distributed:
            dist.barrier()

        progress_bar = tqdm(range(dataset.num_samples), desc='Testing', disable=self.dist_config.is_distributed and not self.dist_config.master_process)
        for i in range(dataset.num_samples):
            #prompt = torch.cat([dataset.tokenized_prompts[i*bsz+j] for j in range(bsz)], dim=0)
            prompt = dataset.tokenized_prompts[i]
            if 'long_bench' in dataset.dataset_name:
                #rets = llm.generate(**(prompt.to(llm.device)), max_new_tokens=dataset.gen_len, top_p=1.0, temperature=0.0, num_logits_to_keep=1, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                rets = llm.generate(**(prompt.to(llm.device)), max_new_tokens=dataset.gen_len, top_p=1.0, temperature=0.0, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                rets = [tokenizer.decode(rets[0][prompt.input_ids.shape[-1]:], skip_special_tokens=True)]
                for (pred, gt, all_classes) in zip(rets, dataset.gt[i*bsz:(i+1)*bsz], dataset.classes[i*bsz:(i+1)*bsz]):
                    scores.append(max([dataset.metric(pred, g, all_classes=all_classes) for g in gt]))
            elif 'niah_multiturn' in dataset.dataset_name:
                # 1. Initial context (prefill and compress)
                # NOTE(max410011): Use `generate` to get compressed KV-cache by calling `_prepare_cache_for_generation`
                first_ret = llm.generate(
                    **(prompt.to(llm.device)),
                    return_dict_in_generate=True,
                    max_new_tokens=dataset.gen_len,
                    top_p=1.0, 
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                context = first_ret.sequences
                past_key_values = first_ret.past_key_values
                
                # Store the first response
                first_generated_ids = first_ret.sequences[0, prompt.input_ids.shape[-1]:]
                first_response = tokenizer.decode(first_generated_ids, skip_special_tokens=True)
                rets = [[first_response]]
                # 2. Multi-turn queries
                for query in dataset.tokenized_queries[i]:
                    generated = []
                    cur_input = query["input_ids"].to(llm.device)
                    # NOTE(max410011): Reduece gen_len to speed up, you may need longer gen_len for other tasks
                    # for step in range(dataset.gen_len):
                    for _ in range(10):
                        with torch.no_grad():
                            output = llm(
                                input_ids=cur_input,
                                past_key_values=past_key_values,
                                return_dict=True,
                            )

                        next_token_logits = output.logits[:, -1, :]  # (B, vocab_size)
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # greedy decoding

                        generated.append(next_token)
                        past_key_values = output.past_key_values
                    
                        if next_token.item() == tokenizer.eos_token_id:
                            break
                        
                        cur_input = next_token

                    generated_ids = torch.cat(generated, dim=-1)  # (1, T)
                    rets.append([tokenizer.decode(generated_ids[0], skip_special_tokens=True)])
                
                # Save the results
                for (pred, gt) in zip(rets, dataset.gt[i]):
                    if isinstance(gt, list):
                        if len(gt) == 1:
                            gt = gt[0]
                    if isinstance(pred, list):
                        if len(pred) == 1:
                            pred = pred[0]
                    scores.append(dataset.metric(pred, gt))
                
            else:
                #rets = llm.generate(**(prompt.to(llm.device)), max_new_tokens=dataset.gen_len, top_p=1.0, temperature=0.0, num_logits_to_keep=1, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                rets = llm.generate(**(prompt.to(llm.device)), max_new_tokens=dataset.gen_len, top_p=1.0, temperature=0.0, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                rets = [tokenizer.decode(rets[0][prompt.input_ids.shape[-1]:], skip_special_tokens=True)]
                for (pred, gt) in zip(rets, dataset.gt[i*bsz:(i+1)*bsz]):
                    if isinstance(gt, list):
                        if len(gt) == 1:
                            gt = gt[0]
                    scores.append(dataset.metric(pred, gt))
            
            progress_bar.update(1)
            avg_score = sum(scores) / len(scores)
            max_gpu_mem = torch.cuda.max_memory_allocated(llm.device)
            progress_bar.set_postfix({'avg_score': avg_score, 'max_memory (GB)': max_gpu_mem / 1024 / 1024 / 1024})

            preds = {
                    "prediction": rets,
                    "ground_truth": dataset.gt[i*bsz:(i+1)*bsz],
                    "correct": scores,
                    "avg_score": avg_score,
                    "rank": self.dist_config.rank,
                }

            with open(output_path, "a", encoding="utf8") as fout:
                fout.write(json.dumps(preds, ensure_ascii=False) + "\n")
            # if self.dist_config.is_distributed:
            #     dist.barrier()

        progress_bar.close()
        avg_score = sum(scores) / len(scores)

        self.all_stats.append(
            {
                'model': llm.config.name_or_path,
                'dataset': dataset.dataset_name,
                'samples': dataset.num_samples,
                f'{setting}': avg_score,
            }
        )
        if self.dist_config.is_distributed:
            dist.barrier()

    def summarize(self, shown_avg=False):
        df = pd.DataFrame(self.all_stats)

        if self.dist_config.is_distributed:
            dist.barrier()
            output = [None for _ in range(self.dist_config.world_size)]
            dist.gather_object(df, output if self.dist_config.master_process else None, dst=0)
            dist.barrier()
            if self.dist_config.master_process:
                df = pd.concat(output)
                # Define the columns you want to calculate the mean for (excluding 'samples')
                setting_columns = [col for col in df.columns if col not in ['model', 'dataset', 'samples']]

                # Calculate the sum for 'samples'
                samples_sum = df.groupby(['model', 'dataset'])['samples'].sum()

                # Define the aggregation dictionary for other settings
                agg_dict = {col: 'mean' for col in setting_columns}

                # Calculate the weighted mean for each setting column based on 'samples'
                weighted_means = df.groupby(['model', 'dataset']).apply(lambda x: pd.Series({
                    col: (x[col] * x['samples']).sum() / x['samples'].sum() for col in setting_columns
                }))

                # Combine the weighted means with the sum of samples
                df = weighted_means.join(samples_sum).reset_index()
        
        if self.dist_config.master_process and shown_avg:
            # add a row for the average
            numeric_columns = df.select_dtypes(include='number')
            mean_values = numeric_columns.mean()
            mean_row = pd.DataFrame({col: [mean_values[col] if col in mean_values else 'mean'] for col in df.columns})
            df_with_mean = pd.concat([df, mean_row], ignore_index=True)
            print(df_with_mean.to_markdown(index=False))
            
        return df