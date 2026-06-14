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


def _is_custom_llm(llm):
    """True for xKV-SR custom engine, False for HF model."""
    return hasattr(llm, 'model_name')


def _model_name(llm):
    return llm.model_name if _is_custom_llm(llm) else llm.config.name_or_path


def _hf_generate(llm, tokenizer, prompt, gen_len, **kwargs):
    """Run HF model.generate and return decoded string list."""
    rets = llm.generate(
        **(prompt.to(llm.device)),
        max_new_tokens=gen_len,
        top_p=1.0, temperature=0.0,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        **kwargs,
    )
    return [tokenizer.decode(rets[0][prompt.input_ids.shape[-1]:], skip_special_tokens=True)]


class Evaluator:
    def __init__(self, dist_config):
        self.dist_config = dist_config
        self.all_stats = []

    def test(self, llm, tokenizer, dataset: Dataset, output_path: str, setting: str = 'baseline'):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if self.dist_config.master_process:
            print(colored(f"[Test] {_model_name(llm)} on {dataset.dataset_name}, results saved to {output_path}", 'green'))

        if not dataset.is_sharded:
            dataset.shard(self.dist_config.rank, self.dist_config.world_size)

        bsz = llm.batch_size if _is_custom_llm(llm) else 1
        scores = []

        open(output_path, 'w').close()
        if self.dist_config.is_distributed:
            dist.barrier()

        progress_bar = tqdm(range(dataset.num_samples // bsz), desc='Testing',
                            disable=self.dist_config.is_distributed and not self.dist_config.master_process)

        for i in range(dataset.num_samples // bsz):
            prompt = dataset.tokenized_prompts[i]
            custom = _is_custom_llm(llm)
            rets = None

            if 'long_bench' in dataset.dataset_name:
                if custom:
                    p = prompt.input_ids if hasattr(prompt, 'input_ids') else prompt
                    rets = llm.generate(p.to(llm.device), gen_len=dataset.gen_len, verbose=False, top_p=1.0, temperature=0.0)
                else:
                    rets = _hf_generate(llm, tokenizer, prompt, dataset.gen_len)
                for pred, gt, classes in zip(rets, dataset.gt[i*bsz:(i+1)*bsz], dataset.classes[i*bsz:(i+1)*bsz]):
                    scores.append(max([dataset.metric(pred, g, all_classes=classes) for g in gt]))

            elif 'ruler' in dataset.dataset_name and custom and dataset.inference_mode == 'multi_turn':
                context = dataset.tokenized_contexts[i]
                query = dataset.tokenized_queries[i]
                ctx_ids = context.input_ids if hasattr(context, 'input_ids') else context
                llm.prefill(ctx_ids.to(llm.device))
                q_ids = query.input_ids if hasattr(query, 'input_ids') else query
                rets = llm.generate(q_ids.to(llm.device), gen_len=dataset.gen_len, cont=True, top_p=1.0, temperature=0.0)
                for pred, gt in zip(rets, dataset.gt[i*bsz:(i+1)*bsz]):
                    if isinstance(gt, list) and len(gt) == 1:
                        gt = gt[0]
                    scores.append(dataset.metric(pred, gt))

            elif 'niah_multiturn' in dataset.dataset_name:
                if custom:
                    p_ids = prompt.input_ids if hasattr(prompt, 'input_ids') else prompt
                    first_ret = llm.generate(p_ids.to(llm.device), gen_len=dataset.gen_len, top_p=1.0, temperature=0.0)
                    rets = [first_ret[0]]
                    for query in dataset.tokenized_queries[i]:
                        q_ids = query.input_ids if hasattr(query, 'input_ids') else query
                        rets.append(llm.generate(q_ids.to(llm.device), cont=True, gen_len=dataset.gen_len, top_p=1.0, temperature=0.0)[0])
                else:
                    first_ret = llm.generate(
                        **(prompt.to(llm.device)),
                        return_dict_in_generate=True,
                        max_new_tokens=dataset.gen_len,
                        top_p=1.0, temperature=0.0,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    past_kv = first_ret.past_key_values
                    first_response = tokenizer.decode(first_ret.sequences[0, prompt.input_ids.shape[-1]:], skip_special_tokens=True)
                    rets = [first_response]
                    for query in dataset.tokenized_queries[i]:
                        cur_input = query["input_ids"].to(llm.device)
                        generated = []
                        for _ in range(10):
                            with torch.no_grad():
                                out = llm(input_ids=cur_input, past_key_values=past_kv, return_dict=True)
                            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                            generated.append(next_tok)
                            past_kv = out.past_key_values
                            if next_tok.item() == tokenizer.eos_token_id:
                                break
                            cur_input = next_tok
                        rets.append(tokenizer.decode(torch.cat(generated, dim=-1)[0], skip_special_tokens=True))
                for pred, gt in zip(rets, dataset.gt[i]):
                    if isinstance(gt, list) and len(gt) == 1:
                        gt = gt[0]
                    if isinstance(pred, list) and len(pred) == 1:
                        pred = pred[0]
                    scores.append(dataset.metric(pred, gt))

            elif 'persona' in dataset.dataset_name and custom:
                llm.generate(prompt.to(llm.device), gen_len=0, verbose=False, top_p=0.9, temperature=0.0)
                queries = dataset.queries[i]
                rets_list = []
                for query, gts in zip(queries, dataset.gt[i]):
                    ret = llm.generate(llm.encode(query, template="chat"), cont=True, gen_len=dataset.gen_len, top_p=0.9, temperature=0.0)
                    rets_list.append(ret)
                    pred = ret[0]
                    if isinstance(gts, list) and len(gts) == 1:
                        gts = gts[0]
                    scores.append(dataset.metric(pred, gts))
                rets = rets_list

            else:
                # Default: single-turn (RULER single_turn, niah, etc.)
                if custom:
                    p = prompt.input_ids if hasattr(prompt, 'input_ids') else prompt
                    rets = llm.generate(p.to(llm.device), gen_len=dataset.gen_len, top_p=1.0, temperature=0.0)
                else:
                    rets = _hf_generate(llm, tokenizer, prompt, dataset.gen_len)
                for pred, gt in zip(rets, dataset.gt[i*bsz:(i+1)*bsz]):
                    if isinstance(gt, list) and len(gt) == 1:
                        gt = gt[0]
                    scores.append(dataset.metric(pred, gt))

            progress_bar.update(1)
            avg_score = sum(scores) / len(scores)
            max_mem = torch.cuda.max_memory_allocated(llm.device if hasattr(llm, 'device') else 0) / 1024**3
            progress_bar.set_postfix({'avg_score': avg_score, 'max_memory (GB)': max_mem})

            if dataset.dataset_name == 'niah':
                preds = {
                    "context_length": dataset.ctx_len[i*bsz:(i+1)*bsz],
                    "depth_percent": dataset.depth_pct[i*bsz:(i+1)*bsz],
                    "response": rets,
                    "answer": dataset.gt[i*bsz:(i+1)*bsz],
                    "correct": scores[i*bsz:(i+1)*bsz],
                    "avg_score": avg_score,
                }
            else:
                preds = {
                    "prediction": rets,
                    "ground_truth": dataset.gt[i*bsz:(i+1)*bsz],
                    "correct": scores,
                    "avg_score": avg_score,
                    "rank": self.dist_config.rank,
                }

            with open(output_path, "a", encoding="utf8") as fout:
                fout.write(json.dumps(preds, ensure_ascii=False) + "\n")

        progress_bar.close()
        avg_score = sum(scores) / len(scores) if scores else 0.0

        self.all_stats.append({
            'model': _model_name(llm),
            'dataset': dataset.dataset_name,
            'samples': dataset.num_samples,
            f'{setting}': avg_score,
        })
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
                setting_columns = [col for col in df.columns if col not in ['model', 'dataset', 'samples']]
                samples_sum = df.groupby(['model', 'dataset'])['samples'].sum()
                weighted_means = df.groupby(['model', 'dataset']).apply(lambda x: pd.Series({
                    col: (x[col] * x['samples']).sum() / x['samples'].sum() for col in setting_columns
                }))
                df = weighted_means.join(samples_sum).reset_index()

        if self.dist_config.master_process:
            numeric_columns = df.select_dtypes(include='number')
            mean_values = numeric_columns.mean()
            mean_row = pd.DataFrame({col: [mean_values[col] if col in mean_values else 'mean'] for col in df.columns})
            df_with_mean = pd.concat([df, mean_row], ignore_index=True)
            print(df_with_mean.to_markdown(index=False))

        return df
