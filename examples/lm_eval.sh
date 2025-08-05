#!/bin/bash

# =============================================================================
# LM-Eval Evaluations for GSM8K and BBH 
# =============================================================================

# Baseline (Full Attention)
CUDA_VISIBLE_DEVICES=1 python test_lm_eval/eval_with_lm_eval.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --tasks gsm8k \
    | tee -a logs/lm_eval/gsm8k_baseline.log

# xKV-1 (Layer group size 1)
CUDA_VISIBLE_DEVICES=2 python test_lm_eval/eval_with_lm_eval.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --tasks gsm8k \
    --xKV --merge_k --merge_v --rank_k 96 --rank_v 144 --layer_group_size 1 --start_layer_idx 0 --end_layer_idx -1 \
    | tee -a logs/lm_eval/gsm8k_xKV-1_k96_v144.log

# xKV-2 (Layer group size 2)
CUDA_VISIBLE_DEVICES=2 python test_lm_eval/eval_with_lm_eval.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --tasks gsm8k \
    --xKV --merge_k --merge_v --rank_k 192 --rank_v 288 --layer_group_size 2 --start_layer_idx 0 --end_layer_idx -1 \
    | tee -a logs/lm_eval/gsm8k_xKV-2_k192_v288.log

# xKV-4 (Layer group size 4)
CUDA_VISIBLE_DEVICES=3 python test_lm_eval/eval_with_lm_eval.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --tasks gsm8k \
    --xKV --merge_k --merge_v --rank_k 384 --rank_v 576 --layer_group_size 4 --start_layer_idx 0 --end_layer_idx -1 \
    | tee -a logs/lm_eval/gsm8k_xKV-4_k384_v576.log


# =============================================================================
# Baseline Methods Comparison  
# =============================================================================

# MiniCache (SLERP-based approach for layers 16-31)
CUDA_VISIBLE_DEVICES=1 python test_lm_eval/eval_with_lm_eval.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --tasks gsm8k \
    --xKV --merge_k --merge_v --layer_merge_impl slerp --layer_group_size 2 --start_layer_idx 16 --end_layer_idx 31 \
    | tee -a logs/lm_eval/gsm8k_minicache.log

# StreamingLLM
CUDA_VISIBLE_DEVICES=1 python test_lm_eval/eval_with_lm_eval.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --tasks gsm8k \
    --streamingllm \
    | tee -a logs/lm_eval/gsm8k_streamingllm.log

# SnapKV
CUDA_VISIBLE_DEVICES=4 python test_lm_eval/eval_with_lm_eval.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --tasks gsm8k \
    --snapKV \
    | tee -a logs/lm_eval/gsm8k_snapkv.log

# PyramidKV
CUDA_VISIBLE_DEVICES=5 python test_lm_eval/eval_with_lm_eval.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --tasks gsm8k \
    --pyramidkv \
    | tee -a logs/lm_eval/gsm8k_pyramidkv.log

# KIVI
CUDA_VISIBLE_DEVICES=4 python test_lm_eval/eval_with_lm_eval.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --tasks gsm8k \
    --kivi \
    | tee -a logs/lm_eval/gsm8k_kivi.log

# Quest
CUDA_VISIBLE_DEVICES=5 python test_lm_eval/eval_with_lm_eval.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --tasks gsm8k \
    --quest \
    | tee -a logs/lm_eval/gsm8k_quest.log
