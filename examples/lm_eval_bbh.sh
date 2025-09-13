#!/bin/bash

# =============================================================================
# LM-Eval Evaluations for BIG-Bench Hard
# =============================================================================

# Baseline (Full Attention)
CUDA_VISIBLE_DEVICES=0 python test_lm_eval/eval_with_lm_eval.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --tasks bbh --limit 50 \
    | tee -a logs/lm_eval/bbh_baseline.log

# xKV-1 (Layer group size 1)
CUDA_VISIBLE_DEVICES=0 python test_lm_eval/eval_with_lm_eval.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --tasks bbh --limit 50 \
    --xKV --merge_k --merge_v --rank_k 96 --rank_v 144 --layer_group_size 1 --start_layer_idx 0 --end_layer_idx -1 \
    | tee -a logs/lm_eval/bbh_xKV-1_k96_v144.log

# xKV-2 (Layer group size 2)
CUDA_VISIBLE_DEVICES=0 python test_lm_eval/eval_with_lm_eval.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --tasks bbh --limit 50 \
    --xKV --merge_k --merge_v --rank_k 192 --rank_v 288 --layer_group_size 2 --start_layer_idx 0 --end_layer_idx -1 \
    | tee -a logs/lm_eval/bbh_xKV-2_k192_v288.log

# xKV-4 (Layer group size 4)
CUDA_VISIBLE_DEVICES=0 python test_lm_eval/eval_with_lm_eval.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --tasks bbh --limit 50 \
    --xKV --merge_k --merge_v --rank_k 384 --rank_v 576 --layer_group_size 4 --start_layer_idx 0 --end_layer_idx -1 \
    | tee -a logs/lm_eval/bbh_xKV-4_k384_v576.log

# xKV-2  6x Comp. rate
CUDA_VISIBLE_DEVICES=0 python test_lm_eval/eval_with_lm_eval.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --tasks bbh --limit 50 \
    --xKV --merge_k --merge_v --rank_k 256 --rank_v 384 --layer_group_size 4 --start_layer_idx 0 --end_layer_idx -1 \
    | tee -a logs/lm_eval/bbh_xKV-2_k256_v384.log

# xKV-4  6x Comp. rate
CUDA_VISIBLE_DEVICES=0 python test_lm_eval/eval_with_lm_eval.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --tasks bbh --limit 50 \
    --xKV --merge_k --merge_v --rank_k 512 --rank_v 768 --layer_group_size 4 --start_layer_idx 0 --end_layer_idx -1 \
    | tee -a logs/lm_eval/bbh_xKV-4_k512_v768.log

# =============================================================================
# Baseline Methods Comparison  
# =============================================================================

# MiniCache (SLERP-based approach for layers 16-31)
CUDA_VISIBLE_DEVICES=0 python test_lm_eval/eval_with_lm_eval.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --tasks bbh --limit 50 \
    --xKV --merge_k --merge_v --layer_merge_impl slerp --layer_group_size 2 --start_layer_idx 16 --end_layer_idx 31 \
    | tee -a logs/lm_eval/bbh_minicache.log

# StreamingLLM
CUDA_VISIBLE_DEVICES=0 python test_lm_eval/eval_with_lm_eval.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --tasks bbh --limit 50 \
    --streamingllm \
    | tee -a logs/lm_eval/bbh_streamingllm.log

# SnapKV
CUDA_VISIBLE_DEVICES=0 python test_lm_eval/eval_with_lm_eval.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --tasks bbh --limit 50 \
    --snapKV \
    | tee -a logs/lm_eval/bbh_snapkv.log

# PyramidKV
CUDA_VISIBLE_DEVICES=0 python test_lm_eval/eval_with_lm_eval.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --tasks bbh --limit 50 \
    --pyramidkv \
    | tee -a logs/lm_eval/bbh_pyramidkv.log

# KIVI
CUDA_VISIBLE_DEVICES=0 python test_lm_eval/eval_with_lm_eval.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --tasks bbh --limit 50 \
    --kivi \
    | tee -a logs/lm_eval/bbh_kivi-128.log

# Quest
CUDA_VISIBLE_DEVICES=0 python test_lm_eval/eval_with_lm_eval.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --tasks bbh --limit 50 \
    --quest \
    | tee -a logs/lm_eval/bbh_quest-64.log
