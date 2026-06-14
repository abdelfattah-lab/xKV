# =============================================================================
# Table 1 — RULER · Qwen2.5-14B-Instruct-1M · 64K
# Row order matches paper Table 1: Full Attn, KIVI-2, PyramidKV, SnapKV,
#   Single SVD (xKV-1), MiniCache, xKV-4
# =============================================================================
mkdir -p logs/ruler_qwen

DATASET="ruler/niah_single_1,ruler/niah_single_2,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2"
MODEL=Qwen/Qwen2.5-14B-Instruct-1M

# Full Attention
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "$DATASET" --model_name_or_path $MODEL --datalen 65536 --batch_size 1 --flash2 \
    | tee -a logs/ruler_qwen/full.log

# KIVI-2 (comp ~6.40×)
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "$DATASET" --model_name_or_path $MODEL --datalen 65536 --batch_size 1 --flash2 --kivi \
    | tee -a logs/ruler_qwen/kivi.log

# PyramidKV (comp ~6.00×)
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "$DATASET" --model_name_or_path $MODEL --datalen 65536 --batch_size 1 --flash2 --pyramidkv \
    | tee -a logs/ruler_qwen/pyramidkv.log

# SnapKV (comp ~6.00×)
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "$DATASET" --model_name_or_path $MODEL --datalen 65536 --batch_size 1 --flash2 --snapKV \
    | tee -a logs/ruler_qwen/snapkv.log

# Single SVD — xKV-1 (gs=1, comp ~6.35×)
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "$DATASET" --model_name_or_path $MODEL --datalen 65536 --batch_size 1 --flash2 \
    --xKV --merge_k --merge_v --rank_k 96 --rank_v 144 --layer_group_size 1 --start_layer_idx 0 --end_layer_idx -1 \
    | tee -a logs/ruler_qwen/xkv-1_k96_v144.log

# MiniCache (SLERP, layers 16-31, comp ~1.30×)
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "$DATASET" --model_name_or_path $MODEL --datalen 65536 --batch_size 1 --flash2 \
    --xKV --merge_k --merge_v --layer_merge_impl slerp --layer_group_size 2 --start_layer_idx 16 --end_layer_idx 31 \
    | tee -a logs/ruler_qwen/minicache.log

# xKV-4 (gs=4, comp ~6.21×) — Table 1 main Qwen result
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "$DATASET" --model_name_or_path $MODEL --datalen 65536 --batch_size 1 --flash2 \
    --xKV --merge_k --merge_v --rank_k 384 --rank_v 576 --layer_group_size 4 --start_layer_idx 0 --end_layer_idx -1 \
    | tee -a logs/ruler_qwen/xkv-4_k384_v576.log
