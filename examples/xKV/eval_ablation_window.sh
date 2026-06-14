# =============================================================================
# Table 8 Ablation: Cross-layer Window Size
# Model: Llama-3.1-8B-Instruct, RULER 64K
# Rank scales linearly with window size (fixed compression rate):
#   base rank (rK_pre, rV) = (96, 144); window W -> rank = (96*W, 144*W)
#   xK-SR: rank_k_pre = 96*W
# All SR runs use sparse_budget=2048, chunk_size=8
# =============================================================================
mkdir -p logs/ablation

DATASET="ruler/niah_single_1,ruler/niah_single_2,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2"
MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct

# -----------------------------------------------------------------------------
# xKV (dense reconstruction)
# -----------------------------------------------------------------------------

# window=1
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "$DATASET" --model_name_or_path $MODEL --datalen 65536 --batch_size 1 --flash2 \
    --xKV --merge_k --merge_v --rank_k 96 --rank_v 144 --layer_group_size 1 --start_layer_idx 0 --end_layer_idx -1 \
    | tee -a logs/ablation/xkv_w1_k96_v144.log

# window=2
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "$DATASET" --model_name_or_path $MODEL --datalen 65536 --batch_size 1 --flash2 \
    --xKV --merge_k --merge_v --rank_k 192 --rank_v 288 --layer_group_size 2 --start_layer_idx 0 --end_layer_idx -1 \
    | tee -a logs/ablation/xkv_w2_k192_v288.log

# window=4 (default, same as Table 1)
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "$DATASET" --model_name_or_path $MODEL --datalen 65536 --batch_size 1 --flash2 \
    --xKV --merge_k --merge_v --rank_k 384 --rank_v 576 --layer_group_size 4 --start_layer_idx 0 --end_layer_idx -1 \
    | tee -a logs/ablation/xkv_w4_k384_v576.log

# window=8
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "$DATASET" --model_name_or_path $MODEL --datalen 65536 --batch_size 1 --flash2 \
    --xKV --merge_k --merge_v --rank_k 768 --rank_v 1152 --layer_group_size 8 --start_layer_idx 0 --end_layer_idx -1 \
    | tee -a logs/ablation/xkv_w8_k768_v1152.log

# -----------------------------------------------------------------------------
# xK-SR (keys compressed, values offloaded, sparse reconstruction)
# -----------------------------------------------------------------------------

# window=1
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "$DATASET" --model_name_or_path $MODEL --datalen 65536 \
    --method shadowkv_xkey --sparse_budget 2048 --layer_group_size 1 --rank_k 96 --chunk_size 8 \
    | tee -a logs/ablation/xk-sr_w1_k96.log

# window=2
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "$DATASET" --model_name_or_path $MODEL --datalen 65536 \
    --method shadowkv_xkey --sparse_budget 2048 --layer_group_size 2 --rank_k 192 --chunk_size 8 \
    | tee -a logs/ablation/xk-sr_w2_k192.log

# window=4 (default, same as Table 2)
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "$DATASET" --model_name_or_path $MODEL --datalen 65536 \
    --method shadowkv_xkey --sparse_budget 2048 --layer_group_size 4 --rank_k 384 --chunk_size 8 \
    | tee -a logs/ablation/xk-sr_w4_k384.log

# window=8
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "$DATASET" --model_name_or_path $MODEL --datalen 65536 \
    --method shadowkv_xkey --sparse_budget 2048 --layer_group_size 8 --rank_k 768 --chunk_size 8 \
    | tee -a logs/ablation/xk-sr_w8_k768.log

# -----------------------------------------------------------------------------
# xKV-SR (both K+V compressed, sparse reconstruction)
# -----------------------------------------------------------------------------

# window=1
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "$DATASET" --model_name_or_path $MODEL --datalen 65536 \
    --method shadowkv_xkv --sparse_budget 2048 --layer_group_size 1 --rank_k 96 --rank_v 144 --chunk_size 8 \
    | tee -a logs/ablation/xkv-sr_w1_k96_v144.log

# window=2
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "$DATASET" --model_name_or_path $MODEL --datalen 65536 \
    --method shadowkv_xkv --sparse_budget 2048 --layer_group_size 2 --rank_k 192 --rank_v 288 --chunk_size 8 \
    | tee -a logs/ablation/xkv-sr_w2_k192_v288.log

# window=4 (default, same as Table 2)
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "$DATASET" --model_name_or_path $MODEL --datalen 65536 \
    --method shadowkv_xkv --sparse_budget 2048 --layer_group_size 4 --rank_k 384 --rank_v 576 --chunk_size 8 \
    | tee -a logs/ablation/xkv-sr_w4_k384_v576.log

# window=8
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "$DATASET" --model_name_or_path $MODEL --datalen 65536 \
    --method shadowkv_xkv --sparse_budget 2048 --layer_group_size 8 --rank_k 768 --rank_v 1152 --chunk_size 8 \
    | tee -a logs/ablation/xkv-sr_w8_k768_v1152.log
