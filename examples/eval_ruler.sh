# =============================================================================
# Main xKV Evaluations
# =============================================================================

# Single SVD (Layer group size 1)
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 --batch_size 1 --flash2 \
    --xKV --merge_k --merge_v --rank_k 96 --rank_v 144 --layer_group_size 1 --start_layer_idx 0 --end_layer_idx -1 \
    | tee -a logs/xKV-1_k96_v144.log

# xKV-2 (Layer group size 2)
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 --batch_size 1 --flash2 \
    --xKV --merge_k --merge_v --rank_k 192 --rank_v 288 --layer_group_size 2 --start_layer_idx 0 --end_layer_idx -1 \
    | tee -a logs/xKV-2_k192_v288.log

# xKV-4 (Layer group size 4) 
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 --batch_size 1 --flash2 \
    --xKV --merge_k --merge_v --rank_k 384 --rank_v 576 --layer_group_size 4 --start_layer_idx 0 --end_layer_idx -1 \
    | tee -a logs/xKV-4_k384_v576.log

# =============================================================================
# Baseline Methods Comparison
# =============================================================================

# Full Attention
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 --batch_size 1 --flash2 \
    | tee -a logs/full.log

# MiniCache (SLERP-based approach for layers 16-31)
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 --batch_size 1 --flash2 \
    --xKV --merge_k --merge_v --layer_merge_impl slerp --layer_group_size 1 --start_layer_idx 16 --end_layer_idx 31 \
    | tee -a logs/minicache.log

# StreamingLLM
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 --batch_size 1 --flash2 --streamingllm \
    | tee -a logs/streamingllm.log

# SnapKV
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 --batch_size 1 --flash2 --snapKV \
    | tee -a logs/snapkv.log

# PyramidKV
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 --batch_size 1 --flash2 --pyramidkv \
    | tee -a logs/pyramidkv.log

# KIVI
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 --batch_size 1 --flash2 --kivi \
    | tee -a logs/kivi.log

# Quest
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 --batch_size 1 --flash2 --quest \
    | tee -a logs/quest.log
