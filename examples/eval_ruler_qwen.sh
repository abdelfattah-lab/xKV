# =============================================================================
# Main xKV Evaluations
# =============================================================================

# Compression Rate ~2x
# Single SVD
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct-1M --datalen 65536 --batch_size 1 --flash2 \
    --xKV --merge_k --merge_v --rank_k 320 --rank_v 480 --layer_group_size 1 --start_layer_idx 0 --end_layer_idx -1 \
    | tee -a logs/ruler/xKV-1_k320_v480.log

# xKV-2
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct-1M --datalen 65536 --batch_size 1 --flash2 \
    --xKV --merge_k --merge_v --rank_k 640 --rank_v 960 --layer_group_size 2 --start_layer_idx 0 --end_layer_idx -1 \
    | tee -a logs/ruler/xKV-2_k640_v960.log

# xKV-4
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct-1M --datalen 65536 --batch_size 1 --flash2 \
    --xKV --merge_k --merge_v --rank_k 1280 --rank_v 1920 --layer_group_size 4 --start_layer_idx 0 --end_layer_idx -1 \
    | tee -a logs/ruler/xKV-4_k1280_v1920.log

# Compression Rate ~6x
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct-1M --datalen 65536 --batch_size 1 --flash2 \
    --xKV --merge_k --merge_v --rank_k 128 --rank_v 192 --layer_group_size 1 --start_layer_idx 0 --end_layer_idx -1 \
    | tee -a logs/ruler/xKV-1_k128_v192.log

# xKV-2
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct-1M --datalen 65536 --batch_size 1 --flash2 \
    --xKV --merge_k --merge_v --rank_k 256 --rank_v 384 --layer_group_size 2 --start_layer_idx 0 --end_layer_idx -1 \
    | tee -a logs/ruler/xKV-2_k256_v384.log

# xKV-4
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct-1M --datalen 65536 --batch_size 1 --flash2 \
    --xKV --merge_k --merge_v --rank_k 512 --rank_v 768 --layer_group_size 4 --start_layer_idx 0 --end_layer_idx -1 \
    | tee -a logs/ruler/xKV-4_k512_v768.log

# =============================================================================
# Baseline Methods Comparison
# =============================================================================

# Full Attention
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct-1M --datalen 65536 --batch_size 1 --flash2 \
    | tee -a logs/ruler/full.log

# MiniCache (SLERP-based approach for layers 12-27)
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct-1M --datalen 65536 --batch_size 1 --flash2 \
    --xKV --merge_k --merge_v --layer_merge_impl slerp --layer_group_size 2 --start_layer_idx 12 --end_layer_idx 27 \
    | tee -a logs/ruler/minicache_12-27.log

# MiniCache (SLERP-based approach for layers 16-27)
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct-1M --datalen 65536 --batch_size 1 --flash2 \
    --xKV --merge_k --merge_v --layer_merge_impl slerp --layer_group_size 2 --start_layer_idx 16 --end_layer_idx 27 \
    | tee -a logs/ruler/minicache_16-27.log

# StreamingLLM-11k
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct-1M --datalen 65536 --batch_size 1 --flash2 --streamingllm --budget 11000 \
    | tee -a logs/ruler/streamingllm-11k.log

# SnapKV-11k
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct-1M --datalen 65536 --batch_size 1 --flash2 --snapKV --budget 11000 \
    | tee -a logs/ruler/snapkv-11k.log

# PyramidKV-11k
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct-1M --datalen 65536 --batch_size 1 --flash2 --pyramidkv --budget 11000 \
    | tee -a logs/ruler/pyramidkv-11k.log

# KIVI-gs64
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct-1M --datalen 65536 --batch_size 1 --flash2 --kivi --gs 64 \
    | tee -a logs/ruler/kivi-gs64.log

# KIVI-gs128
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct-1M --datalen 65536 --batch_size 1 --flash2 --kivi --gs 128 \
    | tee -a logs/ruler/kivi-gs128.log

# Quest-2k
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct-1M --datalen 65536 --batch_size 1 --flash2 --quest --budget 2048 \
    | tee -a logs/ruler/quest-2k.log
