# =============================================================================
# Main xKV Evaluations
# =============================================================================

# Single SVD (Layer group size 1) - Datasets with chat template
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "long_bench/qasper,long_bench/hotpotqa,long_bench/musique,long_bench/multifieldqa_en,long_bench/gov_report" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 --batch_size 1 \
    --xKV --merge_k --merge_v --rank_k 96 --rank_v 144 --layer_group_size 1 --start_layer_idx 0 --end_layer_idx -1 \
    --flash2 --use_chat_template 
    | tee -a logs/xKV-1_k96_v144.log

# Single SVD (Layer group size 1) - Datasets without chat template
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "long_bench/repobench-p,long_bench/lcc" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 --batch_size 1 \
    --xKV --merge_k --merge_v --rank_k 96 --rank_v 144 --layer_group_size 1 --start_layer_idx 0 --end_layer_idx -1 \
    --flash2 
    | tee -a logs/xKV-1_k96_v144.log

# xKV-2 (Layer group size 2) - Datasets with chat template
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "long_bench/qasper,long_bench/hotpotqa,long_bench/musique,long_bench/multifieldqa_en,long_bench/gov_report" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 --batch_size 1 \
    --xKV --merge_k --merge_v --rank_k 192 --rank_v 288 --layer_group_size 2 --start_layer_idx 0 --end_layer_idx -1 \
    --flash2 --use_chat_template 
    | tee -a logs/xKV-2_k192_v288.log

# xKV-2 (Layer group size 2) - Datasets without chat template
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "long_bench/repobench-p,long_bench/lcc" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 --batch_size 1 \
    --xKV --merge_k --merge_v --rank_k 192 --rank_v 288 --layer_group_size 2 --start_layer_idx 0 --end_layer_idx -1 \
    --flash2 
    | tee -a logs/xKV-2_k192_v288.log

# xKV-4 (Layer group size 4) - Datasets with chat template
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "long_bench/qasper,long_bench/hotpotqa,long_bench/musique,long_bench/multifieldqa_en,long_bench/gov_report" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 --batch_size 1 \
    --xKV --merge_k --merge_v --rank_k 384 --rank_v 576 --layer_group_size 4 --start_layer_idx 0 --end_layer_idx -1 \
    --flash2 --use_chat_template 
    | tee -a logs/xKV-4_k384_v576.log

# xKV-4 (Layer group size 4) - Datasets without chat template
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "long_bench/repobench-p,long_bench/lcc" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 --batch_size 1 \
    --xKV --merge_k --merge_v --rank_k 384 --rank_v 576 --layer_group_size 4 --start_layer_idx 0 --end_layer_idx -1 \
    --flash2 
    | tee -a logs/xKV-4_k384_v576.log

# =============================================================================
# Baseline Methods Comparison
# =============================================================================

# Full Attention - Datasets with chat template
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "long_bench/qasper,long_bench/hotpotqa,long_bench/musique,long_bench/multifieldqa_en,long_bench/gov_report" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 --batch_size 1 \
    --flash2 --use_chat_template \
    | tee -a logs/full.log

# Full Attention - Datasets without chat template
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "long_bench/repobench-p,long_bench/lcc" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 --batch_size 1 \
    --flash2 \
    | tee -a logs/full.log

# MiniCache (SLERP-based approach for layers 16-31) - Datasets with chat template
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "long_bench/qasper,long_bench/hotpotqa,long_bench/musique,long_bench/multifieldqa_en,long_bench/gov_report" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 --batch_size 1 \
    --xKV --merge_k --merge_v --layer_merge_impl slerp --layer_group_size 1 --start_layer_idx 16 --end_layer_idx 31 \
    --flash2 --use_chat_template 
    | tee -a logs/minicache.log

# MiniCache (SLERP-based approach for layers 16-31) - Datasets without chat template
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "long_bench/repobench-p,long_bench/lcc" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 --batch_size 1 \
    --xKV --merge_k --merge_v --layer_merge_impl slerp --layer_group_size 1 --start_layer_idx 16 --end_layer_idx 31 \
    --flash2 
    | tee -a logs/minicache.log

# StreamingLLM - Datasets with chat template
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "long_bench/qasper,long_bench/hotpotqa,long_bench/musique,long_bench/multifieldqa_en,long_bench/gov_report" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 --batch_size 1 \
    --flash2 --use_chat_template --streamingllm \
    | tee -a logs/streamingllm.log

# StreamingLLM - Datasets without chat template
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "long_bench/repobench-p,long_bench/lcc" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 --batch_size 1 \
    --flash2 --streamingllm \
    | tee -a logs/streamingllm.log

# SnapKV - Datasets with chat template
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "long_bench/qasper,long_bench/hotpotqa,long_bench/musique,long_bench/multifieldqa_en,long_bench/gov_report" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 --batch_size 1 \
    --flash2 --use_chat_template --snapKV \
    | tee -a logs/snapkv.log

# SnapKV - Datasets without chat template
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "long_bench/repobench-p,long_bench/lcc" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 --batch_size 1 \
    --flash2 --snapKV \
    | tee -a logs/snapkv.log

# PyramidKV - Datasets with chat template
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "long_bench/qasper,long_bench/hotpotqa,long_bench/musique,long_bench/multifieldqa_en,long_bench/gov_report" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 --batch_size 1 \
    --flash2 --use_chat_template --pyramidkv \
    | tee -a logs/pyramidkv.log

# PyramidKV - Datasets without chat template
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "long_bench/repobench-p,long_bench/lcc" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 --batch_size 1 \
    --flash2 --pyramidkv \
    | tee -a logs/pyramidkv.log

# KIVI - Datasets with chat template
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "long_bench/qasper,long_bench/hotpotqa,long_bench/musique,long_bench/multifieldqa_en,long_bench/gov_report" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 --batch_size 1 \
    --flash2 --use_chat_template --kivi \
    | tee -a logs/kivi.log

# KIVI - Datasets without chat template
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "long_bench/repobench-p,long_bench/lcc" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 --batch_size 1 \
    --flash2 --kivi \
    | tee -a logs/kivi.log

# Quest - Datasets with chat template
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "long_bench/qasper,long_bench/hotpotqa,long_bench/musique,long_bench/multifieldqa_en,long_bench/gov_report" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 --batch_size 1 \
    --flash2 --use_chat_template --quest \
    | tee -a logs/quest.log

# Quest - Datasets without chat template
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "long_bench/repobench-p,long_bench/lcc" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 --batch_size 1 \
    --flash2 --quest \
    | tee -a logs/quest.log
