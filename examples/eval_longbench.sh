# =============================================================================
# Main xKV Evaluations
# =============================================================================

# Compression Rate ~2x
# Single SVD
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --batch_size 1 \
    --xKV --merge_k --merge_v --rank_k 320 --rank_v 480 --layer_group_size 1 --start_layer_idx 0 --end_layer_idx -1 \
    --flash2 \
    | tee -a logs/longbench/xKV-1_k320_v480.log

# xKV-2
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --batch_size 1 \
    --xKV --merge_k --merge_v --rank_k 640 --rank_v 960 --layer_group_size 2 --start_layer_idx 0 --end_layer_idx -1 \
    --flash2 \
    | tee -a logs/longbench/xKV-2_k640_v960.log

# xKV-4
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --batch_size 1 \
    --xKV --merge_k --merge_v --rank_k 1280 --rank_v 1920 --layer_group_size 4 --start_layer_idx 0 --end_layer_idx -1 \
    --flash2 \
    | tee -a logs/longbench/xKV-4_k1280_v1920.log


# Compression Rate ~8x
# Single SVD
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --batch_size 1 \
    --xKV --merge_k --merge_v --rank_k 96 --rank_v 144 --layer_group_size 1 --start_layer_idx 0 --end_layer_idx -1 \
    --flash2 \
    | tee -a logs/longbench/xKV-1_k96_v144.log

# xKV-2
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --batch_size 1 \
    --xKV --merge_k --merge_v --rank_k 192 --rank_v 288 --layer_group_size 2 --start_layer_idx 0 --end_layer_idx -1 \
    --flash2 \
    | tee -a logs/longbench/xKV-2_k192_v288.log

# xKV-4
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --batch_size 1 \
    --xKV --merge_k --merge_v --rank_k 384 --rank_v 576 --layer_group_size 4 --start_layer_idx 0 --end_layer_idx -1 \
    --flash2 \
    | tee -a logs/longbench/xKV-4_k384_v576.log

# =============================================================================
# Baseline Methods Comparison
# =============================================================================

# Full Attention
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --batch_size 1 \
    --flash2 \
    | tee -a logs/longbench/full.log

# MiniCache (SLERP-based approach for layers 16-31)
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --batch_size 1 \
    --xKV --merge_k --merge_v --layer_merge_impl slerp --layer_group_size 2 --start_layer_idx 16 --end_layer_idx 31 \
    --flash2 \
    | tee -a logs/longbench/minicache_16-31.log

# MiniCache (SLERP-based approach for layers 20-31)
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --batch_size 1 \
    --xKV --merge_k --merge_v --layer_merge_impl slerp --layer_group_size 2 --start_layer_idx 20 --end_layer_idx 31 \
    --flash2 \
    | tee -a logs/longbench/minicache_20-31.log

# StreamingLLM-840
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --batch_size 1 \
    --flash2 --streamingllm --budget 840 \
    | tee -a logs/longbench/streamingllm-840.log

# SnapKV-840
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --batch_size 1 \
    --flash2 --snapKV --budget 840 \
    | tee -a logs/longbench/snapkv-840.log

# PyramidKV-840
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --batch_size 1 \
    --flash2 --pyramidkv --budget 840 \
    | tee -a logs/longbench/pyramidkv-840.log

# StreamingLLM-1k
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --batch_size 1 \
    --flash2 --streamingllm --budget 1024 \
    | tee -a logs/longbench/streamingllm-1k.log

# SnapKV-1k
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --batch_size 1 \
    --flash2 --snapKV --budget 1024 \
    | tee -a logs/longbench/snapkv-1k.log

# PyramidKV-1k
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --batch_size 1 \
    --flash2 --pyramidkv --budget 1024 \
    | tee -a logs/longbench/pyramidkv-1k.log

# KIVI-gs64
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --batch_size 1 \
    --flash2 --kivi --gs 64 \
    | tee -a logs/longbench/kivi-gs64.log

# KIVI-gs128
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --batch_size 1 \
    --flash2 --kivi --gs 128 \
    | tee -a logs/longbench/kivi-gs128.log

# Quest-1k
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --batch_size 1 \
    --flash2 --quest --budget 1024 \
    | tee -a logs/longbench/quest-1k.log
