# =============================================================================
# Baseline Methods Comparison
# =============================================================================

# Full Attention
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --batch_size 1 --flash2 \
    | tee -a logs/longbench_llama3_4kup/full.log

# MiniCache (SLERP-based approach for layers 12-27)
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --batch_size 1 \
    --xKV --merge_k --merge_v --layer_merge_impl slerp --layer_group_size 2 --start_layer_idx 12 --end_layer_idx 27 --flash2 \
    | tee -a logs/longbench_llama3_4kup/minicache_12-27.log

# StreamingLLM-1.6k
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --batch_size 1 --flash2 --streamingllm --budget 1600\
    | tee -a logs/longbench_llama3_4kup/streamingllm-1.6k.log

# SnapKV-1.6k
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --batch_size 1 --flash2 --snapKV --budget 1600\
    | tee -a logs/longbench_llama3_4kup/snapkv-1.6k.log

# PyramidKV-1.6k
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --batch_size 1 --flash2 --pyramidkv --budget 1600\
    | tee -a logs/longbench_llama3_4kup/pyramidkv-1.6k.log

# KIVI-gs128
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --batch_size 1 --flash2 --kivi --gs 128 \
    | tee -a logs/longbench_llama3_4kup/kivi-gs128.log

# Quest-2k
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 xkv_evaluate/eval_acc.py \
    --dataset_name "long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --batch_size 1 --flash2 --quest --budget 2048 \
    | tee -a logs/longbench_llama3_4kup/quest-2k.log
