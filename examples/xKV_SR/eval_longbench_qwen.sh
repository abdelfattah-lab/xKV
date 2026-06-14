# =============================================================================
# LongBench · Selective Reconstruction · Qwen/Qwen2.5-14B-Instruct-1M
# =============================================================================
mkdir -p logs/longbench_qwen

# ────────────────────────────────────────────────────────────────────────
# XKV
# ────────────────────────────────────────────────────────────────────────

# ── rank 96 · sparse budget 2048 · Qwen2.5-14B-Instruct-1M ──────────────
# ShadowKV‡ baseline (gs=1, single-layer SVD on K+V)
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --model_name_or_path Qwen/Qwen2.5-14B-Instruct-1M --datalen 65536 \
    --dataset_name long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p \
    --method shadowkv_xkv --sparse_budget 2048 --layer_group_size 1 --rank_k 96 --rank_v 144 --chunk_size 8 | tee -a logs/longbench_qwen/shadowkv_dagger_k96_v144_sparse-2048.log

# xKV-SR gs=2
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --model_name_or_path Qwen/Qwen2.5-14B-Instruct-1M --datalen 65536 \
    --dataset_name long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p \
    --method shadowkv_xkv --sparse_budget 2048 --layer_group_size 2 --rank_k 192 --rank_v 288 --chunk_size 8 | tee -a logs/longbench_qwen/xkv-sr-2_k192_v288_sparse-2048.log

# xKV-SR gs=4
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --model_name_or_path Qwen/Qwen2.5-14B-Instruct-1M --datalen 65536 \
    --dataset_name long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p \
    --method shadowkv_xkv --sparse_budget 2048 --layer_group_size 4 --rank_k 384 --rank_v 576 --chunk_size 8 | tee -a logs/longbench_qwen/xkv-sr-4_k384_v576_sparse-2048.log

# ── rank 96 · sparse budget 1024 · Qwen2.5-14B-Instruct-1M ──────────────
# ShadowKV‡ baseline (gs=1, single-layer SVD on K+V)
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --model_name_or_path Qwen/Qwen2.5-14B-Instruct-1M --datalen 65536 \
    --dataset_name long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p \
    --method shadowkv_xkv --sparse_budget 1024 --layer_group_size 1 --rank_k 96 --rank_v 144 --chunk_size 8 | tee -a logs/longbench_qwen/shadowkv_dagger_k96_v144_sparse-1024.log

# xKV-SR gs=2
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --model_name_or_path Qwen/Qwen2.5-14B-Instruct-1M --datalen 65536 \
    --dataset_name long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p \
    --method shadowkv_xkv --sparse_budget 1024 --layer_group_size 2 --rank_k 192 --rank_v 288 --chunk_size 8 | tee -a logs/longbench_qwen/xkv-sr-2_k192_v288_sparse-1024.log

# xKV-SR gs=4
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --model_name_or_path Qwen/Qwen2.5-14B-Instruct-1M --datalen 65536 \
    --dataset_name long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p \
    --method shadowkv_xkv --sparse_budget 1024 --layer_group_size 4 --rank_k 384 --rank_v 576 --chunk_size 8 | tee -a logs/longbench_qwen/xkv-sr-4_k384_v576_sparse-1024.log
