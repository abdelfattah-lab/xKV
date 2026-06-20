# =============================================================================
# LongBench · Selective Reconstruction · meta-llama/Meta-Llama-3.1-8B-Instruct
# =============================================================================
mkdir -p logs/longbench_llama3

# ────────────────────────────────────────────────────────────────────────
# XKEY
# ────────────────────────────────────────────────────────────────────────

# ── rank 64 · ShadowKV 1.68/10.61 ────────────────────────────────────────
# ShadowKV baseline (gs=1 ≡ original ShadowKV); Comp 1.68/10.61
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p \
    --method xk_sr --sparse_budget 2048 --chunk_size 8 --layer_group_size 1 --rank_k 64 | tee -a logs/longbench_llama3/xk-sr-1_rank-64_sparse-2048.log

# xK-SR gs=2
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p \
    --method xk_sr --sparse_budget 2048 --chunk_size 8 --layer_group_size 2 --rank_k 128 | tee -a logs/longbench_llama3/xk-sr-2_rank-128_sparse-2048.log

# xK-SR gs=4
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p \
    --method xk_sr --sparse_budget 2048 --chunk_size 8 --layer_group_size 4 --rank_k 256 | tee -a logs/longbench_llama3/xk-sr-4_rank-256_sparse-2048.log

# ── rank 96 · ShadowKV 1.64/9.08 · xK-SR 1.63/8.90 (Table 7) ───────────
# ShadowKV baseline (gs=1 ≡ original ShadowKV); Comp 1.64/9.08
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p \
    --method xk_sr --sparse_budget 2048 --chunk_size 8 --layer_group_size 1 --rank_k 96 | tee -a logs/longbench_llama3/xk-sr-1_rank-96_sparse-2048.log

# xK-SR (Table 7) — gs=4; Comp 1.63/8.90
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p \
    --method xk_sr --sparse_budget 2048 --chunk_size 8 --layer_group_size 4 --rank_k 384 | tee -a logs/longbench_llama3/xk-sr-4_rank-384_sparse-2048.log

# ── rank 128 · ShadowKV 1.60/7.94 ───────────────────────────────────────
# ShadowKV baseline (gs=1 ≡ original ShadowKV); Comp 1.60/7.94
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p \
    --method xk_sr --sparse_budget 2048 --chunk_size 8 --layer_group_size 1 --rank_k 128 | tee -a logs/longbench_llama3/xk-sr-1_rank-128_sparse-2048.log

# xK-SR gs=2
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p \
    --method xk_sr --sparse_budget 2048 --chunk_size 8 --layer_group_size 2 --rank_k 256 | tee -a logs/longbench_llama3/xk-sr-2_rank-256_sparse-2048.log

# xK-SR gs=4
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p \
    --method xk_sr --sparse_budget 2048 --chunk_size 8 --layer_group_size 4 --rank_k 512 | tee -a logs/longbench_llama3/xk-sr-4_rank-512_sparse-2048.log

# ────────────────────────────────────────────────────────────────────────
# XKV
# ────────────────────────────────────────────────────────────────────────

# ── rank 96 · sparse budget 2048 ─────────────────────────────────────────
# ShadowKV‡ baseline (gs=1, single-layer SVD on K+V)
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p \
    --method xkv_sr --sparse_budget 2048 --layer_group_size 1 --rank_k 96 --rank_v 144 --chunk_size 8 | tee -a logs/longbench_llama3/shadowkv_dagger_k96_v144_sparse-2048.log

# xKV-SR gs=2
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p \
    --method xkv_sr --sparse_budget 2048 --layer_group_size 2 --rank_k 192 --rank_v 288 --chunk_size 8 | tee -a logs/longbench_llama3/xkv-sr-2_k192_v288_sparse-2048.log

# xKV-SR gs=4
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p \
    --method xkv_sr --sparse_budget 2048 --layer_group_size 4 --rank_k 384 --rank_v 576 --chunk_size 8 | tee -a logs/longbench_llama3/xkv-sr-4_k384_v576_sparse-2048.log

# ── rank 96 · sparse budget 1024 · Llama-3.1-8B-Instruct ─────────────────
# ShadowKV‡ baseline (gs=1, single-layer SVD on K+V)
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p \
    --method xkv_sr --sparse_budget 1024 --layer_group_size 1 --rank_k 96 --rank_v 144 --chunk_size 8 | tee -a logs/longbench_llama3/shadowkv_dagger_k96_v144_sparse-1024.log

# xKV-SR gs=2
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p \
    --method xkv_sr --sparse_budget 1024 --layer_group_size 2 --rank_k 192 --rank_v 288 --chunk_size 8 | tee -a logs/longbench_llama3/xkv-sr-2_k192_v288_sparse-1024.log

# xKV-SR gs=4
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p \
    --method xkv_sr --sparse_budget 1024 --layer_group_size 4 --rank_k 384 --rank_v 576 --chunk_size 8 | tee -a logs/longbench_llama3/xkv-sr-4_k384_v576_sparse-1024.log
