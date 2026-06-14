mkdir -p logs/quant

# xKV-2 (16-bit baseline)
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 --batch_size 1 --flash2 \
    --xKV --merge_k --merge_v --rank_k 192 --rank_v 288 --layer_group_size 2 --start_layer_idx 0 --end_layer_idx -1 \
    | tee -a logs/quant/xkv-2_k192_v288_16bit.log

# xKV-2 4-bit
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 --batch_size 1 --flash2 \
    --xKV --merge_k --merge_v --rank_k 192 --rank_v 288 --layer_group_size 2 --start_layer_idx 0 --end_layer_idx -1 \
    --kv_bits 4 --group_size 32 --hadamard \
    | tee -a logs/quant/xkv-2_k192_v288_4bit.log


# xKV-4 (16-bit baseline)
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 --batch_size 1 --flash2 \
    --xKV --merge_k --merge_v --rank_k 384 --rank_v 576 --layer_group_size 4 --start_layer_idx 0 --end_layer_idx -1 \
    | tee -a logs/quant/xkv-4_k384_v576_16bit.log

# xKV-4 4-bit
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 --batch_size 1 --flash2 \
    --xKV --merge_k --merge_v --rank_k 384 --rank_v 576 --layer_group_size 4 --start_layer_idx 0 --end_layer_idx -1 \
    --kv_bits 4 --group_size 32 --hadamard \
    | tee -a logs/quant/xkv-4_k384_v576_4bit_gs32_quantA.log

# xKV-4 3-bit
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 --batch_size 1 --flash2 \
    --xKV --merge_k --merge_v --rank_k 384 --rank_v 576 --layer_group_size 4 --start_layer_idx 0 --end_layer_idx -1 \
    --kv_bits 3 --group_size 32 --hadamard \
    | tee -a logs/quant/xkv-4_k384_v576_3bit_gs32_quantA.log

# xKV-4 2-bit
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 --batch_size 1 --flash2 \
    --xKV --merge_k --merge_v --rank_k 384 --rank_v 576 --layer_group_size 4 --start_layer_idx 0 --end_layer_idx -1 \
    --kv_bits 2 --group_size 32 --hadamard \
    | tee -a logs/quant/xkv-4_k384_v576_2bit_gs32_quantA.log
