#include <torch/extension.h>

void gather_copy(
    torch::Tensor values, torch::Tensor v_cache_buffer, torch::Tensor position_ids,
    int64_t batch_size, int64_t heads, int64_t cpu_v_length, int64_t gpu_v_length, int64_t map_size);

void gather_copy_d2d_with_offsets(
    torch::Tensor keys,             // gpu keys
    torch::Tensor offsets,          // input, offsets computed from reorder_keys_and_compute_offsets, size as elements (numBlocks*256)
    torch::Tensor cnts,             // input, counts computed from reorder_keys_and_compute_offsets, size as numBlocks
    int64_t batch_size, int64_t heads, 
    int64_t gpu_k_length, 
    int64_t gpu_k_offset, 
    int64_t gpu_k_stride, 
    int64_t map_size);

void reorder_keys_and_compute_offsets(
    torch::Tensor cached_pos_ids, // inout, as cached previous position id as input, also reordered position ids, int64_t type
    torch::Tensor cur_pos_ids,    // input, incoming position id, int64_t type
    torch::Tensor offsets,        // output, offsets for gather_copy_with_offsets, size as numBlocks
    torch::Tensor cnts,           // output, counts to separate d2d and h2d, size as numBlocks
    int64_t batch_size, int64_t heads, int64_t map_size);

void gather_copy_with_offsets(
    torch::Tensor values,           // input, cpu values
    torch::Tensor v_cache_buffer,   // inout, gpu values
    torch::Tensor temp,             // a temp gpu memory for copy, size same as single layer v_cache_buffer 
    torch::Tensor offsets,          // input, offsets computed from reorder_keys_and_compute_offsets, size as numBlocks, 
    torch::Tensor cnts,             // input, counts computed from reorder_keys_and_compute_offsets, size as numBlocks
    torch::Tensor signals,          // extra internal signals, all zeros sizes as numBlocks, size as numBlocks
    int64_t batch_size, int64_t heads, int64_t cpu_v_length, int64_t gpu_v_length, int64_t gpu_v_offset, int64_t gpu_v_stride, int64_t map_size);

void apply_rotary_pos_emb(
    torch::Tensor x, torch::Tensor cos, torch::Tensor sin, torch::Tensor position_ids, torch::Tensor output,
    int64_t batch_size, int64_t heads, int64_t seq_len, int64_t embed_dim,
    int64_t stride_xb, int64_t stride_xh, int64_t stride_xs, int64_t stride_xe,
    int64_t stride_cos, int64_t stride_sin,
    int64_t stride_pid_b, int64_t stride_pid_h, int64_t stride_pid_s,
    int64_t half_dim);

void apply_rotary_pos_emb_new(
    torch::Tensor x, torch::Tensor cos_sin, torch::Tensor position_ids, torch::Tensor output,
    int64_t batch_size, int64_t heads, int64_t seq_len, int64_t embed_dim,
    int64_t stride_xb, int64_t stride_xh, int64_t stride_xs, int64_t stride_xe,
    int64_t stride_cos_sin,
    int64_t stride_pid_b, int64_t stride_pid_h, int64_t stride_pid_s,
    int64_t half_dim);

void apply_rotary_pos_emb_new_v2(
    torch::Tensor x, torch::Tensor cos_sin, torch::Tensor position_ids, torch::Tensor output,
    int64_t batch_size, int64_t heads, int64_t seq_len, int64_t embed_dim,
    int64_t stride_xb, int64_t stride_xh, int64_t stride_xs, int64_t stride_xe,
    int64_t stride_cos_sin,
    int64_t stride_pid_b, int64_t stride_pid_h, int64_t stride_pid_s,
    int64_t half_dim, int64_t chunk_size);

void apply_rotary_pos_emb_push_cache(
    torch::Tensor x, torch::Tensor cos_sin, torch::Tensor position_ids, torch::Tensor output,
    torch::Tensor cnts,
    int64_t batch_size, int64_t heads, int64_t seq_len, int64_t embed_dim,
    int64_t stride_xb, int64_t stride_xh, int64_t stride_xs, int64_t stride_xe,
    int64_t stride_cos_sin,
    int64_t stride_pid_b, int64_t stride_pid_h, int64_t stride_pid_s,
    int64_t stride_output_b, int64_t stride_output_h, int64_t stride_output_s,
    int64_t offset_output_s_start, int64_t offset_output_s_end,
    int64_t half_dim, int64_t chunk_size);

void apply_rotary_pos_emb_push_cache_opt(
    torch::Tensor x, torch::Tensor cos_sin, torch::Tensor position_ids, torch::Tensor output,
    torch::Tensor cnts,
    int64_t batch_size, int64_t heads, int64_t seq_len, int64_t embed_dim,
    int64_t stride_xb, int64_t stride_xh, int64_t stride_xs, int64_t stride_xe,
    int64_t stride_cos_sin,
    int64_t stride_pid_b, int64_t stride_pid_h, int64_t stride_pid_s,
    int64_t stride_output_b, int64_t stride_output_h, int64_t stride_output_s,
    int64_t offset_output_s_start, int64_t offset_output_s_end,
    int64_t half_dim, int64_t chunk_size);

void apply_rotary_pos_emb_push_cache_opt_glm(
    torch::Tensor x, torch::Tensor cos_sin, torch::Tensor position_ids, torch::Tensor output,
    torch::Tensor cnts,
    int64_t batch_size, int64_t heads, int64_t seq_len, int64_t embed_dim,
    int64_t stride_xb, int64_t stride_xh, int64_t stride_xs, int64_t stride_xe,
    int64_t stride_cos_sin,
    int64_t stride_pid_b, int64_t stride_pid_h, int64_t stride_pid_s,
    int64_t stride_output_b, int64_t stride_output_h, int64_t stride_output_s,
    int64_t offset_output_s_start, int64_t offset_output_s_end,
    int64_t half_dim, int64_t chunk_size);

void batch_gather_gemm(
    torch::Tensor a, torch::Tensor b,
    torch::Tensor cos, torch::Tensor sin,
    torch::Tensor position_ids,
    torch::Tensor output,
    int64_t batch_size, int64_t heads, int64_t seq_len, int64_t embed_dim, int64_t rank, int64_t sparse_budget,
    int64_t max_seq_len, int64_t chunk_size, torch::Tensor offset_array);

void batch_gemm_softmax(torch::Tensor A, torch::Tensor B,
                        torch::Tensor D, torch::Tensor Norm, torch::Tensor Sum,
                        torch::Tensor Softmax, int64_t batch_count, int64_t m, int64_t n,
                        int64_t k, double alpha, double beta);
