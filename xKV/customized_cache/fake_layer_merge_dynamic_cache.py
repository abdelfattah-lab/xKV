import torch
from transformers.cache_utils import DynamicCache
import gc

from transformers.models.mistral.modeling_mistral import (
    apply_rotary_pos_emb,
)

from ..configurations import xKVConfig

def fake_svd(tensor, rank):
    """Perform fake SVD: SVD -> Truncate -> Multiply back."""
    bs, nh, sl, hd = tensor.shape
    tensor_reshaped = tensor.transpose(1, 2).reshape(bs, sl, nh * hd)
    
    # Step 1: Perform SVD NOTE(brian1009): Have deterministic issue but faster
    #U_trunc, S_trunc, V_trunc = torch.svd_lowrank(tensor_reshaped, q=rank)
    #Vt_trunc = V_trunc.transpose(1, 2)
    
    U, S, V_h = torch.linalg.svd(tensor_reshaped, full_matrices=False)
    U_trunc = U[:, :, :rank]
    S_trunc = S[:, :rank]
    Vt_trunc = V_h[:, :rank, :]
    
    # Step 2: Multiply back to approximate the original tensor
    approx_tensor = torch.matmul(U_trunc, torch.matmul(torch.diag_embed(S_trunc), Vt_trunc)) # (bs, sl, nh * hd)
    approx_tensor = approx_tensor.view(bs, sl, nh, hd).transpose(1, 2)
    
    return approx_tensor


def slerp_merge_rows_batch(X1, X2, t=0.5, gamma=0.05):
    """
    Vectorized row-wise SLERP merge of X1, X2 in shape (L, d), returning E of shape (L, d).
    - If a row is zero-norm in either X1 or X2, or if the angle is extremely small,
    we revert to linear interpolation to avoid numerical issues.

    The SLERP formula per row i is:
        e[i] = sin((1-t)*Omega[i]) / sin(Omega[i]) * (x1[i]/||x1[i]||)
            + sin(t*Omega[i])     / sin(Omega[i]) * (x2[i]/||x2[i]||),
    where Omega[i] = arccos( (x1[i] dot x2[i]) / (||x1[i]|| ||x2[i]||) ).
    """

    
    
    # 1. Compute row-wise norms => shape (L,1)
    norm1 = X1.norm(dim=1, keepdim=True)
    norm2 = X2.norm(dim=1, keepdim=True)


    # 3. Normalized vectors
    u1 = X1 / norm1
    u2 = X2 / norm2

    # 4. Dot product => shape (L,1)
    dot_val = (u1 * u2).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)

    # 5. Angle and sine
    Omega = torch.acos(dot_val)  # shape (L,1)
    sinOmega = torch.sin(Omega)

    # s6. Divergence threshold
    d_min = Omega.min()
    d_max = Omega.max()
    threshold = d_min + (d_max - d_min) * gamma
    diverge_mask = Omega > threshold
    
    # near-parallel => angle < eps => linear fallback
    parallel_mask = (Omega < 1e-7)

    # 6. Compute SLERP coefficients
    #    alpha[i] = sin((1-t)*Omega[i]) / sin(Omega[i])
    #    beta[i]  = sin(t*Omega[i])     / sin(Omega[i])
    alpha = torch.sin((1.0 - t) * Omega) / sinOmega
    beta  = torch.sin(t * Omega)         / sinOmega

    # 7. SLERP vector (on the unit sphere)
    E_slerp = alpha * u1 + beta * u2  # shape (L, d)

    # 8. Fallback: Linear interpolation for zero-norm or near-parallel rows
    E_linear = (1.0 - t) * X1 + t * X2  # shape (L, d)
    fallback_mask = parallel_mask  # shape (L,1) => broadcastable

    # 9. Combine results
    #    E[i] = E_slerp[i] if not fallback_mask[i], else E_linear[i]
    # Note: torch.where requires matching shapes, so we expand the mask to (L,d).
    fallback_mask_full = fallback_mask.expand(-1, X1.shape[1])
    E = torch.where(fallback_mask_full, E_linear, E_slerp)

    return E, diverge_mask, norm1, norm2


def fake_minicache_merge(X1, X2, t=0.5, gamma=0.05):
    E, diverge_mask, n1, n2 = slerp_merge_rows_batch(X1, X2, t=t, gamma=gamma)
    diverge_mask = diverge_mask.squeeze(-1)
    E1 = torch.clone(E) * n1
    E1[~diverge_mask] = X1[~diverge_mask]
    E2 = torch.clone(E) * n2
    E2[~diverge_mask] = X2[~diverge_mask]
    return E1, E2


class FakeLayerMergingCache(DynamicCache):
    def __init__(
        self,
        merge_setup: xKVConfig,
    ):
        """Simplified interface: num_heads and head_dim are inferred from the input tensors."""
        super().__init__()
        self.num_layers = merge_setup.num_layers
        self.merge_setup = merge_setup

    def _should_merge(self, layer_idx):
        """Check if this layer is the last in its merge group using dictionary lookup."""
        group_info = self.merge_setup.get_group_for_layer(layer_idx)
        if group_info is not None: # No group found
            last_layer_idx_in_group = group_info.layers[-1]
            return layer_idx == last_layer_idx_in_group
        return False
        
    def is_value_merged(self):
        return self.merge_setup.merge_value

    def is_key_merged(self):
        return self.merge_setup.merge_key

    def update(self, key, value, layer_idx, mode='prefill', cos=None, sin=None, re_apply_rope=True):
        """Override update to hook into the fake SVD compression process."""
        super().update(key, value, layer_idx)

        if mode == 'prefill':
            # Infer num_heads and head_dim from the key shape
            self.num_heads = key.shape[1]  # Shape: (batch_size, num_heads, seq_len, head_dim)
            self.head_dim = key.shape[3]

            # Apply grouped fake SVD if we have updated the last layer in the group
            if self._should_merge(layer_idx):
                self.grouped_layer_merging(layer_idx)
            
            
            
            group_info = self.merge_setup.get_group_for_layer(layer_idx)
            if group_info is not None: # grouped founded      
                if layer_idx == group_info.layers[-1]: # last layer in the group
                    for layer in group_info.layers:
                        pre_rope_key = self.key_cache[layer]
                        if re_apply_rope:
                            _, self.key_cache[layer] = apply_rotary_pos_emb(pre_rope_key, pre_rope_key, cos, sin)
            else: # no group found  
                pre_rope_key = self.key_cache[layer_idx]
                if re_apply_rope:
                    _, self.key_cache[layer_idx] = apply_rotary_pos_emb(pre_rope_key, pre_rope_key, cos, sin)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    @torch.no_grad()
    def grouped_layer_merging(self, last_layer_idx):
        """Perform fake SVD on grouped layers, inferring dimensions from the tensors."""
        group_info = self.merge_setup.get_group_for_layer(last_layer_idx)
        if group_info is None:
            return  # No valid group found
        start_layer_idx, end_layer_idx = group_info.layers[0], group_info.layers[-1]       


        # Step 1: Collect keys and values for the layers in the group            
        keys, values = zip(*[self.__getitem__(i) for i in range(start_layer_idx, end_layer_idx + 1)])
        split_sizes = [self.num_heads for _ in range(start_layer_idx, end_layer_idx + 1)]
        
        if self.merge_setup.layer_merge_impl == 'svd':
            # Step 2: Concatenate along the sequence length dimension
            combined_key = torch.cat(keys, dim=1)  # Shape: (batch_size, total_num_heads, seq_len, head_dim)
            combined_value = torch.cat(values, dim=1)

            # Step 3: Apply fake SVD (truncate and multiply back)
            #NOTE(brian1009): Experiment with fake SVD on key only for now
            if self.merge_setup.merge_key:
                combined_key = fake_svd(combined_key.float(), rank=group_info.rank_k).to(combined_key.dtype)
            if self.merge_setup.merge_value:
                combined_value = fake_svd(combined_value.float(), rank=group_info.rank_v).to(combined_value.dtype)

            # Step 4: Split and update the cache for each layer
            key_layers = torch.split(combined_key, split_sizes, dim=1)
            value_layers = torch.split(combined_value, split_sizes, dim=1)
        elif self.merge_setup.layer_merge_impl == 'svd_kv_joint':
            # Step 2: Concatenate along the sequence length dimension
            combined_kv = torch.cat(keys+values, dim=1)
            # Step 3: Apply fake SVD (truncate and multiply back)
            combined_kv = fake_svd(combined_kv.float(), rank=group_info.rank_kv).to(combined_kv.dtype)
            # Step 4: Split and update the cache for each layer
            combined_layers = torch.split(combined_kv, split_sizes*2, dim=1)
            key_layers = combined_layers[:len(keys)]
            value_layers = combined_layers[len(keys):]
        elif self.merge_setup.layer_merge_impl == 'slerp':
            assert len(keys) == 2 and len(values) == 2, "SLERP only supports group size 2"
            if self.merge_setup.merge_key: 
                key_reshaped = [k.reshape(-1, k.shape[-1]) for k in keys]
                keys_hat1, keys_hat2 = fake_minicache_merge(key_reshaped[0], key_reshaped[1], t=group_info.slerp_t, gamma=group_info.slerp_gamma)
                key_layers = (keys_hat1.reshape(keys[0].shape), keys_hat2.reshape(keys[1].shape))
            else:
                key_layers = keys
            
            if self.merge_setup.merge_value:
                value_reshaped = [v.reshape(-1, v.shape[-1]) for v in values]
                value_hat1, value_hat2 = fake_minicache_merge(value_reshaped[0], value_reshaped[1], t=group_info.slerp_t, gamma=group_info.slerp_gamma)
                value_layers = (value_hat1.reshape(values[0].shape), value_hat2.reshape(values[1].shape))
            else:
                value_layers = values

        else:
            raise NotImplementedError(f"Unknown implementation: {self.merge_setup.layer_merge_impl}")
        
        for idx, layer_idx in enumerate(range(start_layer_idx, end_layer_idx + 1)):
            self.update_cache(layer_idx, key_layers[idx], value_layers[idx])

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
    def update_cache(self, layer_idx, key_approx, value_approx):
        """Update the cache with the approximated key and value tensors."""
        self.key_cache[layer_idx] = key_approx
        self.value_cache[layer_idx] = value_approx
