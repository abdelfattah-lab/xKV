#!/usr/bin/env python3
"""
Pytest tests for FakeLayerMergingCache integration with optimized fake_svd.
"""

import torch
import sys
import os
import pytest

# Add the xKV module to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from xKV.customized_cache.fake_layer_merge_dynamic_cache import FakeLayerMergingCache
    from xKV.configurations import xKVConfig, LayerGroup
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)

@pytest.fixture
def cuda_device():
    """Fixture to check CUDA availability."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device('cuda')

@pytest.fixture
def test_config():
    """Fixture to create test configuration."""
    layer_groups = [LayerGroup(layers=[0, 1], rank_k=8, rank_v=8)]
    return xKVConfig(
        num_layers=2,
        layer_groups=layer_groups,
        merge_key=True,
        merge_value=True,
        layer_merge_impl='svd'
    )

@pytest.fixture
def cache_instance(test_config):
    """Fixture to create cache instance."""
    return FakeLayerMergingCache(merge_setup=test_config)

@pytest.fixture
def test_tensors(cuda_device):
    """Fixture to create test key-value tensors."""
    torch.manual_seed(42)
    bs, nh, sl, hd = 1, 4, 32, 32
    
    key0 = torch.randn(bs, nh, sl, hd, device=cuda_device)
    value0 = torch.randn(bs, nh, sl, hd, device=cuda_device)
    key1 = torch.randn(bs, nh, sl, hd, device=cuda_device)
    value1 = torch.randn(bs, nh, sl, hd, device=cuda_device)
    
    return key0, value0, key1, value1

def test_integration(cache_instance, test_tensors):
    """Test FakeLayerMergingCache integration with optimized fake_svd."""
    key0, value0, key1, value1 = test_tensors
    
    # Update cache (should trigger merging on layer 1)
    cache_instance.update(key0, value0, layer_idx=0, mode='prefill', re_apply_rope=False)
    cache_instance.update(key1, value1, layer_idx=1, mode='prefill', re_apply_rope=False)
    
    # Verify cache contents
    cached_key0, cached_value0 = cache_instance.key_cache[0], cache_instance.value_cache[0]
    cached_key1, cached_value1 = cache_instance.key_cache[1], cache_instance.value_cache[1]
    
    # Check shapes are preserved
    assert cached_key0.shape == key0.shape
    assert cached_value0.shape == value0.shape
    assert cached_key1.shape == key1.shape
    assert cached_value1.shape == value1.shape
    
    # Check norms to verify processing had an effect
    original_key_norm = (key0.norm() + key1.norm()).item()
    cached_key_norm = (cached_key0.norm() + cached_key1.norm()).item()
    
    # Should have some compression effect
    assert 0.3 < cached_key_norm / original_key_norm < 1.2
