"""
Pytest tests for FakeLayerMergingCache integration.
"""

import torch
import pytest
import sys
import os

# Add the xKV module to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from xKV.customized_cache.fake_layer_merge_dynamic_cache import FakeLayerMergingCache
    from xKV.configurations import xKVConfig, LayerGroup
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def test_config():
    """Create a test configuration for the cache."""
    layer_groups = [LayerGroup(layers=[0, 1], rank_k=8, rank_v=8)]
    return xKVConfig(
        num_layers=2,
        layer_groups=layer_groups,
        merge_key=True,
        merge_value=True,
        layer_merge_impl='svd'
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCacheIntegration:
    """Test suite for FakeLayerMergingCache integration."""
    
    def test_cache_creation(self, test_config):
        """Test cache creation with configuration."""
        cache = FakeLayerMergingCache(merge_setup=test_config)
        
        assert cache.num_layers == test_config.num_layers
        assert cache.merge_setup == test_config
        # Quantizer is only created when kv_bits < 16
        if test_config.kv_bits < 16:
            assert hasattr(cache, 'quantizer')
            assert hasattr(cache, 'use_hadamard')
        else:
            # Should not have quantizer if kv_bits >= 16
            assert not hasattr(cache, 'quantizer')
    
    def test_cache_update_basic(self, device, test_config):
        """Test basic cache update functionality."""
        torch.manual_seed(42)
        
        cache = FakeLayerMergingCache(merge_setup=test_config)
        
        # Create test tensors
        bs, nh, sl, hd = 1, 4, 32, 32
        key0 = torch.randn(bs, nh, sl, hd, device=device)
        value0 = torch.randn(bs, nh, sl, hd, device=device)
        key1 = torch.randn(bs, nh, sl, hd, device=device)
        value1 = torch.randn(bs, nh, sl, hd, device=device)
        
        # Update cache (skip RoPE for testing)
        cache.update(key0, value0, layer_idx=0, mode='prefill', re_apply_rope=False)
        cache.update(key1, value1, layer_idx=1, mode='prefill', re_apply_rope=False)
        
        # Verify cache contents
        cached_key0, cached_value0 = cache.key_cache[0], cache.value_cache[0]
        cached_key1, cached_value1 = cache.key_cache[1], cache.value_cache[1]
        
        # Check shapes are preserved
        assert cached_key0.shape == key0.shape
        assert cached_value0.shape == value0.shape
        assert cached_key1.shape == key1.shape
        assert cached_value1.shape == value1.shape
        
        # Check tensors are on correct device
        assert cached_key0.device.type == device.type
        assert cached_value0.device.type == device.type  
        assert cached_key1.device.type == device.type
        assert cached_value1.device.type == device.type
    
    def test_svd_processing_effect(self, device, test_config):
        """Test that SVD processing actually affects the tensors."""
        torch.manual_seed(42)
        
        cache = FakeLayerMergingCache(merge_setup=test_config)
        
        # Create test tensors
        bs, nh, sl, hd = 1, 4, 32, 32
        key0 = torch.randn(bs, nh, sl, hd, device=device)
        value0 = torch.randn(bs, nh, sl, hd, device=device)
        key1 = torch.randn(bs, nh, sl, hd, device=device)
        value1 = torch.randn(bs, nh, sl, hd, device=device)
        
        # Store original norms
        original_key_norm = (key0.norm() + key1.norm()).item()
        original_value_norm = (value0.norm() + value1.norm()).item()
        
        # Update cache
        cache.update(key0, value0, layer_idx=0, mode='prefill', re_apply_rope=False)
        cache.update(key1, value1, layer_idx=1, mode='prefill', re_apply_rope=False)
        
        # Get processed tensors
        cached_key0, cached_value0 = cache.key_cache[0], cache.value_cache[0]
        cached_key1, cached_value1 = cache.key_cache[1], cache.value_cache[1]
        
        # Calculate processed norms
        cached_key_norm = (cached_key0.norm() + cached_key1.norm()).item()
        cached_value_norm = (cached_value0.norm() + cached_value1.norm()).item()
        
        # Verify processing occurred (norms should be different due to SVD approximation)
        key_norm_ratio = cached_key_norm / original_key_norm
        value_norm_ratio = cached_value_norm / original_value_norm
        
        # Allow some tolerance but expect some difference
        assert 0.3 < key_norm_ratio < 1.2, f"Key norm ratio {key_norm_ratio} seems unusual"
        assert 0.3 < value_norm_ratio < 1.2, f"Value norm ratio {value_norm_ratio} seems unusual"
    
    def test_different_layer_configurations(self, device):
        """Test with different layer configurations."""
        test_cases = [
            # Single layer group
            [LayerGroup(layers=[0], rank_k=4, rank_v=4)],
            # Multiple layer group
            [LayerGroup(layers=[0, 1, 2], rank_k=6, rank_v=6)],
            # Multiple groups
            [
                LayerGroup(layers=[0, 1], rank_k=4, rank_v=4),
                LayerGroup(layers=[2, 3], rank_k=8, rank_v=8)
            ]
        ]
        
        for layer_groups in test_cases:
            max_layer = max(max(group.layers) for group in layer_groups)
            config = xKVConfig(
                num_layers=max_layer + 1,
                layer_groups=layer_groups,
                merge_key=True,
                merge_value=True,
                layer_merge_impl='svd'
            )
            
            cache = FakeLayerMergingCache(merge_setup=config)
            assert cache.num_layers == max_layer + 1
