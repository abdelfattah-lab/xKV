#!/usr/bin/env python3
"""
Pytest tests for the enhanced fake_svd function with Hadamard transform and quantization.
"""

import torch
import sys
import os
import pytest

# Add the xKV module to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from xKV.customized_cache.fake_layer_merge_dynamic_cache import fake_svd
    from xKV.customized_cache.quant import Quantizer
    from xKV.customized_cache.hadamard_utils import apply_hadamard
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)

@pytest.fixture
def cuda_device():
    """Fixture to check CUDA availability."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device('cuda')

@pytest.fixture
def test_tensor(cuda_device):
    """Fixture to create a test tensor."""
    torch.manual_seed(42)
    bs, nh, sl, hd = 2, 4, 16, 32  # All powers of 2
    return torch.randn(bs, nh, sl, hd, device=cuda_device)

@pytest.fixture
def quantizer():
    """Fixture to create a quantizer."""
    return Quantizer(n_bits=4, group_size=0, sym=False, clip_ratio=1.0)

def test_fake_svd_basic(test_tensor):
    """Test basic fake_svd without Hadamard transform."""
    rank = 8
    
    result = fake_svd(test_tensor, rank, apply_hadamard_transform=False)
    
    # Check output shape matches input
    assert result.shape == test_tensor.shape
    
    # Check result is reasonable (not NaN or Inf)
    assert torch.isfinite(result).all()

def test_fake_svd_with_hadamard_and_quantization(test_tensor, quantizer):
    """Test fake_svd with Hadamard transform and quantization."""
    rank = 8
    
    result = fake_svd(
        test_tensor, 
        rank, 
        apply_hadamard_transform=True, 
        quantizer=quantizer
    )
    
    # Check output shape matches input
    assert result.shape == test_tensor.shape
    
    # Check result is reasonable (not NaN or Inf)
    assert torch.isfinite(result).all()
    
    # Check that the result is different from input (compression should change values)
    assert not torch.allclose(result, test_tensor, rtol=1e-3)

def test_fake_svd_hadamard_only(test_tensor):
    """Test fake_svd with Hadamard transform but no quantization."""
    rank = 8
    
    # This should not apply Hadamard since quantizer is None
    result = fake_svd(test_tensor, rank, apply_hadamard_transform=True, quantizer=None)
    
    # Check output shape matches input
    assert result.shape == test_tensor.shape
    
    # Check result is reasonable
    assert torch.isfinite(result).all()

def test_fake_svd_rank_effects(test_tensor):
    """Test that different ranks produce different results."""
    rank_small = 4
    rank_large = 12
    
    result_small = fake_svd(test_tensor, rank_small, apply_hadamard_transform=False)
    result_large = fake_svd(test_tensor, rank_large, apply_hadamard_transform=False)
    
    # Results should be different for different ranks
    assert not torch.allclose(result_small, result_large, rtol=1e-3)
    
    # Both should have same shape as input
    assert result_small.shape == test_tensor.shape
    assert result_large.shape == test_tensor.shape

@pytest.mark.parametrize("rank", [4, 8, 16])
def test_fake_svd_different_ranks(test_tensor, rank):
    """Test fake_svd with different rank values."""
    result = fake_svd(test_tensor, rank, apply_hadamard_transform=False)
    
    assert result.shape == test_tensor.shape
    assert torch.isfinite(result).all()

def test_fake_svd_consistency(cuda_device):
    """Test that fake_svd produces consistent results with same seed."""
    bs, nh, sl, hd = 1, 2, 8, 16
    rank = 4
    
    # First run
    torch.manual_seed(123)
    tensor1 = torch.randn(bs, nh, sl, hd, device=cuda_device)
    result1 = fake_svd(tensor1, rank, apply_hadamard_transform=False)
    
    # Second run with same seed
    torch.manual_seed(123)
    tensor2 = torch.randn(bs, nh, sl, hd, device=cuda_device)
    result2 = fake_svd(tensor2, rank, apply_hadamard_transform=False)
    
    # Results should be identical for same input
    assert torch.allclose(tensor1, tensor2)
    assert torch.allclose(result1, result2, rtol=1e-5)
