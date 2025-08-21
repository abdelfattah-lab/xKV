"""
Pytest tests for fake_svd function with Hadamard transform and quantization.
"""

import torch
import pytest
import sys
import os

# Add the xKV module to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from xKV.customized_cache.fake_layer_merge_dynamic_cache import fake_svd
    from xKV.customized_cache.quant import Quantizer
    from xKV.customized_cache.hadamard_utils import apply_hadamard
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


def fused_hadamard_matrix_standalone(A, B):
    """
    Standalone version of fused_hadamard_matrix for testing.
    """
    # Apply Q to A (left multiplication)
    A_transformed = apply_hadamard(A)
    
    # Apply Q^T to B (right multiplication: B @ Q^T)
    B_transformed = apply_hadamard(B.t()).t()
    
    return A_transformed, B_transformed


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def quantizer():
    """Create a quantizer for testing."""
    return Quantizer(n_bits=4, group_size=0, sym=False, clip_ratio=1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestFakeSVD:
    """Test suite for fake_svd function."""
    
    def test_fake_svd_basic(self, device):
        """Test basic fake_svd without Hadamard transform."""
        torch.manual_seed(42)
        
        # Create test tensor: (batch_size, num_heads, seq_len, head_dim)
        bs, nh, sl, hd = 2, 4, 16, 32  # All powers of 2
        rank = 8
        
        tensor = torch.randn(bs, nh, sl, hd, device=device)
        
        # Test without Hadamard transform
        result = fake_svd(tensor, rank, apply_hadamard_transform=False, quantizer=None)
        
        # Check shapes match
        assert result.shape == tensor.shape, f"Shape mismatch: {result.shape} vs {tensor.shape}"
        
        # Verify tensor is on correct device
        assert result.device.type == device.type
    
    def test_fake_svd_with_hadamard(self, device, quantizer):
        """Test fake_svd with Hadamard transform and quantization."""
        torch.manual_seed(42)
        
        # Create test tensor
        bs, nh, sl, hd = 2, 4, 16, 32  # All powers of 2
        rank = 8
        
        tensor = torch.randn(bs, nh, sl, hd, device=device)
        
        # Test with Hadamard transform and quantization
        result = fake_svd(
            tensor, 
            rank, 
            apply_hadamard_transform=True, 
            quantizer=quantizer
        )
        
        # Check shapes match
        assert result.shape == tensor.shape, f"Shape mismatch: {result.shape} vs {tensor.shape}"
        
        # Verify tensor is on correct device
        assert result.device.type == device.type
        
        # Check that norms are different (indicating processing occurred)
        original_norm = tensor.norm().item()
        result_norm = result.norm().item()
        assert original_norm != result_norm, "Norms should be different after processing"
    
    def test_comparison_different_methods(self, device, quantizer):
        """Compare results with and without Hadamard transform."""
        torch.manual_seed(42)
        
        # Create test tensor
        bs, nh, sl, hd = 1, 4, 32, 32
        rank = 16
        tensor = torch.randn(bs, nh, sl, hd, device=device)
        
        # Test basic SVD
        result_basic = fake_svd(tensor, rank, apply_hadamard_transform=False, quantizer=None)
        
        # Test with Hadamard only (no quantizer)
        result_hadamard = fake_svd(tensor, rank, apply_hadamard_transform=True, quantizer=None)
        
        # Test with Hadamard + quantization
        result_hadamard_quant = fake_svd(tensor, rank, apply_hadamard_transform=True, quantizer=quantizer)
        
        # All should have same shape
        assert result_basic.shape == tensor.shape
        assert result_hadamard.shape == tensor.shape  
        assert result_hadamard_quant.shape == tensor.shape
        
        # Norms should be different
        norm_basic = result_basic.norm().item()
        norm_hadamard = result_hadamard.norm().item()
        norm_hadamard_quant = result_hadamard_quant.norm().item()
        
        # Basic and Hadamard-only should be similar (Hadamard is orthogonal)
        assert abs(norm_basic - norm_hadamard) < 0.1 * norm_basic
        
        # Quantized version should be different (but maybe not as much as expected)
        assert abs(norm_basic - norm_hadamard_quant) > 0.001 * norm_basic, f"Expected quantization to cause more difference, got {abs(norm_basic - norm_hadamard_quant)} vs threshold {0.001 * norm_basic}"
    
    def test_standalone_hadamard_function(self, device):
        """Test the standalone Hadamard function."""
        torch.manual_seed(42)
        
        # Create test matrices with power-of-2 dimensions
        A = torch.randn(8, 4, device=device)
        B = torch.randn(4, 16, device=device)
        
        # Compute original product
        original_product = torch.matmul(A, B)
        original_norm = original_product.norm().item()
        
        # Apply Hadamard transform
        A_transformed, B_transformed = fused_hadamard_matrix_standalone(A, B)
        
        # Verify shapes
        assert A_transformed.shape == A.shape
        assert B_transformed.shape == B.shape
        
        # Compute transformed product
        transformed_product = torch.matmul(A_transformed, B_transformed)
        transformed_norm = transformed_product.norm().item()
        
        # Check norm preservation (within tolerance)
        norm_ratio = transformed_norm / original_norm
        assert abs(norm_ratio - 1.0) < 0.1, f"Norm ratio {norm_ratio} should be close to 1.0"
    
    def test_different_tensor_sizes(self, device, quantizer):
        """Test fake_svd with different tensor sizes."""
        test_cases = [
            (1, 2, 8, 16, 4),   # Small
            (2, 4, 16, 32, 8),  # Medium
            (1, 8, 32, 64, 16), # Large
        ]
        
        for bs, nh, sl, hd, rank in test_cases:
            torch.manual_seed(42)
            tensor = torch.randn(bs, nh, sl, hd, device=device)
            
            # Test both with and without Hadamard
            result_basic = fake_svd(tensor, rank, apply_hadamard_transform=False, quantizer=None)
            result_hadamard = fake_svd(tensor, rank, apply_hadamard_transform=True, quantizer=quantizer)
            
            # Check shapes
            assert result_basic.shape == tensor.shape
            assert result_hadamard.shape == tensor.shape
            
            # Check devices
            assert result_basic.device.type == device.type
            assert result_hadamard.device.type == device.type
