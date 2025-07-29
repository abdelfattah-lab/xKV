#!/usr/bin/env python3
"""
Test script to verify fake quantization is working properly
"""
import pytest
import torch
import numpy as np
from xKV.customized_cache.quant import Quantizer


class TestQuantizer:
    """Test suite for the Quantizer class"""
    
    def test_quantizer_4bit(self):
        """Test 4-bit quantization functionality"""
        print("Testing 4-bit Quantizer functionality...")
        
        # Create a quantizer with 4-bit quantization
        quantizer = Quantizer(n_bits=4, group_size=0, sym=False, clip_ratio=1.0)
        
        # Create test tensor
        x = torch.randn(2, 128, 32, 128, dtype=torch.float16)  # [bs, seq_len, num_heads, head_dim]
        print(f"Original tensor shape: {x.shape}")
        print(f"Original tensor range: [{x.min():.4f}, {x.max():.4f}]")
        print(f"Original tensor dtype: {x.dtype}")
        
        # Apply fake quantization
        with torch.no_grad():
            x_quantized = quantizer(x)
        
        print(f"Quantized tensor shape: {x_quantized.shape}")
        print(f"Quantized tensor range: [{x_quantized.min():.4f}, {x_quantized.max():.4f}]")
        print(f"Quantized tensor dtype: {x_quantized.dtype}")
        
        # Check if quantization actually happened
        diff = (x - x_quantized).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"Max difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")
        
        # Assertions
        assert x_quantized.shape == x.shape, "Shape should be preserved"
        assert x_quantized.dtype == x.dtype, "Dtype should be preserved"
        assert mean_diff > 0, "4-bit quantization should introduce some error"
        assert mean_diff < 1.0, "Error should be reasonable"
        
        # Check if values are quantized (should have discrete values)
        unique_vals = torch.unique(x_quantized).size(0)
        total_vals = x_quantized.numel()
        print(f"Unique values: {unique_vals}/{total_vals} ({unique_vals/total_vals*100:.2f}%)")
        
        print("✅ 4-bit quantization is working correctly!")
    
    def test_quantizer_16bit_no_change(self):
        """Test that 16-bit quantization doesn't change values"""
        print("Testing 16-bit quantization (should not change values)...")
        
        # Create test tensor
        x = torch.randn(2, 64, 16, 64, dtype=torch.float16)
        
        # Test with no quantization (16-bit)
        quantizer_16bit = Quantizer(n_bits=16, group_size=0, sym=False, clip_ratio=1.0)
        x_16bit = quantizer_16bit(x)
        
        diff_16bit = (x - x_16bit).abs().max().item()
        print(f"16-bit quantization difference: {diff_16bit:.6f}")
        
        # Should be very close to original when n_bits >= 16
        assert diff_16bit < 1e-6, f"16-bit quantization should not change values, but got diff={diff_16bit}"
        
        print("✅ 16-bit quantization preserves original values!")
    
    def test_quantizer_symmetric_vs_asymmetric(self):
        """Test symmetric vs asymmetric quantization"""
        print("Testing symmetric vs asymmetric quantization...")
        
        x = torch.randn(1, 32, 8, 32, dtype=torch.float16)
        
        # Symmetric quantization
        quantizer_sym = Quantizer(n_bits=4, group_size=0, sym=True, clip_ratio=1.0)
        x_sym = quantizer_sym(x)
        
        # Asymmetric quantization
        quantizer_asym = Quantizer(n_bits=4, group_size=0, sym=False, clip_ratio=1.0)
        x_asym = quantizer_asym(x)
        
        # Both should work but give different results
        diff_sym = (x - x_sym).abs().mean().item()
        diff_asym = (x - x_asym).abs().mean().item()
        
        print(f"Symmetric quantization error: {diff_sym:.6f}")
        print(f"Asymmetric quantization error: {diff_asym:.6f}")
        
        assert diff_sym > 0, "Symmetric quantization should introduce error"
        assert diff_asym > 0, "Asymmetric quantization should introduce error"
        
        print("✅ Both symmetric and asymmetric quantization work!")
    
    def test_quantizer_group_size(self):
        """Test different group sizes"""
        print("Testing different group sizes...")
        
        x = torch.randn(1, 16, 4, 64, dtype=torch.float16)  # Make sure last dim is divisible
        
        # Per-token quantization (group_size=0)
        quantizer_token = Quantizer(n_bits=4, group_size=0, sym=False, clip_ratio=1.0)
        x_token = quantizer_token(x)
        
        # Group quantization (group_size=32)
        quantizer_group = Quantizer(n_bits=4, group_size=32, sym=False, clip_ratio=1.0)
        x_group = quantizer_group(x)
        
        diff_token = (x - x_token).abs().mean().item()
        diff_group = (x - x_group).abs().mean().item()
        
        print(f"Per-token quantization error: {diff_token:.6f}")
        print(f"Group quantization error: {diff_group:.6f}")
        
        assert diff_token > 0, "Per-token quantization should introduce error"
        assert diff_group > 0, "Group quantization should introduce error"
        
        print("✅ Different group sizes work!")


class TestHadamardTransform:
    """Test suite for Hadamard transform"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_hadamard_transform_basic(self):
        """Test basic Hadamard transform functionality"""
        print("Testing Hadamard transform...")
        
        from xKV.customized_cache.hadamard_utils import apply_hadamard
        
        # Create test tensor (batch_size * seq_len, dim) on GPU
        x = torch.randn(256, 128, dtype=torch.float16, device='cuda:0')
        print(f"Input tensor shape: {x.shape}")
        print(f"Input tensor device: {x.device}")
        
        # Apply Hadamard transform
        x_hadamard = apply_hadamard(x)
        print(f"Hadamard output shape: {x_hadamard.shape}")
        print(f"Input norm: {x.norm():.4f}")
        print(f"Hadamard output norm: {x_hadamard.norm():.4f}")
        
        # Assertions
        assert x_hadamard.shape == x.shape, "Shape should be preserved"
        assert x_hadamard.device == x.device, "Device should be preserved"
        
        # Hadamard should preserve energy (approximately)
        norm_ratio = x_hadamard.norm() / x.norm()
        print(f"Norm ratio: {norm_ratio:.4f}")
        
        # Allow some numerical error in norm preservation
        assert 0.9 < norm_ratio < 1.1, f"Norm should be approximately preserved, got ratio {norm_ratio:.4f}"
        
        print("✅ Hadamard transform is working correctly!")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_hadamard_transform_dimensions(self):
        """Test Hadamard transform with different dimensions"""
        print("Testing Hadamard transform with different dimensions...")
        
        from xKV.customized_cache.hadamard_utils import apply_hadamard
        
        # Test different sizes that should work with Hadamard
        test_dims = [64, 128, 256, 512]
        
        for dim in test_dims:
            print(f"Testing dimension {dim}...")
            x = torch.randn(32, dim, dtype=torch.float16, device='cuda:0')
            
            try:
                x_hadamard = apply_hadamard(x)
                assert x_hadamard.shape == x.shape
                
                # Check norm preservation
                norm_ratio = x_hadamard.norm() / x.norm()
                assert 0.9 < norm_ratio < 1.1, f"Norm preservation failed for dim {dim}"
                
                print(f"  ✅ Dimension {dim} works correctly")
            except Exception as e:
                pytest.fail(f"Hadamard transform failed for dimension {dim}: {e}")
        
        print("✅ All dimensions work correctly!")


class TestIntegratedQuantization:
    """Test integrated quantization with fake SVD"""
    
    def test_fake_svd_with_quantization(self):
        """Test fake SVD with quantization enabled"""
        print("Testing fake SVD with quantization...")
        
        from xKV.customized_cache.fake_layer_merge_dynamic_cache import fake_svd
        from xKV.customized_cache.quant import Quantizer
        
        # Create test tensor (use float32 for SVD compatibility)
        tensor = torch.randn(2, 32, 64, 128, dtype=torch.float32)  # [bs, nh, sl, hd]
        rank = 32
        
        # Create quantizer
        quantizer = Quantizer(n_bits=4, group_size=0, sym=False, clip_ratio=1.0)
        
        # Test fake SVD without quantization
        result_no_quant = fake_svd(tensor, rank, apply_hadamard_transform=False, quantizer=None)
        
        # Test fake SVD with quantization only
        result_quant_only = fake_svd(tensor, rank, apply_hadamard_transform=False, quantizer=quantizer)
        
        print(f"Original tensor shape: {tensor.shape}")
        print(f"Result shape: {result_no_quant.shape}")
        
        # Both should have same shape as input
        assert result_no_quant.shape == tensor.shape
        assert result_quant_only.shape == tensor.shape
        
        # Quantized version should be different from non-quantized
        diff = (result_no_quant - result_quant_only).abs().mean().item()
        print(f"Difference between quantized and non-quantized: {diff:.6f}")
        assert diff > 0, "Quantization should introduce some difference"
        
        print("✅ Fake SVD with quantization works correctly!")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fake_svd_with_hadamard_and_quantization(self):
        """Test fake SVD with both Hadamard transform and quantization"""
        print("Testing fake SVD with Hadamard transform and quantization...")
        
        from xKV.customized_cache.fake_layer_merge_dynamic_cache import fake_svd
        from xKV.customized_cache.quant import Quantizer
        
        # Create test tensor on GPU (use float32 for SVD compatibility)
        tensor = torch.randn(1, 16, 32, 64, dtype=torch.float32, device='cuda:0')  # [bs, nh, sl, hd]
        rank = 16
        
        # Create quantizer
        quantizer = Quantizer(n_bits=4, group_size=0, sym=False, clip_ratio=1.0)
        quantizer = quantizer.to('cuda:0')
        
        # Test fake SVD with both Hadamard and quantization
        result_full = fake_svd(tensor, rank, apply_hadamard_transform=True, quantizer=quantizer)
        
        # Test fake SVD with only Hadamard
        result_hadamard_only = fake_svd(tensor, rank, apply_hadamard_transform=True, quantizer=None)
        
        print(f"Original tensor shape: {tensor.shape}")
        print(f"Result shape: {result_full.shape}")
        
        # Should have same shape as input
        assert result_full.shape == tensor.shape
        assert result_hadamard_only.shape == tensor.shape
        
        # Should be on same device
        assert result_full.device == tensor.device
        
        # Results should be different when quantization is applied
        diff = (result_full - result_hadamard_only).abs().mean().item()
        print(f"Difference with/without quantization: {diff:.6f}")
        assert diff > 0, "Quantization should introduce some difference"
        
        print("✅ Fake SVD with Hadamard and quantization works correctly!")


if __name__ == "__main__":
    # Run tests directly if called as script
    pytest.main([__file__, "-v", "-s"])
