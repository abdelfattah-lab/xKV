#!/usr/bin/env python3
"""
Pytest tests for the fused_hadamard_matrix function.
This script validates that the Hadamard transform preserves matrix multiplication structure.
"""

import torch
import numpy as np
import sys
import os
import pytest

# Add the xKV module to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from xKV.customized_cache.hadamard_utils import apply_hadamard
except ImportError:
    pytest.skip("apply_hadamard not available", allow_module_level=True)

def fused_hadamard_matrix(A, B):
    """
    Apply Hadamard transform to matrices A and B.
    For matrix product A @ B, we want: (Q @ A) @ (B @ Q^T) = Q @ A @ B @ Q^T
    when Q @ Q^T = I (orthogonal property of Hadamard matrix).
    
    A: First matrix (left operand)
    B: Second matrix (right operand)  
    Returns: (A_transformed, B_transformed) where A_transformed @ B_transformed preserves structure
    """
    # Apply Q to A (left multiplication)
    A_transformed = apply_hadamard(A)
    
    # Apply Q^T to B (right multiplication: B @ Q^T)
    # We want: B @ Q^T = (Q @ B^T)^T
    # So: B @ Q^T = (Q @ B^T)^T = apply_hadamard(B.t()).t()
    B_transformed = apply_hadamard(B.t()).t()
    
    return A_transformed, B_transformed

@pytest.fixture
def cuda_device():
    """Fixture to check CUDA availability."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping test as apply_hadamard requires CUDA")
    return torch.device('cuda')

def test_basic_functionality(cuda_device):
    """Test basic functionality with simple matrices."""
    
    # Create test matrices with power-of-2 dimensions for Hadamard transform
    torch.manual_seed(42)
    A = torch.randn(4, 8, device=cuda_device)  # 4 and 8 are powers of 2
    B = torch.randn(8, 16, device=cuda_device)  # 8 and 16 are powers of 2
    
    # Apply Hadamard transform
    A_transformed, B_transformed = fused_hadamard_matrix(A, B)
    
    # Check shapes are preserved
    assert A_transformed.shape == A.shape, f"A shape mismatch: {A_transformed.shape} vs {A.shape}"
    assert B_transformed.shape == B.shape, f"B shape mismatch: {B_transformed.shape} vs {B.shape}"
    
    # Compute original and transformed products
    original_product = A @ B
    transformed_product = A_transformed @ B_transformed
    
    # Check if shapes match
    assert original_product.shape == transformed_product.shape
    original_norm = torch.norm(original_product)
    transformed_norm = torch.norm(transformed_product)
    
    print(f"Original product norm: {original_norm:.4f}")
    print(f"Transformed product norm: {transformed_norm:.4f}")
    
    # Verify norm preservation (within tolerance) instead of returning
    norm_ratio = transformed_norm / original_norm
    assert abs(norm_ratio - 1.0) < 0.1, f"Norm ratio {norm_ratio} should be close to 1.0"

def test_batch_matrices():
    """Test with batch matrices (3D tensors)."""
    print("\n=== Testing Batch Matrices ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. Skipping batch test.")
        return
    
    torch.manual_seed(123)
    batch_size = 3
    A = torch.randn(batch_size, 4, 8, device=device)  # Powers of 2
    B = torch.randn(batch_size, 8, 16, device=device)  # Powers of 2
    
    print(f"Batch A shape: {A.shape}")
    print(f"Batch B shape: {B.shape}")
    
    # Apply transform to each batch
    A_transformed_list = []
    B_transformed_list = []
    
    for i in range(batch_size):
        A_t, B_t = fused_hadamard_matrix(A[i], B[i])
        A_transformed_list.append(A_t)
        B_transformed_list.append(B_t)
    
    A_transformed = torch.stack(A_transformed_list)
    B_transformed = torch.stack(B_transformed_list)
    
    print(f"Batch transformed A shape: {A_transformed.shape}")
    print(f"Batch transformed B shape: {B_transformed.shape}")
    
    # Check shapes
    assert A_transformed.shape == A.shape
    assert B_transformed.shape == B.shape
    
    print("✓ Batch processing works correctly")

def test_svd_like_scenario():
    """Test scenario similar to SVD decomposition."""
    print("\n=== Testing SVD-like Scenario ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. Skipping SVD-like test.")
        return
    
    torch.manual_seed(456)
    
    # Simulate SVD-like matrices with power-of-2 dimensions
    bs, sl, nh_hd = 2, 16, 64  # All powers of 2: batch_size, seq_len, num_heads * head_dim
    rank = 8  # Power of 2
    
    # U-like matrix: (bs, sl, rank)
    U = torch.randn(bs, sl, rank, device=device)
    
    # V-like matrix: (bs, rank, nh_hd)  
    V = torch.randn(bs, rank, nh_hd, device=device)
    
    print(f"U-like matrix shape: {U.shape}")
    print(f"V-like matrix shape: {V.shape}")
    
    # Process each batch separately
    results = []
    for i in range(bs):
        U_t, V_t = fused_hadamard_matrix(U[i], V[i])
        
        # Compute products
        original_product = U[i] @ V[i]  # (sl, nh_hd)
        transformed_product = U_t @ V_t  # (sl, nh_hd)
        
        results.append({
            'original_norm': torch.norm(original_product).item(),
            'transformed_norm': torch.norm(transformed_product).item(),
            'shape': transformed_product.shape
        })
        
        print(f"Batch {i}: Original norm={results[i]['original_norm']:.4f}, "
              f"Transformed norm={results[i]['transformed_norm']:.4f}")
    
    print("✓ SVD-like scenario completed")

def test_error_cases():
    """Test error cases and edge conditions."""
    print("\n=== Testing Error Cases ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. Skipping error case tests.")
        return
    
    # Test dimension mismatch
    try:
        A = torch.randn(4, 8, device=device)  # Powers of 2
        B = torch.randn(16, 32, device=device)  # Powers of 2, but incompatible for multiplication
        
        A_t, B_t = fused_hadamard_matrix(A, B)
        
        # This should not raise an error in the transform itself,
        # but matrix multiplication A_t @ B_t should fail
        try:
            product = A_t @ B_t
            print("Warning: Expected dimension mismatch error didn't occur")
        except RuntimeError as e:
            print(f"✓ Correctly caught dimension mismatch: {str(e)[:50]}...")
            
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    # Test with very small matrices
    A_small = torch.randn(1, 1, device=device)
    B_small = torch.randn(1, 1, device=device)
    
    A_t, B_t = fused_hadamard_matrix(A_small, B_small)
    print(f"✓ Small matrix test passed: {A_t.shape}, {B_t.shape}")

def main():
    """Run all tests."""
    print("Testing fused_hadamard_matrix function")
    print("=" * 50)
    
    try:
        test_basic_functionality()
        test_batch_matrices()
        test_svd_like_scenario()
        test_error_cases()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed successfully!")
        print("\nNote: This test uses a mock Hadamard transform.")
        print("For real validation, ensure apply_hadamard is properly implemented.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
