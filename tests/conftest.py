"""
Pytest configuration and shared fixtures for xKV tests.
"""

import pytest
import torch
import sys
import os

# Add the xKV module to path for all tests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def cuda_device():
    """Session-scoped fixture for CUDA device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        pytest.skip("CUDA not available")


@pytest.fixture(scope="session") 
def cpu_device():
    """Session-scoped fixture for CPU device."""
    return torch.device('cpu')


@pytest.fixture
def device():
    """Default device fixture - uses CUDA if available, otherwise CPU."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture(autouse=True)
def set_random_seed():
    """Ensure reproducible tests by setting random seeds."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "cuda: mark test as requiring CUDA"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add cuda marker to tests that require CUDA
        if "cuda" in item.name.lower() or "gpu" in item.name.lower():
            item.add_marker(pytest.mark.cuda)
        
        # Add integration marker to integration tests
        if "integration" in item.name.lower():
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker to tests that might be slow
        if any(keyword in item.name.lower() for keyword in ["batch", "large", "performance"]):
            item.add_marker(pytest.mark.slow)
