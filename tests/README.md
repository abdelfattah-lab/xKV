# xKV Test Suite

This test suite uses the pytest framework to test the core functionality of the xKV project.

## Test Structure

```
tests/
├── conftest.py                        # pytest configuration and shared fixtures
├── test_hadamard_transform.py         # Hadamard transform functionality tests
├── test_fake_svd.py                   # Basic fake SVD functionality tests
├── test_fake_svd_clean.py             # Clean fake SVD implementation tests
├── test_fake_svd_with_hadamard.py     # Fake SVD with Hadamard transform tests
├── test_fake_quantization.py          # Quantization functionality tests
├── test_cache_integration.py          # Basic cache integration tests
└── test_cache_integration_pytest.py   # Advanced FakeLayerMergingCache integration tests
```

## Running Tests

### Run All Tests
```bash
python run_tests.py
```

### Run Specific Test File
```bash
python -m pytest tests/test_fake_svd.py -v
```

### Run Only CPU Tests (Skip CUDA Tests)
```bash
python -m pytest tests/ -v -m "not cuda"
```

### Run Specific Test Class
```bash
python -m pytest tests/test_fake_svd_with_hadamard.py::TestFakeSVD -v
```

### Run Tests with Different Markers
```bash
# Run only CUDA tests
python -m pytest tests/ -v -m "cuda"

# Run only slow tests
python -m pytest tests/ -v -m "slow"

# Run only integration tests
python -m pytest tests/ -v -m "integration"
```

## Test Coverage

### Hadamard Transform Tests (`test_hadamard_transform.py`)
- Basic Hadamard transform functionality
- Batch matrix processing
- SVD-related scenarios
- Error handling

### Fake SVD Tests (`test_fake_svd.py`)
- Basic SVD decomposition and reconstruction
- Effects of different rank values
- Quantization functionality
- Consistency checks

### Clean Fake SVD Tests (`test_fake_svd_clean.py`)
- Clean implementation of fake SVD
- Comparison of different methods
- Standalone Hadamard function testing
- Various tensor size testing

### Hadamard + SVD Tests (`test_fake_svd_with_hadamard.py`)
- Integration of Hadamard transform with SVD
- Impact of quantization on results
- Comparison of different processing methods
- Testing with various tensor dimensions

### Quantization Tests (`test_fake_quantization.py`)
- 4-bit and 16-bit quantization testing
- Symmetric vs asymmetric quantization
- Group size effects
- Hadamard transform integration with quantization
- Integrated quantization with fake SVD

### Cache Integration Tests (`test_cache_integration.py` & `test_cache_integration_pytest.py`)
- FakeLayerMergingCache creation and configuration
- Cache update functionality
- SVD processing effect verification
- Testing with different layer configurations

## Test Requirements

- **CUDA**: Most tests require CUDA support; tests will be skipped if CUDA is not available
- **PyTorch**: Requires PyTorch with CUDA support
- **Dependencies**: xKV.customized_cache, xKV.configurations

## Test Configuration

Test configuration is defined in `pytest.ini` and `conftest.py`:

- **Automatic random seed setup**: Ensures reproducible test results
- **CUDA detection**: Automatically skips unsupported tests
- **Custom markers**: cuda, slow, integration
- **Verbose output**: Shows detailed test results by default

## Troubleshooting

### Common Issues

1. **CUDA unavailable**: Tests will automatically skip CUDA-related tests
2. **Module import errors**: Ensure the xKV module is in the path
3. **Out of memory**: Adjust tensor sizes in tests

### Debugging Tips

```bash
# Show detailed error messages
python -m pytest tests/ -v --tb=long

# Stop at first failure
python -m pytest tests/ -x

# Run only failed tests from last run
python -m pytest tests/ --lf

# Run tests with specific markers
python -m pytest tests/ -v -m "slow"
python -m pytest tests/ -v -m "integration"
```

## Adding New Tests

When adding new tests, please follow these conventions:

1. Test file names should start with `test_`
2. Test function names should start with `test_`
3. Use `@pytest.mark.skipif` to skip unsupported tests
4. Use fixtures to set up test environments
5. Ensure tests have good error messages
6. Add appropriate markers (cuda, slow, integration) as needed

## Test Markers

- `@pytest.mark.cuda`: Tests that require CUDA
- `@pytest.mark.slow`: Long-running tests
- `@pytest.mark.integration`: Integration tests
