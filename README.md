Install Lucile with the following command:
```
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ Lucile-bigshaq9999
```

To run on amd gpu:

```
# 1. Create a Python 3.12 virtual environment
uv venv --python 3.12

# 2. Install PyTorch with ROCm 6.2 support
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2

# 3. Install project dependencies
uv pip install -e .

# 4. Launch Lucile with the AMD ISA override
HSA_OVERRIDE_GFX_VERSION=11.0.0 uv run lucile
```