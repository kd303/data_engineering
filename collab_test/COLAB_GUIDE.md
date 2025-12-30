# Testing Data Prep Kit Transforms on Google Colab

You can easily test these transforms on Google Colab. **Note: You must select a GPU Runtime (T4 for small models, A100 for larger ones).**

## Step 1: Install Dependencies
Copy and paste this into the first cell of your Colab notebook:

```python
# Install system dependencies
!pip install -q vllm ray[default] data-prep-toolkit-transforms jinja2 pandas pyarrow huggingface_hub

# Install SGLang if needed (optional, takes longer)
# !pip install -q "sglang[all]"
```

## Step 2: Setup Code
You need to make the `dpk` package available. You can either clone your repository or upload the `dpk` folder.

**Option A: Clone Repository**
```python
!git clone https://github.com/kd303/data_engineering.git
import sys
import os

# Add the repo to python path
sys.path.append("/content/data_engineering")

# Verify import
from dpk_extn.vllm_transform import VLLMTransform
print("Import successful!")
```

**Option B: Manual Upload**
1.  Zip your local `dpk` folder.
2.  Upload `dpk.zip` to Colab files.
3.  Unzip it:
    ```python
    !unzip dpk.zip
    ```

## Step 3: Run Inference (Example)
Run a small model (e.g., `Qwen/Qwen2.5-1.5B-Instruct` or `TinyLlama`) that fits on a free Colab GPU.

```python
import pyarrow as pa
from dpk_extn.vllm_transform import VLLMTransform

# 1. Create Dummy Data
data = [
    {"instruction": "What is the capital of France?", "id": 1},
    {"instruction": "Write a python function to add two numbers.", "id": 2},
    {"instruction": "Explain quantum computing in one sentence.", "id": 3}
]
table = pa.Table.from_pylist(data)

# 2. Configure Transform
# We use a small model that fits on Colab T4 GPU
config = {
    "model": "Qwen/Qwen2.5-1.5B-Instruct", 
    "tensor_parallel_size": 1,
    "max_replicas": 1, 
    "batch_size": 10,  # Small chunk for testing
    "map_column": "instruction",
    "system_prompt": "You are a helpful assistant."
}

# 3. Initialize and Run
print("Initializing VLLM Transform...")
transform = VLLMTransform(config)

print("Running Inference...")
result_table = transform.transform(table)

# 4. View Results
df = result_table.to_pandas()
print(df[["instruction", "completions"]])
```
