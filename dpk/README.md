# Data Prep Kit Transforms for Offline Inference

This package provides custom **Data Prep Kit (DPK)** Transforms for high-performance offline inference using **vLLM** and **SGLang**. These transforms allow you to integrate Large Language Model (LLM) generation directly into your DPK pipelines, leveraging Ray for distributed execution.

## Features

- **High Performance**: Uses vLLM and SGLang engines for state-of-the-art inference speed.
- **Distributed**: Built on Ray with support for Tensor Parallelism (TP), Pipeline Parallelism (PP), and Data Parallelism (DP).
- **Persistent Actors**: Models are loaded once and reused across data chunks, avoiding expensive reloading.
- **Memory Safe**: Configurable "Smart Chunking" (`batch_size`) prevents Host CPU OOMs during large-scale processing.
- **Dynamic Templating**: Full Jinja2 support with access to all dataset columns.

## Installation

Ensure you have the following dependencies installed:

```bash
# Core dependencies
pip install data-prep-toolkit-transforms[all] ray[default]

# Inference Engines (Install at least one)
pip install vllm
# - OR -
pip install "sglang[all]"

# Utilities
pip install jinja2 pyarrow pandas
```

> [!IMPORTANT]
> Ensure your environment (Driver and Workers) has access to GPUs and compatible CUDA drivers.

## Usage

You can use these transforms in your Python DPK scripts.

### Basic Import
```python
from dpk.vllm_transform import VLLMTransform
from dpk.sglang_transform import SGLangTransform
```

### Configuration Parameters

Common parameters for `VLLMTransform` and `SGLangTransform`:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` / `model_path` | str | **Required** | HuggingFace model ID or local path. |
| `tensor_parallel_size` / `tp_size` | int | 1 | Number of GPUs to use per model instance (TP). |
| `max_replicas` | int | 1 | **Data Parallelism**: Number of concurrent model instances (Actors) to run. |
| `batch_size` | int | 10000 | **Chunk Size**: Number of rows processed per chunk on Host CPU. |
| `prompt_template` | str | None | Inline Jinja2 template string. |
| `prompt_template_path` | str | None | Path to a Jinja2 template file. |
| `output_column` | str | "completions" | Name of the new column to store results. |

---

## Examples

### 1. Small Model (Llama-3-8B)
**Scenario**: Model fits on 1 GPU. We want high throughput using Data Parallelism (running independent instances on multiple GPUs).

**Setup**: 4 GPUs available.
**Config**: 1 GPU per model * 4 Replicas = 4 GPUs total.

```python
config = {
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "tensor_parallel_size": 1, 
    "max_replicas": 4,         # Utilize all 4 GPUs independently
    "batch_size": 5000,        # Process 5k rows at a time per actor
    "prompt_template": "USER: {{ input_text }}\nASSISTANT:"
}

transform = VLLMTransform(config)
result_table = transform.transform(input_table)
```

### 2. Medium Model (Llama-3-70B)
**Scenario**: Model requires multiple GPUs (Tensor Parallelism) to fit in VRAM.

**Setup**: 4 x A100 (80GB).
**Config**: 4 GPUs per model * 1 Replica = 4 GPUs total.

```python
config = {
    "model": "meta-llama/Meta-Llama-3-70B-Instruct",
    "tensor_parallel_size": 4, # Split model across 4 GPUs
    "max_replicas": 1,         # One giant instance
    "output_column": "model_response",
    # Using a file template
    "prompt_template_path": "templates/chat_template.j2"
}

transform = VLLMTransform(config)
# transform.transform(input_table)
```

### 3. Large / MoE Model (DeepSeek-V3/R1) with SGLang
**Scenario**: Massive Mixture-of-Experts model. SGLang is preferred for better MoE performance.

**Setup**: 8 x H100.
**Config**: 8 GPUs (TP=8).

```python
config = {
    "model_path": "deepseek-ai/DeepSeek-V3",
    "tp_size": 8,            # Tensor Parallel across 8 GPUs
    "max_replicas": 1,
    "mem_fraction_static": 0.90,
    "batch_size": 20000,
    # Jinja Template accessing multiple columns
    "prompt_template": """
    <|system|>
    You are a helpful assistant.
    <|user|>
    Context: {{ context_col }}
    Question: {{ question_col }}
    <|assistant|>
    """
}

transform = SGLangTransform(config)
# transform.transform(input_table)
```

### 4. Dynamic Templating (Jinja2)
You can inject any column from your dataset into the prompt.

**Dataset**:
| instruction | context | complexity |
| :--- | :--- | :--- |
| "Summarize" | "Long text..." | "High" |

**Config**:
```python
config = {
    # ... model config ...
    "prompt_template": "Task: {{ instruction }} (Complexity: {{ complexity }})\nInput: {{ context }}\nOutput:"
}
```

### 5. Selecting Columns (No Template)
If you just want to pass a single column directly as the prompt string.

```python
config = {
    # ... model config ...
    "map_column": "raw_prompt_text", # The column containing the full prompt

### 6. Using External Prompt Files
For large prompts, it's best to keep the template in a separate file (e.g., `templates/my_prompt.j2`).

**File:** `templates/my_prompt.j2`
```jinja2
You are an expert mathematician.

Problem: {{ question }}
Context: {{ context_data }}

Please provide a step-by-step solution.
Solution:
```

**Config:**
```python
config = {
    # ... model config ...
    "prompt_template_path": "templates/my_prompt.j2"
}
```
transform = VLLMTransform(config)
