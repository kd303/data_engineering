import os
import ray
import pyarrow as pa
from typing import List, Optional, Union, Any, Dict
from data_processing.transform import AbstractTableTransform, TransformConfiguration

# Try updates for vllm/jinja2 imports
try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

try:
    from jinja2 import Template, Environment, FileSystemLoader, BaseLoader
except ImportError:
    Template = None
    Environment = None


class VLLMTransformConfig:
    """
    Configuration for VLLMTransform.
    """
    def __init__(
        self,
        model: str,
        output_column: str = "completions",
        map_column: Optional[str] = "instruction_seed",
        prompt_template: Optional[str] = None,
        prompt_template_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        # Parallelism
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        context_parallel_size: int = 1,
        enable_context_parallel: bool = False,
        distributed_executor_backend: str = "ray",
        max_replicas: int = 1,
        batch_size: int = 10000, # Chunk size
        # Model generic
        tokenizer: Optional[str] = None,
        revision: Optional[str] = None,
        trust_remote_code: bool = True,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        enable_prefix_caching: bool = False,
        swap_space: int = 4,
        enforce_eager: bool = False,
        # Sampling
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        max_tokens: int = 16,
        stop: Optional[Union[str, List[str]]] = None,
        stop_token_ids: Optional[List[int]] = None,
        n: int = 1,
        best_of: Optional[int] = None,
        logprobs: Optional[int] = None,
    ):
        self.model = model
        self.output_column = output_column
        self.map_column = map_column
        self.prompt_template = prompt_template
        self.prompt_template_path = prompt_template_path
        self.system_prompt = system_prompt
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.context_parallel_size = context_parallel_size
        self.enable_context_parallel = enable_context_parallel
        self.distributed_executor_backend = distributed_executor_backend
        self.max_replicas = max_replicas
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.revision = revision
        self.trust_remote_code = trust_remote_code
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enable_prefix_caching = enable_prefix_caching
        self.swap_space = swap_space
        self.enforce_eager = enforce_eager
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.max_tokens = max_tokens
        self.stop = stop
        self.stop_token_ids = stop_token_ids
        self.n = n
        self.best_of = best_of
        self.logprobs = logprobs


@ray.remote
class _VLLMActor:
    def __init__(self, config: VLLMTransformConfig):
        self.config = config
        if LLM is None:
             raise ImportError("vLLM is not installed.")
        
        # Initialize Jinja2 Environment
        if self.config.prompt_template_path:
            template_dir = os.path.dirname(self.config.prompt_template_path)
            template_file = os.path.basename(self.config.prompt_template_path)
            self.jinja_env = Environment(loader=FileSystemLoader(template_dir))
            self.template = self.jinja_env.get_template(template_file)
        elif self.config.prompt_template:
            self.jinja_env = Environment(loader=BaseLoader())
            self.template = self.jinja_env.from_string(self.config.prompt_template)
        else:
            self.template = None

        # vLLM args
        vllm_args = {
            "model": config.model,
            "tokenizer": config.tokenizer,
            "revision": config.revision,
            "trust_remote_code": config.trust_remote_code,
            "max_model_len": config.max_model_len,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "enable_prefix_caching": config.enable_prefix_caching,
            "swap_space": config.swap_space,
            "enforce_eager": config.enforce_eager,
            "tensor_parallel_size": config.tensor_parallel_size,
            "pipeline_parallel_size": config.pipeline_parallel_size,
            "distributed_executor_backend": config.distributed_executor_backend,
        }
        
        # Initialize vLLM engine
        self.llm = LLM(**vllm_args)
        
        self.sampling_params = SamplingParams(
            n=config.n,
            best_of=config.best_of,
            presence_penalty=config.presence_penalty,
            frequency_penalty=config.frequency_penalty,
            repetition_penalty=config.repetition_penalty,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            min_p=config.min_p,
            stop=config.stop,
            stop_token_ids=config.stop_token_ids,
            max_tokens=config.max_tokens,
            logprobs=config.logprobs,
        )

    def generate_batch(self, rows: List[Dict[str, Any]]) -> List[Union[str, List[str]]]:
        prompts = []
        if self.template:
             for row in rows:
                 prompts.append(self.template.render(**row))
        elif self.config.map_column:
             for row in rows:
                 prompts.append(str(row.get(self.config.map_column, "")))
        else:
             raise ValueError("Either 'map_column' or 'prompt_template'/'prompt_template_path' must be provided.")
        
        if self.config.system_prompt:
             prompts = [f"{self.config.system_prompt}\n{p}" for p in prompts]

        # Generate (vLLM handles batching internally)
        outputs = self.llm.generate(prompts, self.sampling_params)
        
        generated_texts = []
        for output in outputs:
            if self.config.n == 1:
                generated_texts.append(output.outputs[0].text)
            else:
                generated_texts.append([o.text for o in output.outputs])
        return generated_texts


class VLLMTransform(AbstractTableTransform):
    """
    Data Prep Kit Transform for vLLM Inference.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.vllm_config = VLLMTransformConfig(**config)
        
        # Initialize Actor Pool
        self.actors = []
        num_gpus_per_actor = self.vllm_config.tensor_parallel_size * self.vllm_config.pipeline_parallel_size
        
        for i in range(self.vllm_config.max_replicas):
            actor = _VLLMActor.options(
                num_gpus=num_gpus_per_actor,
                name=f"vllm_transform_actor_{i}"
            ).remote(self.vllm_config)
            self.actors.append(actor)
        
        self.current_actor_idx = 0

    def transform(self, table: pa.Table) -> pa.Table:
        """
        Applies vLLM inference on the input PyArrow Table.
        """
        # Convert to list of dicts for processing
        rows = table.to_pylist()
        
        # Process in chunks of batch_size (configurable)
        chunk_size = self.vllm_config.batch_size
        all_generated_texts = []
        
        futures = []
        
        # Dispatch chunks to actors
        for i in range(0, len(rows), chunk_size):
            chunk_rows = rows[i : i + chunk_size]
            
            # Round-robin dispatch
            actor = self.actors[self.current_actor_idx]
            self.current_actor_idx = (self.current_actor_idx + 1) % len(self.actors)
            
            # Async call
            futures.append(actor.generate_batch.remote(chunk_rows))
        
        # Gather results (in order)
        results = ray.get(futures)
        for res in results:
            all_generated_texts.extend(res)
            
        # Append results to the original table
        # We need to handle list of strings vs string column
        output_col = self.vllm_config.output_column
        
        # Determine PyArrow type
        if self.vllm_config.n > 1:
             pa_type = pa.list_(pa.string())
        else:
             pa_type = pa.string()
             
        new_column = pa.array(all_generated_texts, type=pa_type)
        new_table = table.append_column(output_col, new_column)
        
        return new_table
