import os
import ray
import pyarrow as pa
from typing import List, Optional, Union, Any, Dict
from data_prep_toolkit.transforms import AbstractTableTransform

try:
    import sglang as sgl
except ImportError:
    sgl = None

try:
    from jinja2 import Template, Environment, FileSystemLoader, BaseLoader
except ImportError:
    Template = None
    Environment = None


class SGLangTransformConfig:
    """
    Configuration for SGLangTransform.
    """
    def __init__(
        self,
        model_path: str,
        output_column: str = "completions",
        map_column: Optional[str] = "instruction_seed",
        prompt_template: Optional[str] = None,
        prompt_template_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        # Parallelism
        tp_size: int = 1,
        max_replicas: int = 1,
        batch_size: int = 10000, 
        # Model generic
        tokenizer_path: Optional[str] = None,
        trust_remote_code: bool = True,
        mem_fraction_static: float = 0.9,
        max_prefill_tokens: int = 16384,
        context_length: Optional[int] = None,
        # Sampling
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        max_new_tokens: int = 16,
        stop: Optional[Union[str, List[str]]] = None,
    ):
        self.model_path = model_path
        self.output_column = output_column
        self.map_column = map_column
        self.prompt_template = prompt_template
        self.prompt_template_path = prompt_template_path
        self.system_prompt = system_prompt
        self.tp_size = tp_size
        self.max_replicas = max_replicas
        self.batch_size = batch_size
        self.tokenizer_path = tokenizer_path
        self.trust_remote_code = trust_remote_code
        self.mem_fraction_static = mem_fraction_static
        self.max_prefill_tokens = max_prefill_tokens
        self.context_length = context_length
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.max_new_tokens = max_new_tokens
        self.stop = stop


@ray.remote
class _SGLangActor:
    def __init__(self, config: SGLangTransformConfig):
        self.config = config
        if sgl is None:
             raise ImportError("sglang is not installed.")
        
        # Initialize Jinja2
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

        # Initialize SGLang Engine
        self.engine = sgl.Engine(
            model_path=config.model_path,
            tokenizer_path=config.tokenizer_path,
            trust_remote_code=config.trust_remote_code,
            mem_fraction_static=config.mem_fraction_static,
            tp_size=config.tp_size,
            # context_length=config.context_length
        )

    def generate_batch(self, rows: List[Dict[str, Any]]) -> List[str]:
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

        sampling_params = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty,
            "max_new_tokens": self.config.max_new_tokens,
            "stop": self.config.stop,
        }
        
        outputs = self.engine.generate(prompts, sampling_params)
        
        # Parse outputs (assume simple text list for now, SGLang parsing can be complex)
        # outputs structure usually: list of objects with 'text'
        generated_texts = [o["text"] if isinstance(o, dict) else o.text for o in outputs]
        
        return generated_texts


class SGLangTransform(AbstractTableTransform):
    """
    Data Prep Kit Transform for SGLang Inference.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.sgl_config = SGLangTransformConfig(**config)
        
        # Initialize Actor Pool
        self.actors = []
        num_gpus_per_actor = self.sgl_config.tp_size
        
        for i in range(self.sgl_config.max_replicas):
            actor = _SGLangActor.options(
                num_gpus=num_gpus_per_actor,
                name=f"sglang_transform_actor_{i}"
            ).remote(self.sgl_config)
            self.actors.append(actor)
        
        self.current_actor_idx = 0

    def transform(self, table: pa.Table) -> pa.Table:
        """
        Applies SGLang inference on the input PyArrow Table.
        """
        rows = table.to_pylist()
        chunk_size = self.sgl_config.batch_size
        all_generated_texts = []
        futures = []
        
        # Dispatch chunks
        for i in range(0, len(rows), chunk_size):
            chunk_rows = rows[i : i + chunk_size]
            
            # Round-robin dispatch
            actor = self.actors[self.current_actor_idx]
            self.current_actor_idx = (self.current_actor_idx + 1) % len(self.actors)
            
            futures.append(actor.generate_batch.remote(chunk_rows))
        
        # Gather results
        results = ray.get(futures)
        for res in results:
            all_generated_texts.extend(res)
            
        output_col = self.sgl_config.output_column
        new_column = pa.array(all_generated_texts, type=pa.string())
        new_table = table.append_column(output_col, new_column)
        
        return new_table
