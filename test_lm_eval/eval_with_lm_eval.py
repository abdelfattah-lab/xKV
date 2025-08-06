import os
import sys
import warnings
import argparse
import json
import datetime
from typing import Dict, Any, List

# Add root directory to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

# Add local lm-evaluation-harness to path
lm_eval_path = os.path.join(root_dir, '3rdparty', 'lm-evaluation-harness')
sys.path.insert(0, lm_eval_path)

warnings.filterwarnings("ignore")

import torch
from termcolor import colored

from lm_eval import evaluator as lm_evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table

import numpy as np
import random

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


class XKVModel(HFLM):
    """
    A wrapper around HFLM that supports xKV compression and other KV cache optimizations.
    This allows us to use lm_eval with compressed models by overriding the generate function.
    """
    
    def __init__(self, model_path: str, args: argparse.Namespace, **kwargs):
        """
        Initialize the model with potential compression.
        
        Args:
            model_path: Path to the model
            args: Arguments containing compression settings
            **kwargs: Additional arguments for HFLM
        """
        # Set default kwargs if not provided
        if 'device_map' not in kwargs:
            kwargs['device_map'] = 'auto'
        if 'dtype' not in kwargs:
            kwargs['dtype'] = 'bfloat16'
        if 'trust_remote_code' not in kwargs:
            kwargs['trust_remote_code'] = True
        
        # Initialize the base HFLM model
        super().__init__(pretrained=model_path, **kwargs)
        
        # Store args for generate function
        self.compression_args = args
        
        # Apply compression methods
        self._apply_compression(args)
    
    def _apply_compression(self, args: argparse.Namespace):
        """Apply various compression methods based on arguments."""
        
        if args.xKV:
            print(colored("Applying xKV compression...", "green"))
            from utils import apply_kv_compress_patch
            self._model = apply_kv_compress_patch(self._model, args)
            print(colored("xKV compression applied successfully", "green"))
            return
        
        # Configuration mapping for MInference methods
        minference_configs = {
            'streamingllm': {
                'kv_type': 'streamingllm',
                'attn_kwargs': {"n_local": 96, "n_init": 32}
            },
            'snapKV': {
                'kv_type': 'snapkv',
                'attn_kwargs': {"max_capacity_prompt": 128}
            },
            'pyramidkv': {
                'kv_type': 'pyramidkv',
                'attn_kwargs': {"max_capacity_prompt": 128}
            },
            'kivi': {
                'kv_type': 'kivi',
                'attn_kwargs': {"bits": 2, "group_size": 128, "residual_length": 128}
            },
            'quest': {
                'kv_type': 'quest',
                'attn_kwargs': {"chunk_size": 16, "token_budget": 64}
            }
        }
        
        # Apply MInference-based methods
        for method, config in minference_configs.items():
            if getattr(args, method, False):
                print(colored(f"Applying {method.upper()}...", "green"))
                from minference import MInference
                minference_patch = MInference(
                    attn_type="dense",
                    model_name=self._model.config.name_or_path,
                    **config
                )
                self._model = minference_patch(self._model)
                print(colored(f"{method.upper()} applied successfully", "green"))
                break

    def generate(self, inputs, **kwargs):
        """
        Custom generate function that properly handles xKV compression.
        
        This function overrides the default generate to ensure compatibility
        with xKV's custom cache implementation.
        """
        if hasattr(self.compression_args, 'xKV') and self.compression_args.xKV:
            # For xKV, we need to handle the cache differently
            from transformers import GenerationConfig
            from xKV.customized_cache import FakeLayerMergingCache
            
            # Create generation config from kwargs
            generation_config = GenerationConfig.from_pretrained(self._model.config.name_or_path)
            for key, value in kwargs.items():
                if hasattr(generation_config, key):
                    setattr(generation_config, key, value)
            
            # Ensure we use our custom cache
            if 'past_key_values' not in kwargs:
                kwargs['past_key_values'] = FakeLayerMergingCache()
            
            # Call the model's generate with our custom handling
            with torch.no_grad():
                return self._model.generate(inputs, **kwargs)
        else:
            # For other methods, use the standard generate
            return self._model.generate(inputs, **kwargs)
    
    def _model_call(
        self,
        inps: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Override model call to handle xKV-specific requirements.
        """
        if hasattr(self.compression_args, 'xKV') and self.compression_args.xKV:
            # For xKV, we need to manually handle the model call with proper cache
            from xKV.customized_cache import FakeLayerMergingCache
            
            # Get the xKV config from the model
            config = self._model.kv_compress_config
            
            # Create FakeLayerMergingCache for this call
            past_key_values = FakeLayerMergingCache(config)
            
            # Call the model directly with the cache
            if labels is not None:
                return self._model(input_ids=inps, attention_mask=attn_mask, labels=labels, past_key_values=past_key_values).logits
            else:
                return self._model(input_ids=inps, attention_mask=attn_mask, past_key_values=past_key_values).logits
        else:
            # For other methods, use the parent implementation
            return super()._model_call(inps, attn_mask, labels)


def get_method_name(args: argparse.Namespace) -> str:
    """Determine the method name based on arguments."""
    if args.xKV:
        # Include layer group size in the method name for xKV variants
        return f"xKV-{args.layer_group_size}"
    elif args.streamingllm:
        return "streamingllm"
    elif args.snapKV:
        return "snapKV"
    elif args.pyramidkv:
        return "pyramidkv"
    elif args.kivi:
        return "kivi"
    elif args.quest:
        return "quest"
    return "baseline"


def get_output_filename(args: argparse.Namespace) -> str:
    """Generate output filename based on model and method."""
    model_name = args.model_name_or_path.split('/')[-1]
    
    if args.xKV:
        # Use the new naming convention: xKV-{group_size}_k{rank_k}_v{rank_v}
        method_name = f"xKV-{args.layer_group_size}_k{args.rank_k}_v{args.rank_v}"
        if args.kv_bits < 16:
            method_name += f"_{args.kv_bits}bit"
    else:
        method_name = get_method_name(args)
    
    return f"{model_name}_{method_name}.json"


def add_lm_eval_args(parser: argparse.ArgumentParser):
    """Add lm_eval specific arguments."""
    parser.add_argument(
        "--tasks", 
        type=str, 
        required=True,
        help="Tasks to evaluate on (comma-separated). Examples: 'hellaswag,arc_easy,arc_challenge'"
    )
    parser.add_argument(
        "--num_fewshot", 
        type=int, 
        default=None,
        help="Number of few-shot examples"
    )
    parser.add_argument(
        "--batch_size", 
        type=str, 
        default="auto",
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max_batch_size", 
        type=int, 
        default=None,
        help="Maximum batch size"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Limit number of examples per task"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="lm_eval_results",
        help="Output directory for results"
    )
    return parser


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate language models with lm_eval and xKV compression"
    )
    
    # Add common args from utils
    from utils import add_common_args
    parser = add_common_args(parser)
    
    # Add lm_eval specific args
    parser = add_lm_eval_args(parser)
    
    # Parse and validate
    args = parser.parse_args()
    
    # Validation
    if not args.tasks:
        raise ValueError("--tasks is required")
    
    return args


def run_evaluation(model_path: str, tasks: List[str], args: argparse.Namespace, method_name: str = "baseline") -> Dict[str, Any]:
    """Run evaluation with a single method."""
    
    print(colored(f"\n{'='*60}", "blue"))
    print(colored(f"Evaluating {method_name.upper()} method", "blue"))
    print(colored(f"Model: {model_path}", "blue"))
    print(colored(f"Tasks: {', '.join(tasks)}", "blue"))
    print(colored(f"{'='*60}", "blue"))
    
    # Create model with compression
    model = XKVModel(model_path, args)
    
    # Run evaluation
    results = lm_evaluator.simple_evaluate(
        model=model,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        limit=args.limit,
    )
    
    res = make_table(results)
    print(res)
    
    # Add metadata
    results["method"] = method_name
    results["model_path"] = model_path
    results["timestamp"] = datetime.datetime.now().isoformat()
    results["args"] = vars(args)
    
    # Print summary
    return results['results']


def main():
    """Main evaluation function."""
    args = parse_args()

    seed_everything(42)

    # Parse tasks
    tasks = [task.strip() for task in args.tasks.split(',')]
    
    # Run evaluation
    method_name = get_method_name(args)
    results = run_evaluation(args.model_name_or_path, tasks, args, method_name)
    
    # Save results
    output_dir = args.output_path or "lm_eval_results"
    os.makedirs(output_dir, exist_ok=True)
    
    filename = get_output_filename(args)
    output_file = os.path.join(output_dir, filename)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(colored(f"\nResults saved to: {output_file}", "green"))
    print(colored("Evaluation completed!", "green"))


if __name__ == "__main__":
    main()
