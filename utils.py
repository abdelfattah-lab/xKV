import argparse
import importlib
import numpy as np
import random, torch
from functools import reduce
from typing import Tuple, Any
from xKV.patch import KVCompress
from xKV.configurations import generate_consecutive_xKV_config
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from loguru import logger

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

# Set seed for reproducibility
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_model_numel(model):
    param_cnt = 0
    for name, module in model.named_modules():
        if hasattr(module, '_nelement'):
            param_cnt += module._nelement()
    return param_cnt

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**3
    return size_all_mb


def get_module_by_name(module, module_name):
    names = module_name.split(sep='.')
    return reduce(getattr, names, module)



def load_model_and_tokenizer(model_name_or_path, use_flash_attn2=False):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cuda",
        #device_map="auto",
        attn_implementation="flash_attention_2" if use_flash_attn2 else "sdpa",
    )
            
    model.eval()        
    return model, tokenizer

def apply_kv_compress_patch(model, args, verbose=True) -> Tuple[Any, KVCompress]:
    logger.info("Generating the kv compress configs")
    if args.customized_merge_config:
        logger.info(f"Loading the customized merge config from {args.customized_merge_config}")
        patch = KVCompress(yaml_path=args.customized_merge_config)
    else:
        logger.info("Generating the default merge config (consecutive)")
        xKV_config = generate_consecutive_xKV_config(
            num_layers=model.config.num_hidden_layers,
            rank_k=args.rank_k,
            rank_v=args.rank_v,
            group_size=args.layer_group_size,
            layer_merge_impl=args.layer_merge_impl,
            slerp_t=args.slerp_t,
            slerp_gamma=args.slerp_gamma,
            merge_key=args.merge_key,
            merge_value=args.merge_value,
            start_layer=args.start_layer_idx,
            end_layer=args.end_layer_idx if args.end_layer_idx != -1 else model.config.num_hidden_layers - 1,
            # Add quantization parameters
            kv_bits=args.kv_bits,
            quantizer_group_size=args.group_size,
            sym=args.sym,
            clip_ratio=args.clip_ratio,
            hadamard=args.hadamard,
        )
        patch = KVCompress(xKV_config=xKV_config)
    
    logger.info("compression config: {}".format(patch.config))
    logger.info("Applying the patch to the model")
    model = patch(model)
    return model


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument('--model_name_or_path', type=str, help='model to load')
    parser.add_argument('--flash2', action='store_true', help='whether to use flash-attention2')
    parser.add_argument('--xKV', action='store_true', help='whether to enable xKV patch')
    parser.add_argument('--streamingllm', action='store_true', help='whether to enable StreamingLLM patch')
    parser.add_argument('--snapKV', action='store_true', help='whether to enable snapKV patch')
    parser.add_argument('--pyramidkv', action='store_true', help='whether to enable pyramidkv patch')
    parser.add_argument('--kivi', action='store_true', help='whether to enable KIVI patch')
    parser.add_argument('--quest', action='store_true', help='whether to enable Quest patch')


    # online svd options
    # SVD-related parameters
    parser.add_argument("--rank_k", type=int, default=256, help="Rank for SVD compression of keys")
    parser.add_argument("--rank_v", type=int, default=768, help="Rank for SVD compression of values")
    parser.add_argument(
        '--layer_group_size',
        type=int,
        default=1,
        help='The number of layers that will be grouped and decompose jointly'
    )
    
    parser.add_argument(
        '--layer_merge_impl', 
        type=str, 
        default='svd',
        help='The implementation for layer merge'
    )
    parser.add_argument(
        '--slerp_t',
        type=float,
        default=0.5,
        help='The interpolation ratio for SLERP'
    )
    parser.add_argument(
        '--slerp_gamma',
        type=float,
        default=0.05,
        help='The gamma for identifying divergent token in SLERP',
    )
    
    # Merge control
    parser.add_argument("--merge_key", action="store_true", help="Enable merging for keys")
    parser.add_argument("--merge_value", action="store_true", help="Enable merging for values")
    parser.add_argument("--start_layer_idx", type=int, default=0, help="The starting layer index for layer merging")
    parser.add_argument("--end_layer_idx", type=int, default=-1, help="The ending layer index for layer merging. If -1, it will be the last layer.")
    parser.add_argument('--customized_merge_config', type=str, help='custom config file')
    
    # Quantization parameters
    parser.add_argument('--kv_bits', type=int, default=16, help='KV cache bit width, 16 means no quantization')
    parser.add_argument('--group_size', type=int, default=0, help='Group size for quantization, 0 means per-token quantization')
    parser.add_argument('--sym', action='store_true', help='Use symmetric quantization')
    parser.add_argument('--clip_ratio', type=float, default=1.0, help='Clip ratio for quantization')
    parser.add_argument('--hadamard', action='store_true', help='Use Hadamard transform for quantization')
    
    return parser