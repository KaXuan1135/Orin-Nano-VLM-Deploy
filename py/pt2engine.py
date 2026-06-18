import os
import re
import gc
import time
import json
import torch
import shutil
import psutil
import textwrap
import argparse
import threading
import subprocess
import tensorrt_llm

from tensorrt_llm.mapping import Mapping
from tensorrt_llm._utils import release_gc
from huggingface_hub import snapshot_download
from tensorrt_llm.quantization import QuantAlgo
from tensorrt_llm.models import QWenForCausalLM
from transformers import AutoModel, AutoTokenizer
from tensorrt_llm.models.qwen.convert import load_hf_qwen
from tensorrt_llm.models.modeling_utils import QuantConfig

MODEL_CKPT_MAP = {
    'InternVL3-1B'  : 'OpenGVLab/InternVL3-1B',
    'InternVL3-2B'  : 'OpenGVLab/InternVL3-2B',
    'InternVL3_5-1B': 'OpenGVLab/InternVL3_5-1B',
}

DTYPE_SHORTFORM_MAP = {
    'float32' : 'fp32',
    'float16' : 'fp16',
    'bfloat16': 'bf16',
    'int8'    : 'i8',
    'int4'    : 'i4'
}

assert (MODEL_NAME := 'InternVL3-1B') in list(MODEL_CKPT_MAP.keys())
PATCH_TOKENS = 256
NUM_FRAMES = 6
LLM_BATCH_SIZE = 10
VIS_BATCH_SIZE = 6 # NUM_FRAMES 
MAX_MULTIMODAL_LEN = PATCH_TOKENS * NUM_FRAMES * LLM_BATCH_SIZE # total image len (sum of whole batch)
MAX_INPUT_LEN = PATCH_TOKENS * NUM_FRAMES + 2500 # input text length (for each batch)
MAX_SEQ_LEN = MAX_INPUT_LEN + 100 # output text length (for each batch)

KEEP_ONNX = True
assert (VIS_DTYPE := 'float16') in (VIS_DTYPE_CHOICES := ['float16', 'int8'])

PP_SIZE = 1
WORKERS = 1
LOAD_MODEL_ON_CPU = True
USE_PARALLEL_EMBEDDING = True
INT8_KV_CACHE = False
USE_WEIGHT_ONLY = True
assert (WEIGHT_ONLY_PRECISION := 'int8') in (WEIGHT_ONLY_PRECISION_CHOICES := ['int8', 'int4', 'int4_gptq'])
assert (LLM_DTYPE := 'bfloat16') in (LLM_DTYPE_CHOICES := ['float32', 'bfloat16', 'float16'])
assert (ENABLE_XQA := 'enable') in (BINARY_CHOICES := ['enable', 'disable'])
assert (CONTEXT_FMHA := 'enable') in BINARY_CHOICES # flash attention
assert (REDUCE_FUSION := 'enable') in BINARY_CHOICES
assert (PAGED_KV_CACHE := 'enable') in BINARY_CHOICES
assert (REMOVE_INPUT_PADDING := 'enable') in BINARY_CHOICES # enable for cpp, disable for python
assert (USE_PAGED_CONTEXT_FMHA := 'enable') in BINARY_CHOICES
assert (MOE_PLUGIN := 'disable') in (MOE_PLUGIN_CHOICES := ['auto', 'float16', 'float32', 'bfloat16' , 'int32' , 'disable'])
assert (GEMM_PLUGIN := 'auto') in (GEMM_PLUGIN_CHOICES := ['auto', 'float16', 'float32', 'bfloat16', 'int32', 'fp8', 'disable'])
assert (MAMBA_CONV1D_PLUGIN := 'disable') in (MAMBA_CONV1D_PLUGIN_CHOICES := ['auto', 'float16', 'float32', 'bfloat16', 'int32', 'disable'])
assert (GPT_ATTENTION_PLUGIN := 'auto') in (GPT_ATTENTION_PLUGIN_CHOICES := ['auto', 'float16', 'float32', 'bfloat16', 'int32', 'disable'])

def monitor_memory(proc, results):
    """Monitors the memory usage of a process and all its children."""
    peak_rss = 0
    peak_vms = 0
    start_time = time.time()
    try:
        # Create a psutil process object
        p = psutil.Process(proc.pid)
        while proc.poll() is None:  # While process is still running
            # Get memory for parent + all recursive children
            total_rss = p.memory_info().rss
            total_vms = p.memory_info().vms
            
            for child in p.children(recursive=True):
                try:
                    child_info = child.memory_info()
                    total_rss += child_info.rss
                    total_vms += child_info.vms
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Update peaks
            peak_rss = max(peak_rss, total_rss)
            peak_vms = max(peak_vms, total_vms)
            
            time.sleep(0.1)
    except psutil.NoSuchProcess:
        pass
    
    results['peak_rss_gb'] = peak_rss / (1024**3)
    results['peak_vms_gb'] = peak_vms / (1024**3)
    results['elapsed_time'] = int(time.time() - start_time)

def monitor_pid_memory(pid, results, stop_event):
    """Monitors memory for a specific PID until stop_event is set."""
    peak_rss = 0
    peak_vms = 0
    start_time = time.time()
    try:
        p = psutil.Process(pid)
        while not stop_event.is_set():
            total_rss = p.memory_info().rss
            total_vms = p.memory_info().vms
            
            for child in p.children(recursive=True):
                try:
                    child_info = child.memory_info()
                    total_rss += child_info.rss
                    total_vms += child_info.vms
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            peak_rss = max(peak_rss, total_rss)
            peak_vms = max(peak_vms, total_vms)
            time.sleep(0.1) 
    except psutil.NoSuchProcess:
        pass
    
    results['peak_rss_gb'] = peak_rss / (1024**3)
    results['peak_vms_gb'] = peak_vms / (1024**3)
    results['elapsed_time'] = int(time.time() - start_time)

def overview(model_output_path, vis2onnx_build, vis2engine_build, llm2engine_build):
    print("\n" + "="*50)
    print("       CONVERSION PROCESS SUMMARY OVERVIEW")
    print("="*50)

    # Define the stages and their corresponding result dicts
    stages = [
        ("Vision to ONNX", vis2onnx_build),
        ("Vision to Engine (trtexec)", vis2engine_build),
        ("LLM to Engine (trtllm-build)", llm2engine_build)
    ]

    # Header
    print(f"{'Stage':<30} | {'Status':<10} | {'Time':<8} | {'Peak RSS'} | {'Peak VMS'}")
    print("-" * 70)

    for name, results in stages:
        if results.get('skip'):
            status = "SKIPPED"
            time_str = "N/A"
            rss_str = "N/A"
            vms_str = "N/A"
        else:
            status = "DONE"
            time_str = f"{results.get('elapsed_time', 0)}s"
            rss_str = f"{results.get('peak_rss_gb', 0):.2f} GB"
            vms_str = f"{results.get('peak_vms_gb', 0):.2f} GB"
        
        print(f"{name:<30} | {status:<10} | {time_str:<8} | {rss_str} | {vms_str}")

    print("="*70)
    print(f'Model is saved to {model_output_path}')
    print("="*70)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert Torch to Engine for VLM.")

    parser.add_argument("--model_name", 
                        type=str, 
                        default=MODEL_NAME,
                        choices=list(MODEL_CKPT_MAP.keys()),
                        help=f"Model name (default: {MODEL_NAME})")

    parser.add_argument("--no-vis-onnx",
                        action="store_true",
                        help="No need to convert vision part of VLM to ONNX")

    parser.add_argument("--no-vis-engine",
                        action="store_true",
                        help="No need to convert vision part of VLM to ENGINE (required ONNX)")
    
    parser.add_argument("--no-llm-engine",
                        action="store_true",
                        help="No need to convert language part of VLM to ENGINE")

    # For Converting Vision Part of VLM to ENGINE
    parser.add_argument('--vis_batch_size',
                        type=int,
                        default=VIS_BATCH_SIZE,
                        help='Batch size for the vision engine')

    parser.add_argument('--vis_dtype',
                        type=str,
                        default=VIS_DTYPE,
                        choices=VIS_DTYPE_CHOICES)

    parser.add_argument('--patch_tokens',
                        type=int,
                        default=PATCH_TOKENS,
                        help='Patch tokens nums')

    parser.add_argument('--keep_onnx',
                        action="store_true",
                        default=KEEP_ONNX,
                        help="Keep the ONNX file of Vision Engine after converting")

    # For Converting Language Part of VLM to ENGINE
    parser.add_argument('--tp_size',
                        type=int,
                        default=1,
                        help='N-way tensor parallelism size')

    parser.add_argument('--pp_size',
                        type=int,
                        default=PP_SIZE,
                        help='N-way pipeline parallelism size')

    parser.add_argument('--llm_dtype',
                        type=str,
                        default=LLM_DTYPE,
                        choices=LLM_DTYPE_CHOICES)

    parser.add_argument('--use_weight_only',
                        default=USE_WEIGHT_ONLY,
                        action="store_true",
                        help='Quantize weights for the various GEMMs to INT4/INT8.'
                        'See --weight_only_precision to set the precision')

    parser.add_argument('--disable_weight_only_quant_plugin',
                        default=False,
                        action="store_true",
                        help=
                        'By default, using plugin implementation for weight quantization. Enabling disable_weight_only_quant_plugin flag will use ootb implementation instead of plugin.'
                        'You must also use --use_weight_only for that argument to have an impact.')

    parser.add_argument('--weight_only_precision',
                        const='int8',
                        type=str,
                        nargs='?',
                        default=WEIGHT_ONLY_PRECISION,
                        choices=WEIGHT_ONLY_PRECISION_CHOICES,
                        help=
                        'Define the precision for the weights when using weight-only quantization.'
                        'You must also use --use_weight_only for that argument to have an impact.')

    parser.add_argument('--calib_dataset',
                        type=str,
                        default='ccdv/cnn_dailymail',
                        help=
                        "The huggingface dataset name or the local directory of the dataset for calibration.")

    parser.add_argument("--smoothquant",
                        "-sq",
                        type=float,
                        default=None,
                        help="Set the α parameter (see https://arxiv.org/pdf/2211.10438.pdf)"
                        " to Smoothquant the model, and output int8 weights."
                        " A good first try is 0.5. Must be in [0, 1]")

    parser.add_argument('--per_channel',
                        action="store_true",
                        default=False,
                        help=
                        'By default, we use a single static scaling factor for the GEMM\'s result. '
                        'per_channel instead uses a different static scaling factor for each channel. '
                        'The latter is usually more accurate, but a little slower.')

    parser.add_argument('--per_token',
                        action="store_true",
                        default=False,
                        help=
                        'By default, we use a single static scaling factor to scale activations in the int8 range. '
                        'per_token chooses at run time, and for each token, a custom scaling factor. '
                        'The latter is usually more accurate, but a little slower.')

    parser.add_argument('--int8_kv_cache',
                        default=INT8_KV_CACHE,
                        action="store_true",
                        help=
                        'By default, we use dtype for KV cache. int8_kv_cache chooses int8 quantization for KV')

    parser.add_argument('--per_group',
                        default=False,
                        action="store_true",
                        help=
                        'By default, we use a single static scaling factor to scale weights in the int4 range. '
                        'per_group chooses at run time, and for each group, a custom scaling factor. '
                        'The flag is built for GPTQ/AWQ quantization.')

    parser.add_argument('--group_size',
                        type=int,
                        default=128,
                        help='Group size used in GPTQ quantization.')

    parser.add_argument("--load_model_on_cpu", 
                        action="store_true",
                        default=LOAD_MODEL_ON_CPU)

    parser.add_argument('--use_parallel_embedding',
                        action="store_true",
                        default=False,
                        help=
                        'By default embedding parallelism is disabled. By setting this flag, embedding parallelism is enabled')

    parser.add_argument('--embedding_sharding_dim',
                        type=int,
                        default=0,
                        choices=[0, 1],
                        help=
                        'By default the embedding lookup table is sharded along vocab dimension (embedding_sharding_dim=0). '
                        'To shard it along hidden dimension, set embedding_sharding_dim=1'
                        'Note: embedding sharing is only enabled when embedding_sharding_dim = 0')

    parser.add_argument('--use_embedding_sharing',
                        action="store_true",
                        default=USE_PARALLEL_EMBEDDING,
                        help=
                        'Try to reduce the engine size by sharing the embedding lookup table between two layers.'
                        'Note: the flag might not take effect when the criteria are not met.')

    parser.add_argument('--workers',
                        type=int,
                        default=WORKERS,
                        help='The number of workers for converting checkpoint in parallel')

    parser.add_argument('--moe_tp_size',
                        type=int,
                        default=-1,
                        help=
                        'N-way tensor parallelism size for MOE, default is tp_size, which will do tp-only for MoE')

    parser.add_argument('--moe_ep_size',
                        type=int,
                        default=-1,
                        help=
                        'N-way expert parallelism size for MOE, default is 1, which will do tp-only for MoE')

    parser.add_argument('--num_frames',
                        type=int,
                        default=NUM_FRAMES,
                        help='Number of Frames')

    parser.add_argument('--llm_batch_size',
                        type=int,
                        default=LLM_BATCH_SIZE,
                        help='Maximum number of requests that the engine can schedule.')

    parser.add_argument('--max_input_len',
                        type=int,
                        default=MAX_INPUT_LEN,
                        help='Maximum input length of one request.')

    parser.add_argument('--max_seq_len',
                        type=int,
                        default=MAX_SEQ_LEN,
                        help='Maximum total length of one request, including prompt and outputs.')

    parser.add_argument('--max_multimodal_len',
                        type=int,
                        default=MAX_MULTIMODAL_LEN,
                        help='Maximum multimodal input size for multimodal models.')

    parser.add_argument('--context_fmha',
                        type=str,
                        default=CONTEXT_FMHA,
                        choices=BINARY_CHOICES,
                        help='Enable the fused multi-head attention during the context phase, will trigger a kernel that performs the MHA/MQA/GQA block using a single kernel.')

    parser.add_argument('--paged_kv_cache',
                        type=str,
                        default=PAGED_KV_CACHE,
                        choices=BINARY_CHOICES,
                        help='Enable Paged KV Cache')

    parser.add_argument('--remove_input_padding',
                        type=str,
                        default=REMOVE_INPUT_PADDING,
                        choices=BINARY_CHOICES,
                        help='Pack different tokens together, which reduces both the amount of computations and memory consumption.')

    parser.add_argument('--reduce_fusion',
                        type=str,
                        default=REDUCE_FUSION,
                        choices=BINARY_CHOICES,
                        help='Enable fusion of AllReduce and residual connection operations to reduce kernel launch overhead.')

    parser.add_argument('--use_paged_context_fmha',
                        type=str,
                        default=USE_PAGED_CONTEXT_FMHA,
                        choices=BINARY_CHOICES,
                        help='Enable Paged KV Cache support for the context (prefill) phase to reduce memory fragmentation and improve TTFT.')

    parser.add_argument('--enable_xqa',
                        type=str,
                        default=ENABLE_XQA,
                        choices=BINARY_CHOICES,
                        help='Enable high-performance XQA kernels for the generation (decoding) phase to accelerate token-to-token latency.')

    parser.add_argument('--gemm_plugin',
                        type=str,
                        default=GEMM_PLUGIN,
                        choices=GEMM_PLUGIN_CHOICES,
                        help='The GEMM plugin that utilizes NVIDIA cuBLASLt to perform GEMM operations. Note: it’s only affective for non-quantized gemm operations (except FP8).Note: For FP8, it also requires same calibration in checkpoint.')

    parser.add_argument('--moe_plugin',
                        type=str,
                        default=MOE_PLUGIN,
                        choices=MOE_PLUGIN_CHOICES,
                        help='Enable some customized kernels to speed up the MoE layer of MoE models.')

    parser.add_argument('--mamba_conv1d_plugin',
                        type=str,
                        default=MAMBA_CONV1D_PLUGIN,
                        choices=MAMBA_CONV1D_PLUGIN_CHOICES,
                        help='Enable customized kernels to speed up conv1d operator for Mamba.')

    parser.add_argument('--gpt_attention_plugin',
                        type=str,
                        default=GPT_ATTENTION_PLUGIN,
                        choices=GEMM_PLUGIN_CHOICES,
                        help='The plugin that uses efficient kernels and enables an in-place update of the KV cache for attention layer of GPT-like decoder models.')

    return parser.parse_args()

def main(args):

    assert args.max_multimodal_len == args.patch_tokens * args.num_frames * args.llm_batch_size
    assert args.patch_tokens * args.num_frames < args.max_input_len
    assert args.max_input_len < args.max_seq_len

    print(f'VIS Engine can process {args.vis_batch_size} (vis_batch_size) images in one run')
    print(f'LLM Engine can process {args.llm_batch_size} (llm_batch_size) requests in one run')
    print(f'LLM Engine can process {args.num_frames} (num_frames) images in one request')
    print(f'LLM Engine can process {args.max_input_len - args.patch_tokens * args.num_frames} (max_input_len - patch_tokens * num_frames) tokens in prompt and question for each request')
    print(f'LLM Engine can output {args.max_seq_len - args.max_input_len} (max_seq_len - max_input_len) tokens for each request')

    MODEL_CKPT_PATH = MODEL_CKPT_MAP[args.model_name]
    
    llm_act_dtype_suffix = DTYPE_SHORTFORM_MAP[args.llm_dtype]
    llm_weight_dtype_suffix = llm_act_dtype_suffix
    if args.use_weight_only:
        llm_weight_dtype_suffix = DTYPE_SHORTFORM_MAP[args.weight_only_precision]

    kv_cache_dtype_suffix = llm_act_dtype_suffix
    if args.int8_kv_cache:
        kv_cache_dtype_suffix = "i8"

    MODEL_OUTPUT_PATH = f"/mnt/sdcard/models/{args.model_name}" \
                        + f"_{DTYPE_SHORTFORM_MAP[args.vis_dtype]}" \
                        + f"_w({llm_weight_dtype_suffix})a({llm_act_dtype_suffix})" \
                        + f"_kv({kv_cache_dtype_suffix})"
    os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
    
    ONNX_PATH = os.path.join(os.getcwd(), f"{MODEL_OUTPUT_PATH}/vis.onnx")
    VIS_ENGINE_PATH = os.path.join(os.getcwd(), f"{MODEL_OUTPUT_PATH}/vis_engine")
    LLM_ENGINE_PATH = os.path.join(os.getcwd(), f"{MODEL_OUTPUT_PATH}/llm_engine")
    TKN_PATH = os.path.join(os.getcwd(), f"{MODEL_OUTPUT_PATH}/tokenizers")

    vis2onnx_build = {'skip': False, 'peak_rss_gb': 0, 'peak_vms_gb': 0, 'elapsed_time': 0}
    vis2engine_build = {'skip': False, 'peak_rss_gb': 0, 'peak_vms_gb': 0, 'elapsed_time': 0}
    llm2engine_build = {'skip': False, 'peak_rss_gb': 0, 'peak_vms_gb': 0, 'elapsed_time': 0}

    # Convert Vision part of VLM to ONNX
    if os.path.exists(ONNX_PATH) or args.no_vis_onnx:
        print(f'Skipping: Convert Vision part of VLM to ONNX')
        vis2onnx_build['skip'] = True
    else:
        model = AutoModel.from_pretrained(
            MODEL_CKPT_PATH,
            dtype=torch.float32, # dont change this, change this to float16 will make everything slower, thought peak virtual memory lower
            low_cpu_mem_usage=True,
            trust_remote_code=True).eval()
        model.forward = model.extract_feature

        MEAN = [[0.485 * 255, 0.456 * 255, 0.406 * 255]]
        STD = [[0.229 * 255, 0.224 * 255, 0.225 * 255]]

        N, C, H, W = [args.vis_batch_size, 3, 448, 448]

        stop_signal = threading.Event()
        monitor_thread = threading.Thread(target=monitor_pid_memory, args=(os.getpid(), vis2onnx_build, stop_signal))
        monitor_thread.start()

        try:
            torch.onnx.export(
                model, 
                torch.randn(N, C, H, W, device=model.device, dtype=model.dtype),  
                ONNX_PATH,
                input_names=['input'],
                output_names=['output']
            )
        finally:
            stop_signal.set()
            monitor_thread.join()

    # Convert Vision part of VLM to ENGINE
    if os.path.exists(VIS_ENGINE_PATH) or args.no_vis_engine:
        print(f'Skipping: Convert Vision part of VLM to ENGINE')
        vis2engine_build['skip'] = True
    else:
        assert os.path.exists(ONNX_PATH)
        os.makedirs(VIS_ENGINE_PATH, exist_ok=True)

        args_list = [
            '/usr/src/tensorrt/bin/trtexec',
            f'--onnx={ONNX_PATH}',
            f'--saveEngine={VIS_ENGINE_PATH}/model.engine',
            '--inputIOFormats=fp16:chw',
            '--outputIOFormats=fp16:chw',
            '--best',
            '--fp16'
        ]
        
        if args.vis_dtype == 'int8':
            args_list += ['--int8']

        proc = subprocess.Popen(args_list, text=True)

        monitor_thread = threading.Thread(target=monitor_memory, args=(proc, vis2engine_build))
        monitor_thread.start()

        stdout, stderr = proc.communicate()
        monitor_thread.join()

        with open(os.path.join(VIS_ENGINE_PATH, "config.json"), 'w') as f:
            json.dump({
                "builder_config": {
                    "model_type": "internvl",
                    "precision": args.vis_dtype,
                    "vis_batch_size": args.vis_batch_size,
                    "max_num_frames": args.num_frames,
                    "patch_tokens" : args.patch_tokens
                }
            }, f, indent=4)

    if not args.keep_onnx and os.path.exists(ONNX_PATH):
        os.remove(ONNX_PATH)
        print(f'keep_onnx set to {args.keep_onnx}, Removing {ONNX_PATH}')

    # Convert Language part of VLM to ENGINE
    if os.path.exists(LLM_ENGINE_PATH) or args.no_llm_engine:
        print(f'Skipping: Convert Language part of VLM to ENGINE')
        llm2engine_build['skip'] = True
    else:
        def args_to_quant_config(args: argparse.Namespace) -> QuantConfig:
            '''return config dict with quantization info based on the command line args
            '''
            quant_config = QuantConfig()
            if args.use_weight_only:
                if args.weight_only_precision == 'int8':
                    quant_config.quant_algo = QuantAlgo.W8A16
                elif args.weight_only_precision == 'int4':
                    quant_config.quant_algo = QuantAlgo.W4A16
            elif args.smoothquant:
                quant_config.smoothquant_val = args.smoothquant
                if args.per_channel:
                    if args.per_token:
                        quant_config.quant_algo = QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN
                    else:
                        quant_config.quant_algo = QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TENSOR_PLUGIN
                else:
                    if args.per_token:
                        quant_config.quant_algo = QuantAlgo.W8A8_SQ_PER_TENSOR_PER_TOKEN_PLUGIN
                    else:
                        quant_config.quant_algo = QuantAlgo.W8A8_SQ_PER_TENSOR_PLUGIN

            if args.int8_kv_cache:
                quant_config.kv_cache_quant_algo = QuantAlgo.INT8

            if args.weight_only_precision == 'int4_gptq':
                quant_config.group_size = args.group_size
                quant_config.has_zero_point = True
                quant_config.pre_quant_scale = False
                quant_config.quant_algo = QuantAlgo.W4A16_GPTQ

            return quant_config

        def patch_config_file(model_dir):
            config_file = os.path.join(model_dir, "configuration_internvl_chat.py")

            with open(config_file, "r") as f:
                lines= f.readlines()

            if any("def __getattr__" in line for line in lines):
                print(f"✅ Already patched: {config_file}")
                return

            class_idx = -1
            for i, line in enumerate(lines):
                if "class InternVLChatConfig" in line:
                    class_idx = i
                    break
            
            if class_idx == -1:
                print("❌ Could not find class definition.")
                raise NotImplementedError

            indent_style = "    "
            for i in range(class_idx + 1, len(lines)):
                if lines[i].strip():
                    match = re.match(r"(\s+)", lines[i])
                    if match:
                        indent_style = match.group(1)
                    break

            raw_patch = textwrap.dedent("""
                    def __getattr__(self, name):
                        if name == 'num_experts':
                            return 0
                        if 'llm_config' in self.__dict__:
                            try: return getattr(self.llm_config, name)
                            except AttributeError: pass
                        if 'vision_config' in self.__dict__:
                            try: return getattr(self.vision_config, name)
                            except AttributeError: pass
                        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
                """).strip()

            indented_patch = textwrap.indent(raw_patch, indent_style)

            lines.insert(class_idx + 1, f"\n{indented_patch}\n")

            with open(config_file, "w") as f:
                f.writelines(lines)
                
            print(f"✏️ Patched: {config_file}")

        model_dir = snapshot_download(repo_id=MODEL_CKPT_PATH)
        patch_config_file(model_dir)

        tmp_converted_dir = f'{MODEL_OUTPUT_PATH}/tmp_converted'
        os.makedirs(tmp_converted_dir, exist_ok=True)

        world_size = args.tp_size * args.pp_size
        override_fields = {
            'use_parallel_embedding': args.use_parallel_embedding,
            'embedding_sharding_dim': args.embedding_sharding_dim,
            'share_embedding_table': args.use_embedding_sharing,
            'disable_weight_only_quant_plugin': args.disable_weight_only_quant_plugin
        }

        # Qwen models have GPTQ-quantized checkpoint available on HF.
        use_hf_gptq_checkpoint = (args.use_weight_only
                                and args.weight_only_precision == 'int4_gptq')
        quant_config = args_to_quant_config(args)

        if args.smoothquant is not None or args.int8_kv_cache:
            mapping = Mapping(
                world_size=world_size,
                tp_size=args.tp_size,
                pp_size=args.pp_size,
                moe_tp_size=args.moe_tp_size,
                moe_ep_size=args.moe_ep_size,
            )
            QWenForCausalLM.quantize(model_dir,
                                    tmp_converted_dir,
                                    dtype=args.llm_dtype,
                                    mapping=mapping,
                                    quant_config=quant_config,
                                    calib_dataset=args.calib_dataset,
                                    **override_fields)
        else:
            # When not loading by shard, preload one complete model and then slice per rank weights from this
            # this saves the disk reloading time
            hf_model = load_hf_qwen(model_dir, args.load_model_on_cpu)

            def convert_and_save_rank(args, rank):
                mapping = Mapping(world_size=world_size,
                                rank=rank,
                                tp_size=args.tp_size,
                                pp_size=args.pp_size,
                                moe_tp_size=args.moe_tp_size,
                                moe_ep_size=args.moe_ep_size)
                qwen = QWenForCausalLM.from_hugging_face(
                    model_dir if hf_model is None else hf_model,
                    args.llm_dtype,
                    mapping=mapping,
                    quant_config=quant_config,
                    use_hf_gptq_checkpoint=use_hf_gptq_checkpoint,
                    low_cpu_mem_usage=False,
                    **override_fields)
                    
                qwen.save_checkpoint(tmp_converted_dir, save_config=(rank == 0))
                del qwen

            for rank in range(world_size):
                convert_and_save_rank(args, rank)
            
            release_gc()
            gc.collect()
            
        config_path = f'{tmp_converted_dir}/config.json'

        with open(config_path, 'r') as f:
            config_data = json.load(f)

        config_data['architecture'] = 'Qwen2ForCausalLM'

        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=4)

        proc = subprocess.Popen([
            'trtllm-build',
            f'--checkpoint_dir={tmp_converted_dir}',
            f'--output_dir={LLM_ENGINE_PATH}',
            f'--max_num_tokens={args.max_input_len}',
            f'--max_batch_size={args.llm_batch_size}',
            f'--max_input_len={args.max_input_len}',
            f'--max_seq_len={args.max_seq_len}',
            f'--max_multimodal_len={args.max_multimodal_len}',
            f'--workers={args.workers}',
            f'--enable_xqa={args.enable_xqa}',
            f'--context_fmha={args.context_fmha}',
            f'--reduce_fusion={args.reduce_fusion}',
            f'--paged_kv_cache={args.paged_kv_cache}',
            f'--remove_input_padding={args.remove_input_padding}',
            f'--use_paged_context_fmha={args.use_paged_context_fmha}',
            f'--moe_plugin={args.moe_plugin}',
            f'--gemm_plugin={args.gemm_plugin}',
            f'--mamba_conv1d_plugin={args.mamba_conv1d_plugin}',
            f'--gpt_attention_plugin={args.gpt_attention_plugin}'
        ], text=True)

        monitor_thread = threading.Thread(target=monitor_memory, args=(proc, llm2engine_build))
        monitor_thread.start()

        # Config for polling and timeouts
        HARD_TIMEOUT = 600    # 10 minutes
        CHECK_INTERVAL = 20   # 20 seconds
        start_time = time.time()
        early_exit = False
        last_engine_size = 0

        try:
            print(f"Monitoring build... (Hard timeout: {HARD_TIMEOUT}s)")
            
            while True:
                if proc.poll() is not None:
                    break

                config_path = os.path.join(LLM_ENGINE_PATH, 'config.json')
                engine_path = os.path.join(LLM_ENGINE_PATH, 'rank0.engine')
                
                if os.path.exists(config_path) and os.path.exists(engine_path):
                    if os.path.getsize(engine_path) == last_engine_size:
                        print("Detected complete rank0.engine. Terminating builder early...")
                        proc.terminate()
                        early_exit = True
                        break
                    last_engine_size = os.path.getsize(engine_path)

                # HARD TIMEOUT CHECK
                if (time.time() - start_time) > HARD_TIMEOUT:
                    print("Reached 10-minute hard timeout. Killing process.")
                    proc.kill()
                    break

                time.sleep(CHECK_INTERVAL)

            stdout, stderr = proc.communicate()
            monitor_thread.join()

            engine_exists = os.path.exists(os.path.join(LLM_ENGINE_PATH, 'rank0.engine'))
            
            if (proc.returncode == 0 or early_exit) and engine_exists:
                print("Build Successful!")
            else:
                print(f"Build FAILED. Return Code: {proc.returncode}")
                print(f"Error Log: {stderr}")
        finally:
            if os.path.exists(tmp_converted_dir):
                shutil.rmtree(tmp_converted_dir)
            if os.path.exists('model.cache'):
                os.remove('model.cache')

    # Tokenizers
    model_dir = snapshot_download(repo_id=MODEL_CKPT_PATH)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, fix_mistral_regex=True)
    tokenizer.save_pretrained(TKN_PATH)

    overview(MODEL_OUTPUT_PATH, vis2onnx_build, vis2engine_build, llm2engine_build)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)