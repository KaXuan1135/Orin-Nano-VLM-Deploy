import os
import re
import sys
import time
import json
import torch
import shutil
import argparse
import subprocess
import tensorrt_llm
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel
from tensorrt_llm.mapping import Mapping
from tensorrt_llm._utils import release_gc
from huggingface_hub import snapshot_download
from tensorrt_llm.quantization import QuantAlgo
from tensorrt_llm.models import QWenForCausalLM
from tensorrt_llm.models.qwen.convert import load_hf_qwen
from tensorrt_llm.models.modeling_utils import QuantConfig
from concurrent.futures import ThreadPoolExecutor, as_completed

MODEL_CKPT_MAP = {
    'InternVL3-1B': 'OpenGVLab/InternVL3-1B',
}

assert (MODEL_NAME := 'InternVL3-1B') in list(MODEL_CKPT_MAP.keys())

MODEL_CKPT_PATH = MODEL_CKPT_MAP[MODEL_NAME]
ONNX_PATH = os.path.join(os.getcwd(), f"models/{MODEL_NAME}/{MODEL_NAME}_vis.onnx")
VIS_ENGINE_PATH = os.path.join(os.getcwd(), f"models/{MODEL_NAME}/{MODEL_NAME}_vis_engine")
LLM_ENGINE_PATH = os.path.join(os.getcwd(), f"models/{MODEL_NAME}/{MODEL_NAME}_llm_engine")

os.makedirs(f"models/{MODEL_NAME}/", exist_ok=True)

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
                        help="No need to convert language part of VLM to ONNX")

    # For Converting Language Part of VLM to ENGINE
    parser.add_argument('--tp_size',
                        type=int,
                        default=1,
                        help='N-way tensor parallelism size')

    parser.add_argument('--pp_size',
                        type=int,
                        default=1,
                        help='N-way pipeline parallelism size')

    parser.add_argument('--dtype',
                        type=str,
                        default='bfloat16',
                        choices=['float32', 'bfloat16', 'float16'])

    parser.add_argument('--use_weight_only',
                        default=False,
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
                        default='int8',
                        choices=['int8', 'int4', 'int4_gptq'],
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
                        default=False,
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
                        action="store_true")

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
                        default=False,
                        help=
                        'Try to reduce the engine size by sharing the embedding lookup table between two layers.'
                        'Note: the flag might not take effect when the criteria are not met.')

    parser.add_argument('--workers',
                        type=int,
                        default=1,
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

    parser.add_argument('--gemm_plugin',
                        type=str,
                        default='auto',
                        help='To confirm'
    )

    parser.add_argument('--max_batch_size',
                        type=int,
                        default=1,
                        help='To confirm'
    )

    parser.add_argument('--max_input_len',
                        type=int,
                        default=4096,
                        help='To confirm'
    )

    parser.add_argument('--max_seq_len',
                        type=int,
                        default=4068,
                        help='To confirm'
    )

    parser.add_argument('--max_multimodal_len',
                        type=int,
                        default=3328,
                        help='To confirm'
    )

    return parser.parse_args()

def main(args):

    # Convert Vision part of VLM to ONNX
    if os.path.exists(ONNX_PATH) or args.no_vis_onnx:
        print(f'Skipping: Convert Vision part of VLM to ONNX')
    else:
        start_time = time.time()
        model = AutoModel.from_pretrained(
            MODEL_CKPT_PATH,
            dtype=torch.float32, # try lower this when export multiple weights file instead of just one onnx
            low_cpu_mem_usage=True,
            trust_remote_code=True).eval()
            
        MEAN = [[0.485 * 255, 0.456 * 255, 0.406 * 255]]
        STD = [[0.229 * 255, 0.224 * 255, 0.225 * 255]]

        N, C, H, W = [1, 3, 448, 448]

        pixel_values = torch.randn(N, C, H, W, device=model.device, dtype=torch.float32)
        model.forward = model.extract_feature
        model = model.to(torch.float32).eval()
        torch.onnx.export(
            model, 
            pixel_values, 
            ONNX_PATH,
            input_names=['input'],
            output_names=['output']
        )

        print(f'Export vlm_vision_onnx to {ONNX_PATH} (Elapsed {int(time.time() - start_time)}s)')

    # Convert Vision part of VLM to ENGINE
    if os.path.exists(VIS_ENGINE_PATH) or args.no_vis_engine:
        print(f'Skipping: Convert Vision part of VLM to ENGINE')
    else:
        start_time = time.time()
        assert os.path.exists(ONNX_PATH)
        os.makedirs(VIS_ENGINE_PATH, exist_ok=True)
        subprocess.run([
            '/usr/src/tensorrt/bin/trtexec',
            f'--onnx={ONNX_PATH}',
            f'--saveEngine={VIS_ENGINE_PATH}/model.engine',
            '--fp16',
            '--inputIOFormats=fp16:chw',
            '--outputIOFormats=fp16:chw'
        ])

        with open(os.path.join(VIS_ENGINE_PATH, "config.json"), 'w') as f:
            json.dump({
                "builder_config": {
                    "model_type": "internvl",
                    "precision": "float16"
                }
            }, f, indent=4)

        print(f'Export vlm_vision_engine to {VIS_ENGINE_PATH} (Elapsed {int(time.time() - start_time)}s)')

    # Convert Language part of VLM to ENGINE
    if os.path.exists(LLM_ENGINE_PATH) or args.no_llm_engine:
        print(f'Skipping: Convert Language part of VLM to ENGINE')
    else:
        start_time = time.time()
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
                content = f.read()

            if "def __getattr__" in content:
                print("✅ File is already patched.")
            else:
                patch_code = """
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
                """
                pattern = r"(class InternVLChatConfig\(PretrainedConfig\):)"
                new_content = re.sub(pattern, r"\1" + patch_code, content)
                with open(config_file, "w") as f:
                        f.write(new_content)
                
                print(f"✏️ Patched: {config_file}")

        def execute(workers, func, args):
            if workers == 1:
                for rank, f in enumerate(func):
                    f(args, rank)
            else:
                with ThreadPoolExecutor(max_workers=workers) as p:
                    futures = [p.submit(f, args, rank) for rank, f in enumerate(func)]
                    exceptions = []
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as e:
                            traceback.print_exc()
                            exceptions.append(e)
                    assert len(
                        exceptions
                    ) == 0, "Checkpoint conversion failed, please check error log."

        model_dir = snapshot_download(repo_id=MODEL_CKPT_PATH)
        patch_config_file(model_dir)

        tmp_converted_dir = f'models/{MODEL_NAME}/tmp_converted'
        os.makedirs(tmp_converted_dir, exist_ok=True)

        load_model_on_cpu = args.load_model_on_cpu
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
            QWenForCausalLM.quantize(args.model_dir,
                                    tmp_converted_dir,
                                    dtype=args.dtype,
                                    mapping=mapping,
                                    quant_config=quant_config,
                                    calib_dataset=args.calib_dataset,
                                    **override_fields)
        else:
            # When not loading by shard, preload one complete model and then slice per rank weights from this
            # this saves the disk reloading time
            hf_model = load_hf_qwen(model_dir, load_model_on_cpu)

            def convert_and_save_rank(args, rank):
                mapping = Mapping(world_size=world_size,
                                rank=rank,
                                tp_size=args.tp_size,
                                pp_size=args.pp_size,
                                moe_tp_size=args.moe_tp_size,
                                moe_ep_size=args.moe_ep_size)
                qwen = QWenForCausalLM.from_hugging_face(
                    model_dir if hf_model is None else hf_model,
                    args.dtype,
                    mapping=mapping,
                    quant_config=quant_config,
                    use_hf_gptq_checkpoint=use_hf_gptq_checkpoint,
                    **override_fields)
                qwen.save_checkpoint(tmp_converted_dir, save_config=(rank == 0))
                del qwen

            execute(args.workers, [convert_and_save_rank] * world_size, args)
            release_gc()
    
        config_path = f'{tmp_converted_dir}/config.json'

        with open(config_path, 'r') as f:
            config_data = json.load(f)

        config_data['architecture'] = 'Qwen2ForCausalLM'

        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=4)

        subprocess.run([
            'trtllm-build',
            f'--checkpoint_dir={tmp_converted_dir}',
            f'--output_dir={LLM_ENGINE_PATH}',
            f'--gemm_plugin={args.gemm_plugin}',
            f'--max_batch_size={args.max_batch_size}',
            f'--max_input_len={args.max_input_len}',
            f'--max_seq_len={args.max_seq_len}',
            f'--max_multimodal_len={args.max_multimodal_len}'
        ])

        shutil.rmtree(tmp_converted_dir)
        if os.path.exists('model.cache'):
            os.remove('model.cache')

        print(f'Export vlm_language_engine to {LLM_ENGINE_PATH} (Elapsed {int(time.time() - start_time)}s)')

if __name__ == '__main__':
    args = parse_arguments()
    main(args)