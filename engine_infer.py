
import os
import argparse
import tensorrt_llm
import tensorrt_llm.profiler as profiler

from tensorrt_llm import logger
from huggingface_hub import snapshot_download
from tensorrt_llm.runtime import InternVLRunner

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_new_tokens', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--visual_engine_dir',
                        type=str,
                        default=None,
                        help='Directory containing visual TRT engines')
    parser.add_argument('--visual_engine_name',
                        type=str,
                        default='model.engine',
                        help='Name of visual TRT engine')
    parser.add_argument('--llm_engine_dir',
                        type=str,
                        default=None,
                        help='Directory containing TRT-LLM engines')
    parser.add_argument('--hf_model_name',
                        type=str,
                        default=None,
                        help="HuggingFace model name")
    parser.add_argument('--input_text',
                        type=str,
                        default=None,
                        help='Text prompt to LLM')
    parser.add_argument('--num_beams',
                        type=int,
                        help="Use beam search if num_beams >1",
                        default=1)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--top_p', type=float, default=0.0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--run_profiling',
                        action='store_true',
                        help='Profile runtime over several iterations')
    parser.add_argument('--profiling_iterations',
                        type=int,
                        help="Number of iterations to run profiling",
                        default=20)
    parser.add_argument("--image_path",
                        type=str,
                        default=None,
                        help='List of input image paths, separated by symbol')
    parser.add_argument("--path_sep",
                        type=str,
                        default=",",
                        help='Path separator symbol')
    parser.add_argument('--enable_context_fmha_fp32_acc',
                        action='store_true',
                        default=None,
                        help="Enable FMHA runner FP32 accumulation.")

    return parser.parse_args()

def print_result(model, input_text, output_text, args):
    logger.info("---------------------------------------------------------")
    logger.info(f"\n[Q] {input_text}")
    for i in range(len(output_text)):
        logger.info(f"\n[A]: {output_text[i]}")

    total_generated_tokens = 0
    if args.num_beams == 1:
        for i, txt in enumerate(output_text):
            output_ids = model.tokenizer(txt[0], add_special_tokens=False)['input_ids']
            total_generated_tokens += len(output_ids)
            print(f'Batch {i} : {len(output_ids)} tokens')
        logger.info(f"Total {len(output_text)} output, generated {total_generated_tokens} tokens")

    if args.run_profiling:
        msec_per_batch = lambda name: 1000 * profiler.elapsed_time_in_sec(
            name) / args.profiling_iterations

        logger.info('Latencies per batch (msec)')
        logger.info('TRT vision encoder: %.1f ms' % (msec_per_batch('Vision')))
        logger.info('TRTLLM LLM generate: %.1f ms' % (msec_per_batch('LLM')))
        logger.info('Multimodal generate: %.1f ms' % (msec_per_batch('Generate')))
        logger.info('Time to first token (TTFT): %.1f ms' % (msec_per_batch('TTFT')))

        if total_generated_tokens > 0:
            tps = total_generated_tokens / (msec_per_batch('LLM') / 1000.0)
            ms_per_token = msec_per_batch('LLM') / total_generated_tokens
            
            logger.info('--- Performance Metrics ---')
            logger.info('Tokens per second (TPS): %.2f' % tps)
            logger.info('Time per output token (TPOT): %.1f ms/token' % ms_per_token)


    logger.info("---------------------------------------------------------")

if __name__ == '__main__':
    # set to true for maximum inference speed, but possibly causing deadlock
    # if you are not doing multithread, setting to true should be fine.
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    args = parse_arguments()
    logger.set_level(args.log_level)

    args.hf_model_dir = snapshot_download(repo_id=args.hf_model_name)

    model = InternVLRunner(args)
    raw_image = model.load_test_image()

    image_paths = [
        ['/home/pi/kx/sample_images/cat.jpg', 
        # '/home/pi/kx/sample_images/tiger.jpg',
        # '/home/pi/kx/sample_images/apple.jpg',
        # '/home/pi/kx/sample_images/orange.jpg',
        # '/home/pi/kx/sample_images/airplane.jpg',
        # '/home/pi/kx/sample_images/car.jpg'
        ],

    ]

    args.input_text = [
        'Describe the images.',
        # 'Please describe the images.',
        # 'Can you please describe the images?',
        # 'Hi, can you describe the images for me?',
    ]

    num_iters = args.profiling_iterations if args.run_profiling else 1
    for _ in range(num_iters):

        input_text, output_text = model.run(args.input_text, image_paths,
                                            args.max_new_tokens)

    if tensorrt_llm.mpi_rank() == 0:
        print_result(model, input_text, output_text, args)


