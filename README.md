# Summary
```
This project deploys InternVL3-1B/2B vision-language models on NVIDIA Jetson Orin Nano (8GB) using TensorRT-LLM, establishing the practical operating envelope of edge VLM inference under tight memory constraints.

Key results:
- Achieved 5–6× inference speedup over HuggingFace baseline via TensorRT engines
- Sustained 600+ tokens/sec throughput with batched INT8 inference
- Identified the device’s memory and batch scaling limits
- Built a reproducible pipeline for model conversion, benchmarking, and profiling

Key engineering insights:
- Edge VLM is memory-bound at low batch and compute-bound at high batch
- KV cache dominates memory usage beyond moderate batch sizes
- TensorRT-LLM engine builds become unstable under heavy swap pressure
- INT4 vs INT8 performance depends on saturation regime and bandwidth limits

This repository documents the full deployment workflow, benchmarking methodology, and system-level analysis.
```

# Setup

## Setup Orin Nano (Jetpack 6.2.1)

Pre-Knowledge:

- If has SD card slot, then is [Jetson Orin Nano 8gb Developer Kit Version], else [Jetson Orin Nano 8gb]

1. Power off (unplug) the device, set the device in recovery mode (using jumper), connect usb to windows pc, then power on (plug in) the device. (make sure the usb-type c is plug in, you might need to push it in hard)
2. Install Jetpack 6.2.1 using windows' nvidia sdk manager (https://docs.nvidia.com/sdk-manager/install-with-sdkm-jetson/index.html)
    - include as well the jetson sdk components : CUDA, CUDA-X AI, Computer Vision
    - sometimes need to install apx driver, follow the instruction will do
    - oem configuration : runtime, storage device : nvme (if use ssd card) or sd card (if use sd card)
    - the process on windows should run until you need to continue on the jetson device
    - ! if observed warning about non-optimal usb or what, unplug the power of jetson, redo everything and plug in again, it should works now
3. Connect the monitor to the device to check progress
    - might stuck after finished Daily apt download activities for about an hour, be patience.
    - finished discard unused blocks on filesystems from /etc/fstab, stuck after for an hour already, lets see
    - unplug and replug the ethernet cable can skip it?
    - then something something failed, then can restart the device
    - the progress might stuck, ensure the ethernet is plug in and you wait at least 2 hours, then you are able to plug out the power of the device then plug in again to restart(regardless of it is stucking or not, the important process should have done, leaving not so important process stucking)
4. After restarting / finish progress, setup typical first-time login procedure.
5. After getting in, then continue the installltion of sdk on windows'pc
    - when windows pc installation done, all done
6. Install Jtop in jetson device
    - When using Jtop, you might observe the jetpack not installed, solve with https://my.cytron.io/tutorial/fixing-jtop-for-jetson-jetpack-6-2
7. On right top of the jetson's windows, set power options to maxn super
8. In CMD, run 'sudo jetson_clocks' to boost manually, and use Jtop to check performance, the cpu freq should be 1.7ghz and gpu should be 1.0ghz.


## Setup TensorRT-LLM 0.12.0 (https://github.com/NVIDIA/TensorRT-LLM/blob/v0.12.0-jetson/README4Jetson.md)

```bash
sudo apt-get update
sudo apt-get install -y python3-pip libopenblas-dev git-lfs ccache
wget https://raw.githubusercontent.com/pytorch/pytorch/9b424aac1d70f360479dd919d6b7933b5a9181ac/.ci/docker/common/install_cusparselt.sh
export CUDA_VERSION=12.6
sudo -E bash ./install_cusparselt.sh

python3 -m pip install "https://developer.download.nvidia.cn/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"
python3 -m pip install numpy=='1.26.1'

mkdir -p ./build_env
pip freeze | grep -E "torch|numpy" > ./build_env/constraints.txt
export PIP_CONSTRAINT=$(pwd)/build_env/constraints.txt

# Temporary increase memory with swap space
sudo dd if=/dev/zero of=/var/swap_temp.img bs=1M count=30720
sudo chmod 600 /var/swap_temp.img
sudo mkswap /var/swap_temp.img
sudo swapon /var/swap_temp.img

git clone https://github.com/KaXuan1135/TensorRT-LLM.git
cd TensorRT-LLM
git checkout v0.12.0-jetson
git lfs pull

python3 scripts/build_wheel.py --clean --cuda_architectures 87 -DENABLE_MULTI_DEVICE=0 --build_type Release --benchmarks --use_ccache
# pip install build/tensorrt_llm-*.whl
pip install -e .

# Remove the temporary swap space
sudo swapoff /var/swap_temp.img
sudo rm /var/swap_temp.img

# Manually degrade to fit in requirement of TensorRT-LLM
pip install nvidia-ml-py cuda-python==12.6.0 'transformers>=4.45.0,<5.0.0'

# Add trtllm-engine to PATH
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

# Conversion
```
python pt2engine.py \
--model_name InternVL3-2B \
--num_frames 1 \
--llm_batch_size 1 \
--vis_batch_size 1 \
--max_multimodal_len 256 \
--max_input_len 356 \
--max_seq_len 856 \
--use_weight_only \
--weight_only_precision int4
```

# Inference
```
python engine_infer.py \
--hf_model_name OpenGVLab/InternVL3-2B \
--visual_engine_dir /home/pi/kx/pt2engine_vlm/models/InternVL3-2B_i8/InternVL3-2B_vis_engine \
--llm_engine_dir /home/pi/kx/pt2engine_vlm/models/InternVL3-2B_i8/InternVL3-2B_llm_engine
```

# Benchmark
- The Jetson Orin Nano is upgrade to Jetpack 6.2.1 Super
  - 8GB Ram, with a 40GB Swap Size
- RK3588 32GB

## Conversion Benchmark
- num_frames = 1
- llm_batch_size = 1
- vis_batch_size = 1
- max_multimodal_len = 256
- max_input_len = 256 + 100
- max_seq_len = 256 + 100 + 500

### InternVL3-1B, No Quantized (bf16)
```
==================================================
       CONVERSION PROCESS SUMMARY OVERVIEW
==================================================
Stage                          | Status     | Time     | Peak RSS | Peak VMS
----------------------------------------------------------------------
Vision to ONNX                 | DONE       | 89s      | 6.60 GB | 22.01 GB
Vision to Engine (trtexec)     | DONE       | 169s     | 5.07 GB | 12.98 GB
LLM to Engine (trtllm-build)   | DONE       | 183s     | 3.27 GB | 18.48 GB
======================================================================
```

### InternVL3-1B, Quantized (int8)
```
==================================================
       CONVERSION PROCESS SUMMARY OVERVIEW
==================================================
Stage                          | Status     | Time     | Peak RSS | Peak VMS
----------------------------------------------------------------------
Vision to ONNX                 | DONE       | 80s      | 6.53 GB | 21.75 GB
Vision to Engine (trtexec)     | DONE       | 175s     | 5.09 GB | 12.98 GB
LLM to Engine (trtllm-build)   | DONE       | 67s      | 2.69 GB | 16.37 GB
======================================================================
```

### InternVL3-1B, Quantized (int4)
```
==================================================
       CONVERSION PROCESS SUMMARY OVERVIEW
==================================================
Stage                          | Status     | Time     | Peak RSS | Peak VMS
----------------------------------------------------------------------
Vision to ONNX                 | DONE       | 57s      | 6.60 GB | 22.01 GB
Vision to Engine (trtexec)     | DONE       | 193s     | 5.15 GB | 13.25 GB
LLM to Engine (trtllm-build)   | DONE       | 98s      | 2.47 GB | 16.04 GB
======================================================================
```
### InternVL-2B, Quantized (int8)
```
The conversion of InternVL3-2B-INT8 to a TensorRT engine on the Orin Nano (8GB) is inconsistent. 
Even with a 50GB swap file, the process may crash or produce a "partially complete" yet functional .engine file.

Observed Stability Pattern:
- Rebooting the system.
- Warm-up Effect: Successfully converting a smaller/working model (e.g., InternVL3-1B) immediately before the 2B model significantly increases the 2B model's success rate.
```

### InternVL-2B, Quantized (int4)
```
==================================================
       CONVERSION PROCESS SUMMARY OVERVIEW
==================================================
Stage                          | Status     | Time     | Peak RSS | Peak VMS
----------------------------------------------------------------------
Vision to ONNX                 | DONE       | 99s      | 5.96 GB | 26.50 GB
Vision to Engine (trtexec)     | DONE       | 178s     | 5.05 GB | 13.03 GB
LLM to Engine (trtllm-build)   | DONE       | 235s     | 2.15 GB | 17.70 GB
======================================================================
```

## Inference Tokens per second & Memory Usage
- Batch Size: 1 (Single sequence processing)
- Input: 1 Image (Frame) + Prompt: "Describe this image in detail."

|   Platform  |  Device    |    Model     |  Dtype |  TPS (Super) | Out Tokens | Memory (GB) | 
|     :---    |   :---     |    :---      |  :---: |     :---:    |   :---:    |    :---:    | 
| HuggingFace |  Orin Nano | InternVL3-1B |  BF16  |  8.62(11.94) |    115     |     2.2     |
| HuggingFace |  Orin Nano | InternVL3-2B |  BF16  |  6.75(9.68)  |     89     |     4.3     |
|   TensorRT  |  Orin Nano | InternVL3-1B |  BF16  | 38.19(43.53) |     76     |     2.9     |
|   TensorRT  |  Orin Nano | InternVL3-1B |  INT8  | 46.36(53.31) |     67     |     2.3     |
|   TensorRT  |  Orin Nano | InternVL3-1B |  INT4  | 50.02(58.80) |     58     |     2.1     |
|   TensorRT  |  Orin Nano | InternVL3-2B |  INT8  | 35.34(39.93) |     64     |     3.4     |
|   TensorRT  |  Orin Nano | InternVL3-2B |  INT4  | 44.14(55.26) |     70     |     2.9     |

|   Platform  |  Device    |    Model     |  Dtype |   TPS  | Out Tokens | Memory (GB) | 
|     :---    |   :---     |    :---      |  :---: |  :---: |   :---:    |    :---:    | 
|     RKNN    |   RK3588   | InternVL3-1B |  INT8  |  11.79 |     56     |     1.6     |
|     RKNN    |   RK3588   | InternVL3-2B |  INT8  |   7.36 |     43     |     2.6     | 

* TPS: Tokens per second
* Dtype: Data Precision of the model, the higher the precision, the better the model, but lower the speed.
* Values in parentheses (XX.XX) represent performance in Super Mode (MAXN)

## Maximizing Throughput by Batch Inference
- Batch Size: N
- Input: 6 Image (Frame) + Prompt: "Describe the images in detail."

| Platform |     Model    | Batch Size | Frames | Dtype |  TTFT (sec) |  TPS  | Out Tokens |  Memory (GB) |
|   :---   |     :---     |    :---:   |  :---: | :---: |    :---:    | :---: |    :---:   |     :---:    |
| TensorRT | InternVL3-1B |      2     |    6   | INT8  |    0.40     |  149  |     807    |      2.5     |
| TensorRT | InternVL3-1B |      4     |    6   | INT8  |    0.79     |  259  |    1804    |      2.8     |
| TensorRT | InternVL3-1B |      8     |    6   | INT8  |    1.58     |  447  |    3368    |      3.4     |
| TensorRT | InternVL3-1B |     12     |    6   | INT8  |    2.38     |  573  |    5342    |      3.6     |
| TensorRT | InternVL3-1B |     20     |    6   | INT8  |    3.95     |  672  |    8540    |      5.1     |
| TensorRT | InternVL3-1B |     24     |    6   | INT8  |    4.69     |  702  |   10279    |      5.5     |
| TensorRT | InternVL3-1B |      2     |    6   | INT4  |    0.39     |  189  |    1000    |      2.5     |
| TensorRT | InternVL3-1B |      4     |    6   | INT4  |    0.81     |  263  |    1854    |      2.7     |
| TensorRT | InternVL3-1B |      8     |    6   | INT4  |    1.62     |  462  |    3738    |      3.3     |
| TensorRT | InternVL3-1B |     12     |    6   | INT4  |    2.39     |  586  |    5607    |      3.9     |
| TensorRT | InternVL3-1B |     24     |    6   | INT4  |    4.70     |  647  |    8640    |      5.4     |

* Int4 faster than Int8 in lower batch size due to Bandwidth Limited. Moving 4-bit weights is much faster.
* Int4 slower than Int8 in higher batch size due to Compute/Kernel Limited. Int8 kernels are more mature and have less overhead at high saturation.
* both Int4 and Int8 has no significantly memory used as the KV Cache dominants the memory usage

## Roadblocks
1. InternVL 3.5 Mismatch: Direct weight migration currently fails. Standard "forced alignment" results in corrupted output. A dedicated conversion script for the 3.5 architecture is necessary to align the layer mapping correctly.
2. The conversion of InternVL3-2B-INT8 to a TensorRT engine on the Orin Nano (8GB) is inconsistent. Even with a 50GB swap file, the process may crash or produce a "partially complete" yet functional .engine file.

## Further Optimized Path
1. Quantized using AWQ can further increase the accuracy and precision of model.
2. Setting KVCache to int8 can further increase the speed or decreese the memory.