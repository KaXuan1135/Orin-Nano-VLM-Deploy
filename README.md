# Setup TensorRT-LLM 0.12.0 in Orin Nano (Jetpack 6.2.1)

## Upgrade to Super Version

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

## Setup TensorRT-LLM (https://github.com/NVIDIA/TensorRT-LLM/blob/v0.12.0-jetson/README4Jetson.md)

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

# Converting InternVL3-1B/2B to .engine

for 1B, convert success with 40GB of memory (incl. swap)
for 2B, convert success with 50GB of memory and with int4 (incl. swap)
* On success converted case, the printed memory will be significantly low (ex. 2B peak vram only 20GB), but if you dont have enough memory, the building process will get killed and print large vram required (2B peak vram is 44GB). 

# Inference with InternVL3-1B/2B

python engine_infer.py \
    --max_new_tokens 50 \
    --input_text "Where is the country of the image?How can you tell?" \
    --hf_model_name OpenGVLab/InternVL3-1B \
    --visual_engine_dir /home/pi/kx/pt2engine_vlm/models/InternVL3-1B_i4/InternVL3-1B_vis_engine \
    --llm_engine_dir /home/pi/kx/pt2engine_vlm/models/InternVL3-1B_i4/InternVL3-1B_llm_engine

python engine_infer.py \
    --max_new_tokens 50 \
    --input_text "Where is the country of the image?How can you tell?" \
    --hf_model_name OpenGVLab/InternVL3-2B \
    --visual_engine_dir /home/pi/kx/pt2engine_vlm/models/InternVL3-2B_i4/InternVL3-2B_vis_engine \
    --llm_engine_dir /home/pi/kx/pt2engine_vlm/models/InternVL3-2B_i4/InternVL3-2B_llm_engine