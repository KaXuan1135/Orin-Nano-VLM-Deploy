# --- Building ---
FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential python3-dev python3-pip libopenblas-dev \
    git git-lfs ccache wget cmake curl && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
RUN wget https://raw.githubusercontent.com/pytorch/pytorch/9b424aac1d70f360479dd919d6b7933b5a9181ac/.ci/docker/common/install_cusparselt.sh \
    && export CUDA_VERSION=12.6 && bash ./install_cusparselt.sh

RUN python3 -m pip install "https://developer.download.nvidia.cn/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl" \
    && python3 -m pip install numpy=='1.26.1'

WORKDIR /opt
RUN git lfs install && \
    git clone --recursive --branch v0.12.0-jetson https://github.com/KaXuan1135/TensorRT-LLM.git && \
    cd TensorRT-LLM && \
    git lfs pull

# WORKDIR /opt/TensorRT-LLM
# RUN git submodule sync && \
#     git submodule update --init --recursive && \
#     python3 scripts/build_wheel.py --clean --cuda_architectures 87 \
#     -DENABLE_MULTI_DEVICE=0 --build_type Release --benchmarks --use_ccache

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /opt

RUN git clone --recursive https://github.com/KaXuan1135/Orin-Nano-VLM-Deploy.git

WORKDIR /opt/Orin-Nano-VLM-Deploy

RUN ln -s /usr/bin/gcc /usr/bin/aarch64-unknown-linux-gnu-gcc && \
    ln -s /usr/bin/g++ /usr/bin/aarch64-unknown-linux-gnu-g++ && \
    ln -s /usr/bin/ar /usr/bin/aarch64-unknown-linux-gnu-ar

# shoudl move this to top section
RUN apt-get update && apt-get install -y nlohmann-json3-dev

# # --- Runtime ---
# FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

# ENV DEBIAN_FRONTEND=noninteractive
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     python3-pip libopenblas-dev libgomp1 && rm -rf /var/lib/apt/lists/*

# COPY --from=builder /opt/tensorrt_llm/build/tensorrt_llm-*.whl /tmp/
# RUN pip3 install /tmp/tensorrt_llm-*.whl && rm /tmp/tensorrt_llm-*.whl

# RUN pip3 install nvidia-ml-py cuda-python==12.6.0 'transformers>=4.45.0,<5.0.0' fastapi uvicorn gradio

# WORKDIR /app
# COPY . .
# ENV LD_LIBRARY_PATH=/app/build/lib:$LD_LIBRARY_PATH

# EXPOSE 8000
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]