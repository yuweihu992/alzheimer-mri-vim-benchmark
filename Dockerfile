ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/src \
    TORCH_CUDA_ARCH_LIST="9.0" \
    MAX_JOBS=8 \
    CAUSAL_CONV1D_FORCE_BUILD=TRUE \
    MAMBA_FORCE_BUILD=TRUE

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libglib2.0-0 \
        libgl1 \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel packaging ninja \
    && python -m pip install -r /workspace/requirements.txt \
    && python -m pip install causal-conv1d==1.6.2.post1 --no-build-isolation --no-deps --no-binary causal-conv1d \
    && python -m pip install mamba-ssm==2.2.6.post3 --no-build-isolation --no-deps --no-binary mamba-ssm

COPY . /workspace

CMD ["python", "tools/smoke_test.py"]
