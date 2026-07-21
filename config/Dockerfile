# ===== Base image =====
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

# ===== System dependencies =====
RUN apt-get update && apt-get install -y \
    git vim \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# ===== Python dependencies =====
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# ===== Workspace =====
WORKDIR /workspace

# ===== GPU/runtime defaults =====
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
ENV MPLCONFIGDIR=/workspace/.config/matplotlib
ENV PYTHONUNBUFFERED=1
ENV NCCL_DEBUG=INFO
ENV NCCL_IB_DISABLE=1
ENV OMP_NUM_THREADS=1

# Start an interactive shell by default.
# CMD ["/bin/bash"]
