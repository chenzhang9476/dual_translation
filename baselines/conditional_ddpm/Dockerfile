FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget git ca-certificates bzip2 \
    && rm -rf /var/lib/apt/lists/*

# ---- Miniconda ----
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH=/opt/conda/bin:$PATH

# ★★★ これを追加 ★★★
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# ---- conda更新 ----
# RUN conda update -y conda

WORKDIR /workspace
COPY environment.yml .

RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "pytorch-img2img", "/bin/bash", "-c"]

CMD ["/bin/bash"]
