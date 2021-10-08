FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    sudo \
    git

ARG USER_ID
RUN useradd -m --no-log-init --system --uid ${USER_ID} vissluser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER vissluser

# install apex
WORKDIR /home/vissluser/tmp/installapex
RUN sudo git clone https://github.com/NVIDIA/apex.git
WORKDIR /home/vissluser/tmp/installapex/apex
ARG TORCH_CUDA_ARCH_LIST="7.5+PTX"
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
WORKDIR /home/vissluser/tmp
RUN sudo rm -r installapex

# install vissl (from sorce for the latest version)
WORKDIR /home/vissluser/tmp/installvissl
RUN sudo git clone --recursive https://github.com/facebookresearch/vissl.git
WORKDIR /home/vissluser/tmp/installvissl/vissl
RUN pip install --user classy-vision@https://github.com/facebookresearch/ClassyVision/tarball/main
RUN pip install --user -v --no-cache-dir .
WORKDIR /home/vissluser/tmp
RUN sudo rm -r installvissl

COPY ./docker/vissl/requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /home/vissluser/workspace
