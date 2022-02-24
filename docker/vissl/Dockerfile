FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

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
RUN pip install -f https://dl.fbaipublicfiles.com/vissl/packaging/apexwheels/py37_cu111_pyt191/download.html apex

# install vissl (from sorce for the latest version)
WORKDIR /home/vissluser/tmp/installvissl
RUN sudo git clone --recursive https://github.com/facebookresearch/vissl.git
WORKDIR /home/vissluser/tmp/installvissl/vissl
RUN pip install --user classy-vision@https://github.com/facebookresearch/ClassyVision/tarball/main
RUN pip install --user fairscale@https://github.com/facebookresearch/fairscale/tarball/main
RUN pip install --user --no-cache-dir .
WORKDIR /home/vissluser/tmp
RUN sudo rm -r installvissl

COPY ./docker/vissl/requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /home/vissluser/workspace
