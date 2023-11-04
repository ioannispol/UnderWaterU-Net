
FROM nvcr.io/nvidia/pytorch:22.11-py3

# Install miniconda
ENV CONDA_DIR /opt/miniconda/

RUN apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update && \
    apt-get install -y wget  \
    git -y && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/miniconda

# Add conda to path
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda create -n uw-unet python=3.8

RUN rm -rf /workspace/*
WORKDIR /workspace/uw-unet

COPY requirements.txt requirements.txt
RUN echo "source activate uw-unet" >~/.bashrc && \
    /opt/miniconda/envs/uw-unet/bin/pip install -r requirements.txt

COPY . .

CMD [ "python3", "-m" , "UnderWaterU-Net" ]
