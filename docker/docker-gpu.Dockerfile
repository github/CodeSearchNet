FROM tensorflow/tensorflow:1.12.0-gpu-py3
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN add-apt-repository -y ppa:git-core/ppa
RUN add-apt-repository -y ppa:deadsnakes/ppa

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    byobu \
    ca-certificates \
    git-core git \
    htop \
    libglib2.0-0 \
    libjpeg-dev \
    libpng-dev \
    libxext6 \
    libsm6 \
    libxrender1 \
    libcupti-dev \
    openssh-server \
    python3.6 \
    python3.6-dev \
    software-properties-common \
    vim \
    unzip \
    && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*

RUN apt-get -y update

#  Setup Python 3.6 (Need for other dependencies)
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.5 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 2
RUN apt-get install -y python3-setuptools
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3 get-pip.py
RUN pip install --upgrade pip

# Pin TF Version on v1.12.0
RUN pip --no-cache-dir install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.12.0-cp36-cp36m-linux_x86_64.whl

# Other python packages
RUN pip --no-cache-dir install --upgrade \
    altair==3.2.0 \
    annoy==1.16.0 \
    docopt==0.6.2 \
    dpu_utils==0.2.17 \
    ipdb==0.12.2 \
    jsonpath_rw_ext==1.2.2 \
    jupyter==1.0.0 \
    more_itertools==7.2.0 \
    numpy==1.16.5 \
    pandas==0.25.0 \
    parso==0.5.1 \
    pygments==2.4.2 \
    pyyaml==5.3 \
    requests==2.22.0 \
    scipy==1.3.1 \
    SetSimilaritySearch==0.1.7 \
    toolz==0.10.0 \
    tqdm==4.34.0 \
    typed_ast==1.4.0 \
    wandb==0.8.12 \
    wget==3.2

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Open Ports for TensorBoard, Jupyter, and SSH
EXPOSE 6006
EXPOSE 7654
EXPOSE 22

WORKDIR /home/dev/src
COPY src/docs/THIRD_PARTY_NOTICE.md .

CMD bash
