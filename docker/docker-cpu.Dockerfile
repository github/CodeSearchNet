FROM python:3.6
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
# Install Python packages
RUN pip --no-cache-dir install --upgrade \
    docopt \
    dpu-utils \
    ipdb \
    wandb \
    https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.12.0-cp36-cp36m-linux_x86_64.whl \
    typed_ast \
    more_itertools \
    scipy \
    toolz \
    tqdm \
    pandas \
    parso \
    pytest \
    mypy

RUN pip --no-cache-dir install --upgrade \
    ipdb

COPY src/docs/THIRD_PARTY_NOTICE.md .
COPY . /
WORKDIR /src
CMD bash