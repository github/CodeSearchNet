FROM python:3.7.3

RUN touch /etc/inside-container

RUN set -ex && pip3 install --upgrade pip
RUN set -ex && pip3 --no-cache-dir install --upgrade jupyter \
    tree_sitter==0.0.5 \
    requests \
    pyhive \
    tqdm \
    pandas \
    python-arango \
    docopt \
    elasticsearch \
    dpu_utils

RUN mkdir -p /src/vendor
RUN mkdir -p /src/build
RUN cd /src/vendor && git clone https://github.com/tree-sitter/tree-sitter-python.git
RUN cd /src/vendor && git clone https://github.com/tree-sitter/tree-sitter-javascript.git
RUN cd /src/vendor && git clone https://github.com/tree-sitter/tree-sitter-typescript.git
RUN cd /src/vendor && git clone https://github.com/tree-sitter/tree-sitter-go.git
RUN cd /src/vendor && git clone https://github.com/tree-sitter/tree-sitter-ruby.git
RUN cd /src/vendor && git clone https://github.com/tree-sitter/tree-sitter-java.git
RUN cd /src/vendor && git clone https://github.com/tree-sitter/tree-sitter-cpp.git
RUN cd /src/vendor && git clone https://github.com/tree-sitter/tree-sitter-c-sharp.git
RUN cd /src/vendor && git clone https://github.com/tree-sitter/tree-sitter-php.git

COPY script/setup.py /src/function-parser/script/setup.py

RUN cd /src/function-parser/script && python setup.py
WORKDIR /src/function-parser

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
