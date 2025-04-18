FROM nvcr.io/nvidia/dgl:25.01-py3

WORKDIR /workspace

COPY . /workspace/Fused3S

RUN pip install torch_geometric
RUN pip install ogb

WORKDIR /workspace/Fused3S/src
RUN bash build.sh

WORKDIR /workspace/Fused3S/baselines/flashSparse/FlashSparse
RUN bash compile.sh

WORKDIR /workspace/Fused3S/baselines/DF-GNN
RUN bash install.sh

WORKDIR /workspace
