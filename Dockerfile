FROM nvidia/cuda:10.2-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get update && apt-get install -y python3-pip python-pip wget git curl

RUN git clone https://github.com/sokrypton/af_backprop.git
RUN pip3 install -qU setuptools
RUN pip3 install biopython dm-haiku==0.0.5 py3Dmol ml-collections==0.1.0

RUN mkdir params
RUN curl -fsSL https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar | tar x -C params
RUN wget -qnc https://raw.githubusercontent.com/sokrypton/ColabFold/main/beta/colabfold.py
RUN wget -qnc https://raw.githubusercontent.com/sokrypton/ColabDesign/main/af/design.py

COPY biolib/root.py root.py
COPY design.py design.py
