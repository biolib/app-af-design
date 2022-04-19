FROM nvidia/cuda:11.6.0-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get update && apt-get install -y wget git curl build-essential
RUN apt-get install -y python3-pip

RUN git clone https://github.com/sokrypton/af_backprop.git 
RUN pip3 install -qU setuptools cuda-python
RUN pip3 install -qU biopython dm-haiku==0.0.5 py3Dmol ml-collections==0.1.0 tqdm matplotlib 
RUN pip3 install -qU tensorflow tbp-nightly dm-tree

RUN pip3 install -qU jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_releases.html
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcudnn8_8.4.0.27-1+cuda11.6_amd64.deb
RUN apt install ./libcudnn8_8.4.0.27-1+cuda11.6_amd64.deb


RUN mkdir params
RUN mkdir output

RUN curl -fsSL https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar | tar x -C params
RUN wget -qnc https://raw.githubusercontent.com/sokrypton/ColabFold/main/beta/colabfold.py
#RUN wget -qnc https://raw.githubusercontent.com/sokrypton/ColabDesign/beta/af/design.py
RUN wget -qnc https://raw.githubusercontent.com/sokrypton/ColabDesign/beta/af/bkg_prob.txt


COPY biolib/root.py root.py
# TODO For now hard copy design so I can do development on it, remember to comment this after and uncomment the above desing.py
COPY design.py design.py  

#CMD ["python3", "root.py"]