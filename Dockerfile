FROM nvidia/cuda:11.3.1-base-ubuntu20.04

RUN apt update
RUN apt install software-properties-common -y
RUN apt install build-essential time -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt install -y python3.7 python3.7-distutils python3.7-venv
RUN python3.7 -m ensurepip
RUN python3.7 -m venv RUN /opt/venv
RUN /opt/venv/bin/pip install -U pip
RUN /opt/venv/bin/pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN /opt/venv/bin/pip install auto_LiRPA==0.3.0
RUN /opt/venv/bin/pip install numpy==1.21.6
RUN /opt/venv/bin/pip install z3-solver==4.8.17.0
RUN /opt/venv/bin/pip install torchinfo
RUN /opt/venv/bin/pip install tqdm

RUN echo "export PATH=/opt/venv/bin:$PATH" >> ~/.bashrc
RUN echo "source /opt/venv/bin/activate" >> ~/.bashrc

COPY . /comp-indinv-verif-nncs
WORKDIR /comp-indinv-verif-nncs