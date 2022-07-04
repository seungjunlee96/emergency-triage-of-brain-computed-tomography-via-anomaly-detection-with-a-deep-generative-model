FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04
MAINTAINER ubuntu
ARG PYTHON_VERSION=3.8

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include ninja cython typing && \
     /opt/conda/bin/conda install -y -c conda-forge jupyterlab && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

WORKDIR /workspace
RUN chmod -R a+w .

# to skip build error for timezone
ENV TZ=Europe/Minsk
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt search vim
RUN apt-cache search vim

RUN apt-get update -y
RUN apt-get install -y ufw

RUN python3 -m pip install --upgrade pip
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

RUN pip install wandb
RUN pip install numpy matplotlib tqdm pandas sklearn opencv-python scikit-image openpyxl seaborn
RUN pip install SimpleITK nibabel pydicom
RUN pip install pickle5
RUN pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3

CMD ["python3"]
RUN apt update
RUN apt install -y git libgtk2.0-dev libsm6 libxext6 libxrender-dev python-tk libglib2.0-0 libsm6 libxrender1 libfontconfig1 dcm2niix

#https://github.com/conda-forge/pygridgen-feedstock/issues/10
RUN apt update -y
RUN apt install -y libgl1-mesa-glx

# for user info
RUN apt-get install -y sudo

RUN sudo addgroup --gid 1000001 gusers
RUN sudo adduser --uid 1000073 --gid 1000001 --disabled-password --gecos '' ubuntu
RUN sudo adduser ubuntu sudo
USER ubuntu                                                                                                                                                         