FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get -y update \

 && apt-get install --fix-missing -y wget vim curl pv unzip build-essential cmake pkg-config ca-certificates gedit x11-apps \

 && apt-get clean && rm -rf /tmp/* /var/tmp/* \
 && python3 -m pip install --upgrade pip \
 && python3 -m pip install --upgrade setuptools


RUN pip3 install --no-cache-dir -Iv notebook ipython ipykernel psutil==5.9.2 tensorflow pandas scikit-learn tensorflow-addons imblearn numpy matplotlib tqdm 

WORKDIR /workspace 