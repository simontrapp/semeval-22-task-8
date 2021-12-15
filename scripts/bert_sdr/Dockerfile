FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# config
ARG USER=trapp
ARG UID=1309

RUN mkdir -p /home/stud/${USER}/scripts/bert_sdr
RUN mkdir -p /home/stud/${USER}/models
RUN adduser stud${USER} --uid ${UID} --home /home/stud/${USER}/ --disabled-password --gecos "" --no-create-home

# Build from project root!
WORKDIR /home/stud/${USER}
COPY requirements.txt /home/stud/${USER}/
RUN apt update && apt install -y python3-pip
RUN pip3 install --upgrade pip
# install correct pytorch version for the cluster gpus
RUN pip3 install --no-cache-dir torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip3 install --no-cache-dir -r requirements.txt

# add the files necessary
ADD data /home/stud/${USER}/data
COPY scripts/bert_sdr/util.py /home/stud/${USER}/scripts/bert_sdr/
COPY scripts/bert_sdr/calculate_article_similarity.py /home/stud/${USER}/scripts/bert_sdr/
COPY scripts/bert_sdr/train_classifier.py /home/stud/${USER}/scripts/bert_sdr/

RUN chown -R stud${USER} /home/stud/${USER}

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
