# FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
FROM pure/python:3.7-cuda10.2-cudnn7-runtime

WORKDIR /home/user
COPY requirements.txt /home/user/
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

## config
ARG USER=wangsadirdja
ARG UID=1174

RUN adduser stud${USER} --uid ${UID} --home /home/stud/${USER}/ --disabled-password --gecos "" --no-create-home
RUN mkdir -p /home/stud/${USER}
RUN chown -R stud${USER} /home/stud/${USER}

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# NAME=ls6-stud-registry.informatik.uni-wuerzburg.de/studwangsadirdja-project-nlp:0.0.1

