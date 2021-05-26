# Details of the base image are here: https://hub.docker.com/r/tensorflow/tensorflow/tags
# It runs Python 3.6

FROM tensorflow/tensorflow:nightly-jupyter

RUN apt-get update && apt-get install -y git
RUN /usr/bin/python3 -m pip install --upgrade pip

RUN mkdir -p /tf
WORKDIR /tf
ENV PYTHONPATH "${PYTHONPATH}:/tf"

COPY ./requirements.txt ./
RUN pip install -r requirements.txt
