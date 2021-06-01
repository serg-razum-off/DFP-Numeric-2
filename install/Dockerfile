# Details of the base image are here: https://hub.docker.com/r/tensorflow/tensorflow/tags
# It runs Python 3.6

FROM tensorflow/tensorflow:nightly-jupyter

RUN apt-get update && apt-get install -y git
# SR as per https://askubuntu.com/questions/94102/what-is-the-difference-between-apt-get-update-and-upgrade
RUN apt-get -y upgrade
# SR [2021/06/01]: enabling autocomplete in Jupyter
RUN pip install jedi==0.17.2 

RUN /usr/bin/python3 -m pip install --upgrade pip

RUN mkdir -p /tf
WORKDIR /tf
ENV PYTHONPATH "${PYTHONPATH}:/tf"

COPY ./requirements.txt ./
RUN pip install -r requirements.txt