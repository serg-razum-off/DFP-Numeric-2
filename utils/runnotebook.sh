#!/bin/bash
# Runs a Jupyter notebook server for a prebuilt docker image named 'myproject'
# see script builddocker.sh for how to build the image
# it also maps the current directory to a folder within the docker image

GPU=$(sudo lshw -numeric -C display)

if [[ $GPU == *"NVIDIA"* ]]; then
    echo "Trying to run with GPU (make sure you have nvidia-docker installed)"
    sudo docker run --gpus all -u $(id -u):$(id -g) -v $(pwd):/tf -it --rm -p 8888:8888 myproject
else
    echo "Running with CPU only"
    sudo docker run -u $(id -u):$(id -g) -v $(pwd):/tf -it --rm -p 8888:8888 myproject
fi

# docker run -v $(pwd):/tf -it --rm -p 8888:8888 btc_price_prediction ##SR: doesn't work in any of the cases. Some bash err. For now [2021/06/02] I'm leaving it as it is
