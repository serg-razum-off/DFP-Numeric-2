# DFP-Numeric-2

###**Project name**: Bitcoin Price Prediction

**GitHub**: https://github.com/Serg-Protsenko/DFP-Numeric-2

###**Docker**: <br>
<u>Creating the image</u> -- from the /install folder:<br>
$ docker build . -t btc_price_prediction <br>
<u>Running the image</u> -- from the project folder:<br> 
$ docker run -v $(pwd):/tf -it --rm -p 8888:8888 btc_price_prediction

#####**Docker file**:
```commandline
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
```

###**Data sources**: <br>
* Bitcoin Exchange Rates Statistics https://www.investing.com/crypto/bitcoin/historical-data <br>
* Bitcoin Average Block Size https://www.quandl.com/data/BCHAIN/AVBLS-Bitcoin-Average-Block-Size <br>
* Bitcoin Hash Rate https://www.quandl.com/data/BCHAIN/HRATE-Bitcoin-Hash-Rate <br>
* Bitcoin Miners Revenue https://www.quandl.com/data/BCHAIN/MIREV-Bitcoin-Miners-Revenue <br>
* Bitcoin My Wallet Number of Transaction Per Day https://www.quandl.com/data/BCHAIN/MWNTD-Bitcoin-My-Wallet-Number-of-Transaction-Per-Day <br>
* Bitcoin Cost Per Transaction https://www.quandl.com/data/BCHAIN/CPTRA-Bitcoin-Cost-Per-Transaction
* Bitcoin USD Exchange Trade Volume https://www.quandl.com/data/BCHAIN/TRVOU-Bitcoin-USD-Exchange-Trade-Volume
* Bitcoin My Wallet Transaction Volume https://www.quandl.com/data/BCHAIN/MWTRV-Bitcoin-My-Wallet-Transaction-Volume

