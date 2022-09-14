# DFP-Numeric-2

This is a repo for DS project on Bitcoin Prediction. 
Project was created in relation to product DS bootcamp that I was undergoing. 
Project was accepted by the bootcamp leader and was left unfinished on the stage of 90% ready.
This was a team of 2 people project, yet 99% of commits are mine =)

Core characteristics:
--> Docker usage
--> Python Classes usage
--> Jupyter Notebook as a modeling environment

### **Project name**: Bitcoin Price Prediction

**GitHub**: https://github.com/Serg-Protsenko/DFP-Numeric-2

### **Docker**: <br>
<u>Creating the image</u> -- from the /install folder:<br>
$ docker build . -t btc_price_prediction <br>
<u>Running the image</u> -- from the project folder:<br> 
$ docker run -v $(pwd):/tf -it --rm -p 8888:8888 btc_price_prediction

##### **Docker file**:
```commandline
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
```

### **Data sources**: <br>
* Bitcoin Exchange Rates Statistics https://www.investing.com/crypto/bitcoin/historical-data
* Bitcoin Average Block Size https://www.quandl.com/data/BCHAIN/AVBLS-Bitcoin-Average-Block-Size
* Bitcoin Hash Rate https://www.quandl.com/data/BCHAIN/HRATE-Bitcoin-Hash-Rate 
* Bitcoin Miners Revenue https://www.quandl.com/data/BCHAIN/MIREV-Bitcoin-Miners-Revenue 
* Bitcoin My Wallet Number of Transaction Per Day https://www.quandl.com/data/BCHAIN/MWNTD-Bitcoin-My-Wallet-Number-of-Transaction-Per-Day
* Bitcoin Cost Per Transaction https://www.quandl.com/data/BCHAIN/CPTRA-Bitcoin-Cost-Per-Transaction
* Bitcoin USD Exchange Trade Volume https://www.quandl.com/data/BCHAIN/TRVOU-Bitcoin-USD-Exchange-Trade-Volume
* Bitcoin My Wallet Transaction Volume https://www.quandl.com/data/BCHAIN/MWTRV-Bitcoin-My-Wallet-Transaction-Volume
* Bitcoin My Wallet Number of Users https://www.quandl.com/data/BCHAIN/MWNUS-Bitcoin-My-Wallet-Number-of-Users
