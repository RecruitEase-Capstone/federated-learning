## Client
1. Clone repository using :
```sh
git clone https://github.com/RecruitEase-Capstone/federated-learning.git
```
2. Download data.zip [Here!](https:www.youtube.com) and extract on `federated-learning` directory
3. Create a directory `models` that contains directories  `local` & `global` to create it you can use :
* Linux
```sh
make dir-models
```
* Windows
```sh
mkdir models/local ; mkdir models/global
```
4. The code on the `Client` side will only work if the code on the `Server` side has been running, to run the code on the `client` side you can use :
```sh
python client.py
```

## Server
1. Clone repository using :
```sh
git clone https://github.com/RecruitEase-Capstone/federated-learning.git
```
2. Make sure `num_clients` in the `server.py` same as the number of client that will train the model
3. To run the code on the `Server` side you can use :
```sh
python client.py
```