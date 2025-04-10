1. Clone repository using :
```sh
git clone https://github.com/RecruitEase-Capstone/federated-learning.git
```
## Client
2. Download data.zip [Here!](https://pages.github.com/) and extract on your computer

3. Create a directory `models` that contains directories  `local` & `global` to create it you can use :
* Linux
```sh
make dir-models
```
* Windows
```sh
mkdir models/local ; mkdir models/global
```
4. The code on the `Client` side will only work if the code on the `Server` side has been running, to run the code on the client side you can use :
```sh
python client.py
```

## Server