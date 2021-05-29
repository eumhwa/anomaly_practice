## anomaly detection examples

1. Memory augmented AutoEncoder
    - official code: 
    https://github.com/donggong1/memae-anomaly-detection

    - 비정상 입력에 대해서도 정상에 가까운 출력을 만들어 anomaly score를 높여주기 위해 memory module을 사용한 autoencoder

2. Multi-scale Features based MemAE
    - resnet의 각 block에서 추출한 계층적 feature를 MemAE 인풋으로 사용

3. Oneclass-SVM with BYOL(self-supervised learning)
    - BYOL을 이용하여 CNN(resnet)을 pretraining시킨 후, 
    downstream task로 anomaly detection을 수행
