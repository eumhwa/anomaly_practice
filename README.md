# anomaly detection examples

## Memory augmented AutoEncoder
비정상 입력에 대해서도 정상에 가까운 출력을 만들어 anomaly score를 높여주기 위해 memory module을 사용한 autoencoder
encoder를 거쳐 계산된 code가 정상패턴이 기록된 memory module과 attention 연산을 하여 새로운 code를 계산하고 이를 decoding 하는 방법

## Citation

```
@inproceedings{gong2019memorizing,
  title={Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection},
  author={Gong, Dong and Liu, Lingqiao and Le, Vuong and Saha, Budhaditya and Mansour, Moussa Reda and Venkatesh, Svetha and Hengel, Anton van den},
  booktitle={IEEE International Conference on Computer Vision (ICCV)},
  year={2019}
}
```
