# MM106: StainedGlass

standing: 7th / 52 (provisional: 8th)  
score: 871,080.00 (provisional: 865,600.00)  
rating: 1814 -> 1906 (+92, highest)

## 概要

* ステンドグラスを作りなさ～い

## やったこと

* ハニカム状に母点を配置して波面法っぽい何かを使う

![wavefront](vis/wavefront.gif)

![wavefront2](vis/wavefront2.gif)

* 母点の近傍移動による焼きなまし (30 itr に一回差分最小のセルを削除して差分最大のセルの周辺に挿入)

![annealing](vis/annealing.gif)

* kmeans++ で色空間をクラスタリングして減色 (2 ~ 70 色で全探索)

![kmeans](vis/kmeans.gif)

* 偏差絶対値和は中央値で最小化される

## 結果

seed 1:  
![seed1](vis/1.png)

seed 2:  
![seed2](vis/2.png)

seed 3:  
![seed3](vis/3.png)

seed 4:  
![seed4](vis/4.png)

seed 5:  
![seed5](vis/5.png)

seed 6:  
![seed6](vis/6.png)

seed 7:  
![seed7](vis/7.png)

seed 8:  
![seed8](vis/8.png)

seed 9:  
![seed9](vis/9.png)

seed 10:  
![seed10](vis/10.png)