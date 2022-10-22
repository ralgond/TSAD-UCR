# TSAD-UCR
Time Series Anomaly Detection Solutions for KDD2021 UCR dataset

## How to run
Download ucr data from [here](https://github.com/ralgond/KDD2021-UCR), and create a directory named ".data" in this repo directory,
unzip the downloaded data to ".data".

## 得分
|solution name|score|directory|resources|
|-------------|-----|---------|---------|
| genta       | 83.6|genta    |[Video](https://www.youtube.com/watch?v=J_Ebbql9jCo)|
| Augment | 66.4 | Augment | 参考[Paper](https://arxiv.org/pdf/1812.04606.pdf)|
| matrix-profile-1| 52.8| matrixprofile-1| |
| deep svdd (tf) | 44.4 | deep_svdd-tf | 参考[Code](https://github.com/lukasruff/Deep-SVDD-PyTorch)|
| LOF | 42.8 | LOF | 参考[Doc](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html) |
| deep svdd (torch)  | 32.8| deep_svdd | 参考[Code](https://github.com/lukasruff/Deep-SVDD-PyTorch)|
| vanilla_ae  | 29.2| ae-torch/ae_conv.py ||
| vanilla_vae | 27.6| ae-tf/vae.py | 参考[Doc](https://keras.io/examples/generative/vae/)|
| Isolation Forest | 23.6 | IF| 参考[Doc](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) |
