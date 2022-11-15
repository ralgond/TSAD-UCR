# TSAD-UCR
Time Series Anomaly Detection Solutions for KDD2021 UCR dataset

## The rule

https://compete.hexagon-ml.com/practice/competition/39/

## How to run
1. Download ucr data from [here](https://github.com/ralgond/KDD2021-UCR), and create a directory named ".data" in this repo directory,
unzip the downloaded data to ".data".
2. Enter a directory, such as Augment, install package according to "requirements.txt", and execute "python main.py".

## 得分
|solution name|score|directory|resources|
|-------------|-----|---------|---------|
| genta       | 83.6|genta    |[Video](https://www.youtube.com/watch?v=J_Ebbql9jCo)|
| Augment (3way) | 73.2 | Augment | 参考[Paper](https://arxiv.org/pdf/1812.04606.pdf)|
| DeepSAD  | 64.8 | DeepSAD | 参考[Code](https://github.com/lukasruff/Deep-SAD-PyTorch)|
| DeepSVDD  | 54.4 | DeepSVDD | 参考[Code](https://github.com/lukasruff/Deep-SVDD-PyTorch)|
| matrix-profile-1| 52.8| matrixprofile-1| 参考[Page](https://www.cs.ucr.edu/~eamonn/MatrixProfile.html) |
| LOF | 42.8 | LOF | 参考[Doc](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html) |
| vanilla_ae  | 29.2| ae-torch/ae_conv.py ||
| vanilla_vae | 27.6| ae-tf/vae.py | 参考[Doc](https://keras.io/examples/generative/vae/)|
| Isolation Forest | 23.6 | IF| 参考[Doc](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) |
