import pandas as pd
import numpy as np
from ucr_dataset import get_series

train_series, test_series, abnormal_range = get_series(1)

all_data = np.append(train_series, test_series)

abnormal_list = [0 for i in all_data]
for i in range(abnormal_range[0], abnormal_range[1]+1):
    abnormal_list[i] = 1

df = pd.DataFrame({'signal':all_data, 'anomaly':abnormal_list})

df.to_csv("001_UCR.csv", index=False, sep=',')