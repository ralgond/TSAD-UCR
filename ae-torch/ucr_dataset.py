import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def analyze_filename(fn):
    x = fn.split('.')[0].split('_')
    train_test_split_pos = int(x[-3])
    abnormal_range = (int(x[-2]), int(x[-1]))

    return train_test_split_pos, abnormal_range

def get_series(num):
    l = []
    num_str = '%03d' % num
    file_name = ""
    for filename in os.listdir("../.data/ucr/"):
        x = filename.split('_')
        if x[0] == num_str:
            file_name = filename
            for line in open(os.path.join("../.data/ucr/", filename)):
                l.append(float(line.strip()))

    train_test_split_pos, abnormal_range = analyze_filename(file_name)
    
    return l, train_test_split_pos, abnormal_range

def get_data(num):
    all_data, split_pos, anomaly_range = get_series(num)
    train_data, test_data = np.array(all_data[:split_pos]), np.array(all_data[split_pos:])

    scaler = MinMaxScaler((0, 1))
    scaler.fit(train_data.reshape(-1, 1))

    train_data = scaler.transform(train_data.reshape(-1, 1))
    test_data = scaler.transform(test_data.reshape(-1, 1))

    return train_data.squeeze(1), test_data.squeeze(1)
