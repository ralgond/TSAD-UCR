
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from slide_window import create_window_list


def analyze_filename(fn):
    x = fn.split('.')[0].split('_')
    split_pos = int(x[-3])
    anomaly_range = (int(x[-2]), int(x[-1]))

    return split_pos, anomaly_range

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

    split_pos, anomaly_range = analyze_filename(file_name)
    
    return l, split_pos, anomaly_range

def load_data(num, win_size):
    all_data, split_pos, anomaly_range = get_series(num)

    train_data, test_data = np.array(all_data[:split_pos]), np.array(all_data[split_pos:])

    scaler = MinMaxScaler()
    scaler.fit(train_data.reshape(-1,1))

    train_data = scaler.transform(train_data.reshape(-1,1)).squeeze(1)
    test_data = scaler.transform(test_data.reshape(-1,1)).squeeze(1)

    X_train_tmp = np.array(create_window_list(train_data, win_size))
    X_train = [np.expand_dims(train, 1) for train in X_train_tmp]

    # Predict test, found the anomalies
    X_test_tmp = np.array(create_window_list(test_data, win_size))
    X_test = [np.expand_dims(test, 1) for test in X_test_tmp]

    return np.array(X_train), np.array(X_test), all_data, split_pos, anomaly_range
