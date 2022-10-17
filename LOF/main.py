from os import stat
import pyscamp as mp
from ucr_dataset import get_series
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import rolling
from sklearn.neighbors import LocalOutlierFactor
import time

def create_window_list(data, win_size):
    ret = []

    for i in range(len(data)-win_size+1):
        term = data[i:i+win_size]
        ret.append(term)

    return ret

def minmax_scale(l):
    scaler = MinMaxScaler()
    return scaler.fit_transform(l.reshape(-1, 1)).reshape(1, -1).squeeze(0)

def tail_padding_zero(l, length):
    ret = []
    ret.extend(l)
    for i in range(length - len(l)):
        ret.append(0)
    return ret

def aggregate(ts, win_size=10):
    return list(rolling.Mean(ts, win_size))

def aggregate2(ts, win_size=3):
    '''
    when win_size is 3, 
    [1,2,3,4,5,6,7,8,9,10] to
    [2.0, 2.0, 2.0, 5.0, 5.0, 5.0, 8.0, 8.0, 8.0, 10.0]
    '''
    ret = []
    start = 0 
    while start < len(ts):
        end = min(start+win_size, len(ts))
        slice = ts[start:end]
        slice_mean_value = np.mean(slice)

        for _ in range(end-start):
            ret.append(slice_mean_value)
            
        start += win_size
    return ret

def main(file_no):
    all_data, split_pos, anomaly_range = get_series(file_no)

    all_data = aggregate(all_data, win_size=5)

    train_data, test_data = all_data[:split_pos], all_data[split_pos:]

    #for win_size in [50, 100, 150, 200]:
    train_window_list = create_window_list(train_data, 128)
    start_time = time.time()
    clf = LocalOutlierFactor(novelty=True, n_jobs=5).fit(train_window_list)
    print("fit_time:", (time.time() - start_time))

    test_window_list = create_window_list(test_data, 128)
    scores = clf.score_samples(test_window_list)
    
    correct_range = (anomaly_range[0]-100, anomaly_range[1]+100)
    pos = np.argmin(scores) + len(train_data)
    if pos >= correct_range[0] and pos <= correct_range[1]:
        return 1
    else:
        return -1

if __name__ == "__main__":
    correct_cnt = 0
    error_cnt = 0

    ret = None
    for i in range(1,251):
        if i in [239,240,241]:
            ret = -1
        else:
            ret = main(i)

        if (ret > 0):
            correct_cnt += 1
            print (f"({i}) correct, ==========>correct_cnt:{correct_cnt}, error_cnt:{error_cnt}")
        else:
            error_cnt += 1
            print (f"({i}) error, ============>correct_cnt:{correct_cnt}, error_cnt:{error_cnt}")