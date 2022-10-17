import pyscamp as mp
from ucr_dataset import get_series
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import rolling

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

def main(file_no):
    all_data, split_pos, anomaly_range = get_series(file_no)
    
    all_data = aggregate(all_data, win_size=5)

    train_data, test_data = all_data[:split_pos], all_data[split_pos:]

    profile_list = []
    for win_size in [8,16,32,64,96,128,256,320,512]:
        # profile, _ = mp.abjoin(test_data, train_data, win_size)
        # scaled_profile = minmax_scale(profile)
        # scaled_padding_profile = tail_padding_zero(scaled_profile, len(test_data))
        # profile_list.append(scaled_padding_profile)

        profile, _ = mp.selfjoin(test_data, win_size)
        scaled_profile = minmax_scale(profile)
        scaled_padding_profile = tail_padding_zero(scaled_profile, len(test_data))
        profile_list.append(scaled_padding_profile)
        
    sum = np.array(profile_list[0])
    for l in profile_list[1:]:
        sum  = sum + np.array(l)

    scores = sum / len(profile_list)
    
    correct_range = (anomaly_range[0]-100, anomaly_range[1]+100)
    pos = np.argmax(scores) + len(train_data)
    if pos >= correct_range[0] and pos <= correct_range[1]:
        return 1
    else:
        return -1

if __name__ == "__main__":
    correct_cnt = 0
    error_cnt = 0

    for i in range(1,251):
        ret = main(i)

        if (ret > 0):
            correct_cnt += 1
            print (f"({i}) correct, ==========>correct_cnt:{correct_cnt}, error_cnt:{error_cnt}")
        else:
            error_cnt += 1
            print (f"({i}) error, ============>correct_cnt:{correct_cnt}, error_cnt:{error_cnt}")