
from sklearn.preprocessing import MinMaxScaler,

import numpy as np
from ucr_dataset import get_series



def main(file_no:int):
    all_data, split_pos, anomaly_range = get_series(file_no)
    train_data, test_data = np.array(all_data[:split_pos]), np.array(all_data[split_pos:])

    scaler = MinMaxScaler()
    scaler.fit(train_data.reshape(-1,1))

    train_data = scaler.transform(train_data.reshape(-1,1)).squeeze(1)
    test_data = scaler.transform(test_data.reshape(-1,1)).squeeze(1)

    correct_range = (anomaly_range[0]-100, anomaly_range[1]+100)
    pos = None
    if pos >= correct_range[0] and pos <= correct_range[1]:
        return 1
    else:
        return -1


if __name__ == "__main__":
    correct_count = 0
    error_count = 0
    for i in range(1,251):
        ret = main(i)
        status = None
        if ret > 0:
            correct_count += 1
            status = "correct"
        else:
            error_count += 1
            status = "error"
        print (f"({i}) {status}=========> correct:{correct_count}, error:{error_count}")