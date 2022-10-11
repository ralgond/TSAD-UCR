import numpy as np

def get_series_with_abnormal_1():
    train_ret = []

    ret = []
    for cnt in range(30):
        l = [i for i in range(64)]
        ret.append(l)
    train_ret = np.concatenate(ret)

    print (len(train_ret))

    ret = []
    for cnt in range(30):
        l = [i for i in range(64)]
        ret.append(l)
    test_ret = np.concatenate(ret)

    for i in range(15*64, 17*64):
        test_ret[i] = 0

    return train_ret, test_ret, (len(train_ret)+15*64, len(train_ret)+17*64)


