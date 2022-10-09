import os
from re import A
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"

from tensorflow.keras.layers import Input, Dense, Softmax, Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from ucr_dataset import get_series

def gen_up_stab(series):
    ret = [] #a list contain augmentated series
    series = np.array(series)
    max_value = series.max()
    stab_value = 2*max_value
    for i in range(len(series)):
        series[i] = stab_value
        ret.append(series.copy())
    return np.array(ret)

def gen_model(win_size):
    input = Input(shape=(win_size,))
    x = Dense(250, use_bias=False)(input)
    x = Dense(2500, use_bias=False)(x)
    x = Softmax()(x) #2500个类
    return Model(inputs=input, outputs=x)

def gen_test():
    x = [1 for _ in range(256)]
    x [31] = 3
    return x

def main():
    all_data, train_test_split_pos, abnormal_range = get_series(1)
    train_series, test_series = all_data[:train_test_split_pos], all_data[train_test_split_pos:]
    aug_data = gen_up_stab(train_series[:128])
    model = gen_model(128)
    model.summary()
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print ("============>aug_data.len:",len(aug_data))
    model.fit(aug_data, np.array([1 for _ in range(len(aug_data))]))
    
    test = gen_test() #长度为256
    for i in range(len(test)-128+1):
        test_seg = tf.constant(test[i:i+128].copy())
        print ("=============>predict:", i, model(test_seg, training=False))

if __name__ == "__main__":
    main()
