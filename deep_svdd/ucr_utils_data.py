import random
import numpy as np
import tensorflow as tf

class Dataset:
    '''Time series dataset.'''
    def __init__(self, data, win_size) -> None:
        if win_size > len(data):
            raise ValueError(f"win_size {win_size} > data length {len(data)}.")

        self.data = data
        self.win_size = win_size
        self.len = len(self.data) - self.win_size + 1 

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.__len__():
            raise IndexError(f"Output [0,{self.__len__()}]")     
        return self.data[idx:idx+self.win_size]

class XTrainDataLoader:
    def __init__(self, x_train, batch_size:int=64, seed:int=0) -> None:
        self.x_train = x_train
        self.batch_size = batch_size
        self.finished = False
        self.idx = 0
        self.seed = seed

    def __iter__(self):
        return XTrainDataLoader(self.x_train, batch_size=self.batch_size, seed=self.seed)

    def __next(self):
        if self.finished:
            raise StopIteration

        ret = []
        for i in range(self.idx, min(self.idx + self.batch_size, len(self.x_train))):
            ret.append(self.x_train[i])

        self.idx += len(ret)
        
        if len(ret) == 0:
            self.finished = True
            raise StopIteration

        if len(ret) < self.batch_size:
            self.finished = True
            return np.array(ret)
        
        return np.array(ret)

    def __next__(self):
        ret = self.__next()
        ret1 = tf.constant(ret)
        return tf.reshape(ret1, (ret1.shape[0], ret1.shape[1], 1))

    def shuffle(self):
        self.seed += 1
        #print ("==============>self.seed:", self.seed)
        random.Random(self.seed).shuffle(self.x_train)

class DataLoader:
    '''Time series dataloader.'''
    def __init__(self, ds, batch_size:int=64) -> None:
        if ds is None:
            raise ValueError("ds should not be None")

        if batch_size <= 0:
            raise ValueError("batch_size should > 0")

        self.ds = ds
        self.batch_size = batch_size
        self.idx = 0
        self.finished = False

    def __iter__(self):
        return DataLoader(self.ds, batch_size=self.batch_size)

    def __next(self):
        batch = []

        if self.finished:
            raise StopIteration

        j = 0
        while j < self.batch_size and self.idx < len(self.ds):
            batch.append(self.ds[self.idx])
            self.idx += 1
            j += 1

        if len(batch) == 0:
            raise StopIteration

        if len(batch) <= self.batch_size:
            if self.idx >= len(self.ds):
                self.finished = True
            return np.array(batch)

        raise NotImplementedError()

    def __next__(self):
        ret1 = tf.constant(self.__next())
        return tf.reshape(ret1, (ret1.shape[0], ret1.shape[1], 1))
    

def shuffle_x_train(x_train):
    idxList = [i for i in range(len(x_train))]
    random.Random(42).shuffle(idxList)
    
    ret = []
    for idx in idxList:
        ret.append(x_train[idx])
    return ret
    
if __name__ == "__main__":
    x_train = []
    x_train.append((1,2,3))
    x_train.append((2,3,4))
    x_train.append((3,4,5))
    x_train.append((4,5,6))
    x_train.append((5,6,7))
    x_train.append((6,7,8))
    x_train.append((7,8,9))
    x_train.append((8,9,10))

    # loader = DataLoader(x_train, batch_size=6)
    # for data in loader:
    #     print (data)

    loader = XTrainDataLoader(x_train, 8)
    for data in loader:
        print (data)

