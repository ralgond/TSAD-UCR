import numpy as np

def mse(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.square(np.subtract(actual,pred)).mean() 

def moving_avg(s, win_size):
    ret = [float('nan') for _ in range(0, win_size)]
    ret2 = np.convolve(s, np.ones(win_size)/(win_size), mode='valid').tolist()
    ret.extend(ret2)
    return ret
    

