import numpy as np

def ndarray_pad_zero(nda, size):
    l = []
    if isinstance(nda, np.ndarray):
        l = nda.tolist()
    if len(l) >= size:
        return l
    else:
        while len(l) < size:
            l.append(0)
    return l

def min_max_scale(l):
    min_v = l.min()
    max_v = l.max()
    return (l - min_v) / (max_v - min_v)

def mean_min_max_scale(l):
    min_v = l.min()
    max_v = l.max()
    mean_v = l.mean()
    return (l - mean_v) / (max_v - min_v)

def standard_scale(l):
    mean_v = l.mean()
    std_v = l.std()
    return (l-mean_v) / std_v

def diff(l):
    ret = []
    for i in range(1,len(l)):
        ret.append(l[i] - l[i-1])
    return ret

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def exponential_smoothing(alpha, s):
    s2 = np.zeros(s.shape)
    s2[0] = s[0]
    for i in range(1, len(s2)):
        s2[i] = alpha*s[i]+(1-alpha)*s2[i-1]
    return s2

def has_intersection(r1, r2):
    '''
    判断两个区间是否有交集
    '''
    if max(r1[0], r2[0]) <= min(r1[1], r2[1]):
        return True
    else:
        return False