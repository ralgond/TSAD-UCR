import os
import sys
import pyscamp as mp
import numpy as np
import matplotlib.pyplot as plt
import numpy as  np
from utils_profile import find_first_second_pos, do_moving_average
import rolling
from scipy.signal import find_peaks

def p2p_profile(test_data, win_size):
    max_l = list(rolling.Max(test_data, win_size))
    min_l = list(rolling.Min(test_data, win_size))
    p2p_l = np.array(max_l) - np.array(min_l)
    return p2p_l.tolist()

def inv_p2p_profile(test_data, win_size):
    max_l = list(rolling.Max(test_data, win_size))
    min_l = list(rolling.Min(test_data, win_size))
    p2p_l = (np.array(max_l) - np.array(min_l)).tolist()
    for i in range(len(p2p_l)):
        if p2p_l[i] == 0 or p2p_l[i] == 0.:
            pass
        else:
            p2p_l[i] = 1./p2p_l[i]
    return p2p_l

def peek_count_profile(test_data, win_size):
    l = [0 for _ in range(len(test_data))]
    peeks, _ = find_peaks(test_data, width=5)
    for idx in peeks:
        l[idx] = 1
    return list(rolling.Sum(l, win_size))

def entropy_profile(test_data, win_size):
    return list(rolling.Entropy(test_data, win_size))

def inv_std_profile(test_data, win_size):
    l = list(rolling.Std(test_data, win_size))
    for i in range(len(l)):
        if l[i] > 0:
            l[i] = 1.0/l[i]
    return l

def std_profile(test_data, win_size):
    return list(rolling.Std(test_data, win_size))

def main(all_data, split_pos):
    # win_size_list = [16, 32, 64, 96, 128, 160, 192, 224, 256, 512, 768]
    # ma_win_size_list=[50, 70, 100, 200, 250, 300, 350, 400, 450, 500, 550]
    win_size_list = [8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 512, 768]
    ma_win_size_list = [i for i in win_size_list]

    train_data, test_data = all_data[:split_pos], all_data[split_pos:]

    test_data_diff = []
    for i in range(1, len(test_data)):
        test_data_diff.append(test_data[i] - test_data[i-1])

    test_data_acc = []
    for i in range(1, len(test_data_diff)):
        test_data_acc.append(test_data_diff[i] - test_data_diff[i-1])

    profile_list = []
    def __padding_zero(_p, win_size, ma_win_size, padding_zero_length):
        profile = [0 for _ in range(padding_zero_length)]
        profile.extend(_p)
        profile_list.append((profile, win_size, ma_win_size))

    for win_size, ma_win_size in zip(win_size_list, ma_win_size_list):
        _profile, _ = mp.abjoin(test_data, train_data, win_size)
        __padding_zero(_profile, win_size, ma_win_size, len(train_data))

        _profile, _ = mp.selfjoin(test_data, win_size)
        __padding_zero(_profile, win_size, ma_win_size, len(train_data))

        _profile, _ = mp.selfjoin(test_data_diff, win_size)
        __padding_zero(_profile, win_size, ma_win_size, len(train_data))

        # _profile, _ = mp.selfjoin(test_data_acc, win_size)
        # __padding_zero(_profile, win_size, ma_win_size, len(train_data))

        _profile = p2p_profile(test_data, win_size)
        __padding_zero(_profile, win_size, ma_win_size, len(train_data))

        _profile = p2p_profile(test_data_diff, win_size)
        __padding_zero(_profile, win_size, ma_win_size, len(train_data))

        # _profile = p2p_profile(test_data_acc, win_size)
        # __padding_zero(_profile, win_size, ma_win_size, len(train_data))

        # _profile = peek_count_profile(test_data, win_size)
        # __padding_zero(_profile, win_size, ma_win_size, len(train_data))

        # _profile = peek_count_profile(test_data_diff, win_size)
        # __padding_zero(_profile, win_size, ma_win_size, len(train_data))

        # _profile = entropy_profile(test_data, win_size)
        # __padding_zero(_profile, win_size, ma_win_size, len(train_data))

        # _profile = inv_std_profile(test_data, win_size)
        # __padding_zero(_profile, win_size, ma_win_size, len(train_data))

        _profile = inv_std_profile(test_data_acc, win_size)
        __padding_zero(_profile, win_size, ma_win_size, len(train_data))

        # _profile = inv_p2p_profile(test_data, win_size)
        # __padding_zero(_profile, win_size, ma_win_size, len(train_data))

        _profile = std_profile(test_data_acc, win_size)
        __padding_zero(_profile, win_size, ma_win_size, len(train_data))

    first_div_second_score_list = []

    for profile, win_size, ma_win_size in profile_list:
        ma_profile = do_moving_average(profile, ma_win_size)
        (idx1,score1), (idx2, score2) = find_first_second_pos(ma_profile, win_size)
        if score2 == 0 or score2 == 0.:
            first_div_second_score_list.append((idx1, 0.))
        else:
            first_div_second_score_list.append((idx1, score1*1.0/score2))

    max_first_div_second_score = -1
    max_first_div_second_idx = -1
    for pos, ratio in first_div_second_score_list:
        if ratio > max_first_div_second_score:
            max_first_div_second_score = ratio
            max_first_div_second_idx = pos

    return max_first_div_second_idx
