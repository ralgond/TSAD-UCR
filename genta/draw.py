import pyscamp as mp
import numpy as np
import matplotlib.pyplot as plt
from dataset_ucr import get_series
from utils import ndarray_pad_zero, standard_scale, moving_average, has_intersection
from utils_profile import find_first_second_pos
from scipy.signal import find_peaks
import sys

def draw1():
    file_no = int(sys.argv[1])
    all_data, split_pos, anormal_range = get_series(file_no)

    win_size_list = [16, 32, 64, 96, 128, 160, 192, 224, 256, 512, 768]

    profile_list = []
    for win_size in win_size_list:
        profile, _ = mp.selfjoin(all_data, win_size)
        profile_padded = ndarray_pad_zero(profile, len(all_data))
        
        # 让profile变平滑, 抖动越大，平滑窗口越大（直觉？）
        ma_profile = []
        for _ in range(win_size):
            ma_profile.append(0)
        _ma_profile = moving_average(profile_padded, win_size) 
        ma_profile.extend(_ma_profile)
        profile_list.append(ma_profile)


    plt.figure(figsize=(16,9))

    plt.subplot(len(win_size_list)+1, 1, 1)
    plt.plot([i for i in range(len(all_data))], all_data)
    

    for idx, profile in enumerate(profile_list, start=1):
        plt.subplot(len(win_size_list)+1, 1, idx+1)
        plt.plot([i for i in range(len(profile))], profile)

        (idx1,score1), (idx2, score2) = find_first_second_pos(profile, win_size)
        # idx1,_,_ = first_pos
        plt.axvline(idx1, 0, 1, color='red')

        # idx2,_,_ = second_pos
        plt.axvline(idx2, 0, 1, color='black')

    plt.show()
    
def draw_peek():
    all_data, split_pos, anormal_range = get_series(24)
    train_data = all_data[:split_pos]

    peeks, _  = find_peaks(train_data, width=20)

    plt.figure(figsize=(16,9))
    plt.plot([i for i in range(len(train_data))], train_data)

    for peek in peeks:
        plt.axvline(peek, color="red")

    plt.show()

if __name__ == "__main__":
    draw1()