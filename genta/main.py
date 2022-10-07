import pyscamp as mp
import numpy as np
from dataset_ucr import get_series
from utils import ndarray_pad_zero, standard_scale
#from selfjoin_vanilla import main
#from selfjoin_downsample import main
from first_div_second import main
import time

if __name__ == "__main__":
    error_file = open("error_pos.txt", "w+")
    correct_count = 0
    error_count = 0
    for i in range(1, 250+1): 
    #for i in [36,38,46,47,61,63,73,79,80,81,82,88,105,106,107,108,146,155,190,196,203,206,210,212,214,222,225,228,229,230,233,234,243,244,247,248,249,250]:

        start_time = time.time()
        if i in [239,240,241]:
            error_count += 1
            continue
        print("\n")
        all_data, split_pos, anormal_range = get_series(i)
        pred_pos = main(all_data, split_pos)

        correct_range = (anormal_range[0]-100, anormal_range[1]+100)
        print(correct_range, pred_pos)

        ret = 1
        if pred_pos >= correct_range[0] and pred_pos <= correct_range[1]:
            ret = 1
        else:
            ret = -1
        if ret > 0: 
            correct_count += 1
        else: 
            error_count += 1
            error_file.write(f"{i}\n")
            error_file.flush()

        end_time = time.time()
        print (f"({i}) correct:", ret > 0, "================>correct_count:",correct_count, " error_count:", error_count, " time: %.1fs"%(end_time-start_time))
        
    error_file.close()