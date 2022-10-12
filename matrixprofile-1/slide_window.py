def create_window_list(data, win_size):
    ret = []

    for i in range(len(data)-win_size+1):
        term = data[i:i+win_size]
        ret.append(term)

    return ret