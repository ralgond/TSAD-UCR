import numpy as np
from utils import moving_average, ndarray_pad_zero
import sys
import os



def find_first_second_pos(_profile, win_size):
    p = _profile.copy()

    idx1 = np.argmax(p)
    score1 = p[idx1]

    profile_len = len(p)
    i = max(0, idx1-win_size)
    while i < min(profile_len, idx1+win_size):
        p[i] = 0
        i += 1

    idx2 = np.argmax(p)
    score2 = p[idx2]

    return (idx1,score1), (idx2, score2)

def do_moving_average(profile, w):
    ret = [0 for _ in range(w)]
    ret.extend(moving_average(profile, w))
    return ret

def profile_list_mean(profile_list, padding_length):
    mean_profile = ndarray_pad_zero(np.array(profile_list[0][0]), padding_length)
    for profile,_,_ in profile_list[1:]:
        profile_padded = ndarray_pad_zero(profile, padding_length)
        mean_profile = mean_profile + np.array(profile_padded)
    mean_profile = mean_profile / len(profile_list)

    return mean_profile


# sys.path.append(os.path.abspath(os.path.join(os.path.dirname('__file__'), os.path.pardir)))
# from finddiscord.thrift.v1 import ttypes
# from finddiscord.thrift.v1.FindDiscordService import Client

# from thrift.transport import TSocket
# from thrift.transport import TTransport
# from thrift.protocol import TBinaryProtocol
# from thrift import Thrift

# def get_hotsax_rra_discord(series, win_size):
#     try:
#         tsocket = TSocket.TSocket("127.0.0.1", 9090)
#         transport = TTransport.TFramedTransport(tsocket)
#         protocol = TBinaryProtocol.TBinaryProtocol(transport)
#         client = Client(protocol)

#         req = ttypes.Request()
#         req.test = series
#         req.win_size = win_size
#         transport.open()
#         res = client.say(req)

#         transport.close()

#         return (res.hotsax_abnormal_1_pos,res.hotsax_abnormal_1_score), (res.hotsax_abnormal_2_pos,res.hotsax_abnormal_2_score)

#     except Thrift.TException as ex:
#         print(ex.message)

if __name__ == "__main__":
    pass