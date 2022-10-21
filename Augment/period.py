from scipy.fftpack import fft, fftfreq
import numpy as np
from statsmodels.tsa.stattools import acf

def get_top_k_period(data):
    fft_series = fft(data)
    power = np.abs(fft_series)
    sample_freq = fftfreq(fft_series.size)

    pos_mask = np.where(sample_freq > 0)
    freqs = sample_freq[pos_mask]
    powers = power[pos_mask]

    top_k_seasons = 3
    # top K=3 index
    top_k_idxs = np.argpartition(powers, -top_k_seasons)[-top_k_seasons:]
    top_k_power = powers[top_k_idxs]
    fft_periods = (1 / freqs[top_k_idxs]).astype(int)

    #print(f"top_k_power: {top_k_power}")
    #print(f"fft_periods: {fft_periods}")

    return fft_periods

def find_best_period(data):
    top3_periods = get_top_k_period(data)

    max_period = None
    max_acf_score = -np.inf
    for lag in top3_periods:
        # lag = fft_periods[np.abs(fft_periods - time_lag).argmin()]
        acf_score = acf(data, nlags=lag)[-1]
        #print(f"lag: {lag} fft acf: {acf_score}")
        if max_acf_score < acf_score:
            max_acf_score = acf_score
            max_period = lag 
    
    return max_period