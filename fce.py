import numpy as np
from scipy import stats
from scipy import signal

def functional_connectivity(data: np.ndarray, window_len: int) -> np.ndarray:
    """
    Calculates a stream of functional connectivity matrices across the length of the signal.

    data (np.ndarray): matrix of shape (number of channels, time of the measurement)
    window_len (int): length of a sliding window
    """
    # korelace mezi časovými řadami dvou senzorů - personův korelační koeficient

    number_of_channels, time = data.shape

    FCs = np.empty((time- window_len,number_of_channels,number_of_channels))

    for t in range(time - window_len):
        for i, channel_i in enumerate(data):
            for j, channel_j in enumerate(data):
                FCs[t,i,j] = stats.pearsonr(channel_i[t:t+window_len],channel_j[t:t+window_len]).statistic

    return FCs


def dynamic_functional_connectivity(FCs):
    time = FCs.shape[0]
    
    dFC = np.empty((time,time))

    indices = np.triu_indices(FCs.shape[1],k=1) # k=1, protože jedničky na diagonále nejsou zajímavé

    for t1 in range(time):
        for t2 in range(time):
            dFC[t1,t2] = stats.pearsonr(FCs[t1][indices],FCs[t2][indices]).statistic

    return dFC
    

data = np.array([[1, 2, 2, 4, 8, 10], [3, 4, 5, 4, 8, 10], [5, 6, 1, 10, 14, 2]])
time_window_length = 3

FCs = functional_connectivity(data, time_window_length)
print(FCs)

dFC = dynamic_functional_connectivity(FCs)
print(dFC)