import numpy as np
from scipy import stats
from scipy import signal

def functional_connectivity(data: np.ndarray, window_len: int) -> np.ndarray:
    """
    Calculates functional connectivity matrices across the length of the signal.

    data (np.ndarray): matrix of shape (number of channels, time of the measurement)
    window_len (int): length of a sliding window

    Returns
    ---
    FCs (np.ndarray): matrix of shape (time, number of channels, number of channels)
    """
    # korelace mezi časovými řadami dvou senzorů - Pearsonův korelační koeficient

    number_of_channels, time = data.shape

    FCs = np.empty((time- window_len,number_of_channels,number_of_channels))

    for t in range(time - window_len):
        for i, channel_i in enumerate(data):
            for j, channel_j in enumerate(data):
                FCs[t,i,j] = stats.pearsonr(channel_i[t:t+window_len],channel_j[t:t+window_len]).statistic

    return FCs


def dynamic_functional_connectivity(FCs):
    """
    Calculates dynamic functional connectivity matrix from functional connectivity matrices.

    FCs (np.ndarray): matrix of shape (time, number of channels, number of channels)

    Returns
    ---
    dFCs (np.ndarray): matrix of shape (time, time)
    """
    time = FCs.shape[0]
    
    dFC = np.empty((time,time))

    indices = np.triu_indices(FCs.shape[1],k=1) # k=1, protože jedničky na diagonále nejsou zajímavé

    for t1 in range(time):
        for t2 in range(time):
            dFC[t1,t2] = stats.pearsonr(FCs[t1][indices],FCs[t2][indices]).statistic

    return dFC

def compute_fluidity(dFC,window_len,n_overlap=None):
    if n_overlap is None:
        n_overlap = window_len - 1
    
    offset = n_overlap/(window_len-n_overlap) + 1
    indices = np.triu_indices(dFC.shape[0],k=offset) 

    fluidity = stats.variation(dFC[indices])

    return fluidity    

data = np.array([[1, 2, 2, 4, 8, 10], [3, 4, 5, 4, 8, 10], [5, 6, 1, 10, 14, 2]])
time_window_length = 3

FCs = functional_connectivity(data, time_window_length)
print(FCs)

dFC = dynamic_functional_connectivity(FCs)
print(dFC)

fluidity = compute_fluidity(dFC,time_window_length,n_overlap=0)
print(fluidity)