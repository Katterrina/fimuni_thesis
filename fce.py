import numpy as np
from scipy import stats

def functional_connectivity(data: np.ndarray, window_len: int) -> np.ndarray:
    """
    Calculates a stream of functional connectivity matrices across the length of the signal.

    data (np.ndarray): matrix of shape (number of channels, time of the measurement)
    window_len (int): length of a sliding window
    """
    # korelace mezi časovými řadami dvou senzorů - personův korelační koeficient

    number_of_channels, time = data.shape

    for t in range(time - window_len):

        FC = np.zeros((number_of_channels,number_of_channels))
        
        for i, channel_i in enumerate(data):
            for j, channel_j in enumerate(data):
                FC[i,j] = stats.pearsonr(channel_i[t:t+window_len],channel_j[t:t+window_len]).statistic

        yield FC

        
def dynamic_functional_connectivity():
    pass

data = np.array([[1, 2, 2, 4, 8, 10], [3, 4, 5, 4, 8, 10], [5, 6, 1, 10, 14, 2]])
time_window_length = 3

for fc in functional_connectivity(data, time_window_length):
