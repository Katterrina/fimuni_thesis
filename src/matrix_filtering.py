import numpy as np

def find_pivot_to_keep_xpercent_edges(matrix,percent=0.85):
    """
    Find a value such that if we use this value as threshold and discard
    all values below the threshold, x% edges remain.    

    Parameters:
    matrix (2D np.array)
    percent (float)

    Returns:
    float: threshold
    """
    n_roi = matrix.shape[0]
    pivot_id = int((n_roi**2)*percent)
    matrix_flat_sorted = np.sort(np.nan_to_num(matrix.flatten()))
    return matrix_flat_sorted[pivot_id]

def find_pivot_to_keep_x_edges(matrix,x):
    """
    Find a value such that if we use this value as threshold and discard
    all values below the threshold, x edges remain.

    Parameters:
    matrix (2D np.array)
    x (int)

    Returns:
    float: threshold
    """
    matrix_flat_sorted = np.flip(np.sort(np.nan_to_num(matrix.flatten())))
    return matrix_flat_sorted[x]

def keep_val_where_weight(SC_W,SC_L):
    """
    Filter structural connectivity lenghts based on weights -- keep lengths
    where weight not None and nonzero.

    Parameters:
    tuple(2D np.array, 2D np.array): 
        structural connectivity weights, structural connectivity lengths

    Returns:
    tuple(2D np.array, 2D np.array): 
        structural connectivity weights, structural connectivity lengths
    """
    np.fill_diagonal(SC_W,np.nan) # do not consider self-loops
    SC_W = np.where(SC_W==0,np.nan, SC_W) # if weight is 0, there is no connection
    SC_L = np.where(np.isnan(SC_W),np.nan,SC_L) # keep lengths only for edges where is connection (based on weights)
    np.nan_to_num(SC_W, copy=False) # convert nan to 0, because the metrics can not handle nans
    return SC_W,SC_L