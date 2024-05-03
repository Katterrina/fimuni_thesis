import numpy as np

def find_pivot_to_keep_xpercent_edges(matrix,n_roi=200,percent=0.85):
    pivot_id = int((n_roi**2)*percent)
    matrix_flat_sorted = np.sort(np.nan_to_num(matrix.flatten()))
    return matrix_flat_sorted[pivot_id]

def find_pivot_to_keep_x_edges(matrix,x):
    matrix_flat_sorted = np.flip(np.sort(np.nan_to_num(matrix.flatten())))
    return matrix_flat_sorted[x]

def keep_val_where_weight(SC_W,SC_L):
    np.fill_diagonal(SC_W,np.nan) # do not consider self-loops
    SC_W = np.where(SC_W==0,np.nan, SC_W) # if weight is 0, there is no connection
    SC_L = np.where(np.isnan(SC_W),np.nan,SC_L) # keep lengths only for edges where is connection (based on weights)
    np.nan_to_num(SC_W, copy=False) # convert nan to 0, because the metrics can not handle nans
    return SC_W,SC_L