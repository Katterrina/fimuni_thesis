from src.struct_consensus_netneurotools import *

def create_averaged_matrix_based_on_mode(mode,M,distances=None):
    """
    Create group representative average matrix based on 
    avereging method selcted by mode parameter from options
    ["simple","cons","dist","rh"].
        - simple: simple average
        - cons: consensus thresholding
        - dist: distance-dependent consensus thresholding
        - rh: Rosen and Halgren's ageraging method

    Parameters:
    mode (str): indicator of averaging method, options
        ["simple","cons","dist","rh"]
    M (3D numpy array (number_of_subjects,n_roi,n_roi)):
        structural connectivity matrices for all subjects 
    distances (2D np.array): ROI distance matrix, i.e. Euclidean
        distances of ROIs, necessary for "dist" mode

    Returns:
    2D np.array, group-averaged matrix
    """
    SC_W = None

    if mode == "simple":
        SC_W = simple_averaging(M)
    elif mode == "dist":
        if distances is None:
            print(f"Node distances necessary to calculate distance dependent thresholding!")
        else:
            SC_W = struct_consensus(np.transpose(M),distances,weighted=True)
    elif mode == "rh":
        SC_W = rosenhalgren_sc_averaging(M)
    elif mode == "cons":
        SC_W = consensus_thresholding(M)
    else:
        print("Invalid mode!")

    # we do not consider directed edges, so the resulting matrix should be symmetric
    # it is not always the case because of numerical instability, so we enforce symetry here
    SC_W = (SC_W+SC_W.T) /2 
    
    np.fill_diagonal(SC_W,np.nan)
    return SC_W

def consensus_thresholding(M,tau=0.75):
    """
    Consensus thresholding method for group-averaging structural matrices 
    stored in 3D input matrix M of shape (number_of_subjects,n_roi,n_roi)

    Parameters: 
    M (3D numpy array (number_of_subjects,n_roi,n_roi)):
        structural connectivity matrices for all subjects 
    tau (float form interval (0,1)): ratio of subjects that are
        supposed to have an edge ij to keep the edge ij in the group
        average

    Returns:
    2D np.array, group-averaged matrix
    """
    
    counts = np.count_nonzero(M,axis=0)
    n_subjects = M.shape[0]
    frac = counts / n_subjects

    SC = np.where(frac > tau,np.nanmean(M,axis=0),np.nan)
    return SC

def rosenhalgren_sc_averaging(M):
    """
    Rosen and Halgren's method for group-averaging structural matrices 
    stored in 3D input matrix M of shape (number_of_subjects,n_roi,n_roi)

    Parameters:
    M (3D numpy array (number_of_subjects,n_roi,n_roi)):
        structural connectivity matrices for all subjects 

    Returns:
    2D np.array, group-averaged matrix
    """
    M_mean = np.nanmean(M,axis=0)
    SC = np.zeros(M_mean.shape)

    for i in range(M_mean.shape[0]):
        for j in range(M_mean.shape[1]):
            SC[i][j] = M_mean[i][j] / (np.nansum(M_mean[i,:]) + np.nansum(M_mean[:,j]) - M_mean[i][i] - M_mean[j][j])

    return SC

def simple_averaging(M):
    """
    Simple method for group-averaging structural matrices stored in 3D 
    input matrix M of shape (number_of_subjects,n_roi,n_roi)

    Parameters:
    M (3D numpy array (number_of_subjects,n_roi,n_roi)):
        structural connectivity matrices for all subjects 

    Returns:
    2D np.array, group-averaged matrix
    """
    M = np.nan_to_num(M,nan=0)
    return np.nanmean(M,axis=0)