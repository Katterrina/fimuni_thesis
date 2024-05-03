from src.struct_consensus_netneurotools import *

def create_averaged_matrix_based_on_mode(mode,M,distances=None):
    SC_W = None

    if mode == "simple":
        SC_W = simple_averaging(M)
    elif mode == "dist":
        if distances is None:
            print(f"Node distances necessary to calculate distance dependent thresholding!")
        else:
            M = np.transpose(M)
            SC_W = struct_consensus(M,distances,weighted=True)
    elif mode == "rh":
        SC_W = rosenhalgren_sc_averaging(M)
    elif mode == "cons":
        SC_W = consensus_thresholding(M,tau=0.75)
    else:
        print("Invalid mode!")

    # we do not consider directed edges, so the resulting matrix should be symmetric
    # it is not always the case because of numerical instability, so we enforce symetry here
    SC_W = (SC_W+SC_W.T) /2 

    np.fill_diagonal(SC_W,np.nan)
    return SC_W

def consensus_thresholding(M,tau=0.5):
    M = np.where(M==0,np.nan,M)
    counts = np.count_nonzero(M,axis=0)
    n_subjects = M.shape[0]
    frac = counts / n_subjects

    SC = np.where(frac > tau,np.nanmean(M,axis=0),np.nan)
    return SC

def rosenhalgren_sc_averaging(M):
    M_mean = np.mean(M,axis=0)
    SC = np.zeros(M_mean.shape)

    for i in range(M_mean.shape[0]):
        for j in range(M_mean.shape[1]):
            SC[i][j] = M_mean[i][j] / (np.sum(M_mean[i,:]) + np.sum(M_mean[:,j]) - M_mean[i][i] - M_mean[j][j])

    return SC

def simple_averaging(M):
    return np.mean(M,axis=0)