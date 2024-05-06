import os
import numpy as np

def load_averaged_matrix(path_to_dir,mode,matrix_name):
    """
    Load averaged matrix.

    Parameters:
    path_to_dir (str/path): path to data directory
    mode (str): wich group averaging method was used to create the matrix
    matrix_name (str): usually SC_W for weights or SC_L for lengths
    
    Returns:
    2D np.array or None
    """
    mean_file_path = path_to_dir + f"/{mode}_{matrix_name}.npy"

    if os.path.isfile(mean_file_path):
        with open(mean_file_path,"rb") as f:
            M = np.load(f)
            return M
    else:
        return None

def save_averaged_matrix(M,path_to_dir,mode,matrix_name):
    """
    Save averaged matrix.

    Parameters:
    path_to_dir (str/path): path to data directory
    mode (str): wich group averaging method was used to create the matrix
    matrix_name (str): usually SC_W for weights or SC_L for lengths
    """
    mean_file_path = path_to_dir + f"/{mode}_{matrix_name}.npy"

    with open(mean_file_path,"xb") as f:
        np.save(f,M)