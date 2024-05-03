import os
import numpy as np

def load_averaged_matrix(path_to_dir,mode,matrix_name):
    mean_file_path = path_to_dir + f"/{mode}_{matrix_name}.npy"

    if os.path.isfile(mean_file_path):
        with open(mean_file_path,"rb") as f:
            M = np.load(f)
            return M
    else:
        return None

def save_averaged_matrix(M,path_to_dir,mode,matrix_name):
    mean_file_path = path_to_dir + f"/{mode}_{matrix_name}.npy"

    with open(mean_file_path,"xb") as f:
        np.save(f,M)