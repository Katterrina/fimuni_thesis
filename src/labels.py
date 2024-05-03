import pandas as pd
import numpy as np

from src.paths import path


def ftract_compatible_glasser_labels():
    file = pd.read_csv(path('external/glasser_parcellation_centriods/HCP-MMP1_UniqueRegionList.csv'))  

    def premute_region_name(name):
        if name[:3] == "7Pl": # in F-Tract is uppercase L and we want to match the labels
            name = "7PL"+ name[3:]
        if name[-1] == 'L':
            return 'L_' + name[:-2]
        else:
            return 'R_' + name[:-2]
        
    return file["regionName"].apply(premute_region_name)

def get_labels_from_file(centroids_file,label_column):
    df = pd.read_csv(centroids_file)
    return df[label_column]

def load_ftract_labels(parcellation):
    """
    Load only ROI labels for specified parcellation from F-TRACT dataset.

    Parameters:
    parcellation (str): parcellation name, possble names: Destrieux, DKT, MNI-HCP-MMP1 

    Returns:
    list[str]: list od ROI names
    """
    ftract_path = 'external/F-TRACT/'
    labels = np.loadtxt(path(f'{ftract_path}{parcellation}/{parcellation}.txt'), dtype=str)
    return labels