import pandas as pd
import numpy as np

from src.paths import *
from src.labels import *

def roi_distances_from_centroids(centroids):
    """
    Calculate all distances between individual ROIs represented 
    by centroids given as coordinates.
    
    Parameters:
    centroids (2D np.array of shape (n_roi,3)): coordinates for each ROI

    Returns:
    2D np.array of shape (n_roi,n_roi): distances between all pair of ROIs
    """
    n_roi = len(centroids)

    distances = np.zeros((n_roi,n_roi))
    for i in range(n_roi):
        for j in range(n_roi):
            distances[i,j] = np.linalg.norm(centroids[i]-centroids[j])

    return distances

def get_centroids_from_file(centroids_file,geom_column):
    """
    Load coordinates of ROI centroids from csv file, expecting the geom_column
    in the file contains coordinates in [x,y,z] form.

    Parameters:
    centroids_file (path): path to csv file with column containing centroid
        coordinates in a list [x,y,z]
    geom_column (str): string indicating in which column in csv file are
        the coordinates of centroids

    Returns:
    centroids (2D np.array of shape (n_roi,3)): coordinates for each ROI
    """
    df = pd.read_csv(centroids_file)
    centroids = np.stack(df[geom_column].apply(lambda x: np.fromstring(x[1:-1], dtype=float, sep=' ')).to_numpy(),axis=0)
    return centroids

def load_glasser_centroids(ftract_labels=None):
    """
    Load coordinates of ROI centroids for Glasser parcellation.

    Parameters:
    ftract_labels (list[str]|None): list of F-TRACT labels if we want 
        to reorder based on F-TRACT, if None do not reorder

    Returns:
    centroids (2D np.array of shape (n_roi,3)): coordinates for each ROI
    """
    centroids_file = pd.read_csv(path('external/glasser_parcellation_centriods/HCP-MMP1_UniqueRegionList.csv'))   
    
    if ftract_labels is not None:
        centroids_file['ftract_labels'] = ftract_compatible_glasser_labels()
        centroids_file = centroids_file.set_index('ftract_labels')
        centroids_file = centroids_file.reindex(index=ftract_labels)
    
    n_roi = len(centroids_file)
    centroids = np.zeros((n_roi,3))
    centroids[:,0] = centroids_file["x-cog"]
    centroids[:,1] = centroids_file["y-cog"]
    centroids[:,2] = centroids_file["z-cog"]

    return centroids

def glasser_roi_distances(ftract_labels=None):
    """
    Load matrix of Euclidean distances for ROIs in Glasser parcellation.
    If ftract_labels list not None, reorder based on the labels.

    Parameters:
    ftract_labels (list[str]|None): list of F-TRACT labels if we want 
        to reorder based on F-TRACT, if None do not reorder

    Returns:
    2D np.array of shape (n_roi,n_roi): distances between all pair of ROIs
    """
    centroids = load_glasser_centroids(ftract_labels=ftract_labels)
    distance_matrix = roi_distances_from_centroids(centroids)

    return distance_matrix

def dkt_roi_distances():
    """
    Load matrix of Euclidean distances for ROIs in DeskianKilliany parcellation.

    Returns:
    2D np.array of shape (n_roi,n_roi): distances between all pair of ROIs
    """
    file = path("external/dk_parcellation_centroids/dk_centroids.csv")
    centroids = get_centroids_from_file(file,"geom")
    distance_matrix = roi_distances_from_centroids(centroids)

    return distance_matrix

def schaefer_roi_distances():
    """
    Load matrix of Euclidean distances for ROIs in Schaefer200 parcellation.

    Returns:
    2D np.array of shape (n_roi,n_roi): distances between all pair of ROIs
    """
    centroids_file = path('external/schaefer_parcellation_centroids/ROI_MAPPING_pytepfit.csv')
    centroids_mne = get_centroids_from_file(centroids_file,"geom_mne")
    distance_matrix = roi_distances_from_centroids(centroids_mne)

    return distance_matrix