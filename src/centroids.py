import pandas as pd
import numpy as np

from src.paths import *
from src.labels import *

def roi_distances_from_centroids(centroids):
    n_roi = len(centroids)

    distances = np.zeros((n_roi,n_roi))
    for i in range(n_roi):
        for j in range(n_roi):
            distances[i,j] = np.linalg.norm(centroids[i]-centroids[j])

    return distances

def get_centroids_from_file(centroids_file,geom_column):
    df = pd.read_csv(centroids_file)
    centroids = np.stack(df[geom_column].apply(lambda x: np.fromstring(x[1:-1], dtype=float, sep=' ')).to_numpy(),axis=0)
    return centroids

def load_glasser_centroids(ftract_labels=None):
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

    centroids = load_glasser_centroids(ftract_labels=ftract_labels)
    distance_matrix = roi_distances_from_centroids(centroids)

    return distance_matrix

def dkt_roi_distances():

    file = path("external/dk_parcellation_centroids/dk_centroids.csv")
    centroids = get_centroids_from_file(file,"geom")
    distance_matrix = roi_distances_from_centroids(centroids)

    return distance_matrix

def schaefer_roi_distances():
    centroids_file = path('external/schaefer_parcellation_centroids/ROI_MAPPING_pytepfit.csv')
    centroids_mne = get_centroids_from_file(centroids_file,"geom_mne")

    ED = roi_distances_from_centroids(centroids_mne)
    return ED
