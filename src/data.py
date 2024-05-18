import os 
import h5py
import numpy as np
import pandas as pd
import logging

from src.roi_remappnig import *
from src.paths import *
from src.labels import *
from src.group_averaging import *
from src.matrix_filtering import *
from src.save_and_load_averaged import *
from src.centroids import *

# ===================================================================
# ======== loading sets of SC matrices for various purposes =========
# ===================================================================

def load_matrices_for_specified_preprocessings(matrix_loader,
                                               dataset_name,
                                               parcellation,
                                               ED=None,
                                               min_streamlines_count=5,
                                               averaging_methods =["simple","cons","dist","rh"]):
    """
    Load a set of matrices from a dataset, one matrix per averaging method specified
    in a list of averaging methods.

    Parameters:
    matrix_loader (function): function loading individual matrices based on 
        parameters parcellation (str), mode (str, indicates the averaging method),
        and distances (ROI distances in the parcellation, necessary when dist in 
        list of averaging methods) 
    dataset_name (str): string for indication of the data source in the result
    parcellation (str): string indicating the brain parcellation, posible options
        depend on the dataset
    ED (2D np.array): matrix of Euclidean distances of ROI in selected parcellation
    averaging_methods (list[str]): list of averaging methods we want to load,
        possible options ["simple","cons","dist","rh"]

    Returns:
    list[(str, 2D np.array, 2D np.array, 2D np.array)]: 
        string in a form <dataset>_<averaging method>
        structural connectivity weights
        structural connectivity lengths
        logarithm of structural connectivity weights
    """
    
    SC_matrices= []

    for mode in averaging_methods:
        if mode == "dist" and ED is None:
            continue
        SC_W, SC_L = matrix_loader(parcellation=parcellation,min_streamlines_count=min_streamlines_count,mode=mode,distances=ED)
        SC_matrices.append((f"{dataset_name}_{mode}",SC_W, SC_L,np.log(SC_W)))
    
    return SC_matrices

def load_set_of_glasser_matrices_for_ftract(ED=None,min_streamlines_count=5):
    """
    Load all matrices available in Glasser (MNI-HCP-MMP1) parcellation 
    and reorder them to match the ordering of ROIs in F-TRACT.

    Parameters:
    ED (2D np.array): matrix of Euclidean distances of ROI in Glasser parcellation
        ordered by F-TRACT labels
    min_streamlines_count (int): minimal number of streamlines between two ROIs
        in one subject to assume that there is an edge connecting them

    Returns:
    list[(str, 2D np.array, 2D np.array, 2D np.array)]: 
        string in a form <dataset>_<averaging method>
        structural connectivity weights
        structural connectivity lengths
        logarithm of structural connectivity weights    
    """

    ftract_labels = load_ftract_labels("MNI-HCP-MMP1")

    # load and preprocess Glasser labels such that they match 
    # the labels in F-TTRACT
    labels = ftract_compatible_glasser_labels() 

    SC_matrices = []

    # Mica-Mics dataset
    SC_matrices_mica = load_matrices_for_specified_preprocessings(load_mica_matrix,"Mica-Mics","glasser360",ED,min_streamlines_count=min_streamlines_count)

    for name, SC_W, SC_L, SC_W_log in SC_matrices_mica:
        SC_W, SC_L  = keep_val_where_weight(SC_W, SC_L)

        SC_W = reorder_matrix_based_on_reference(labels,ftract_labels,SC_W)
        SC_L = reorder_matrix_based_on_reference(labels,ftract_labels,SC_L)
        SC_W_log = reorder_matrix_based_on_reference(labels,ftract_labels,SC_W_log)

        SC_matrices.append((name, SC_W, SC_L, SC_W_log))

    # Enigma dataset
    SC_W_E = load_enigma_sc_matrix(parcellation="glasser_360")
    SC_W_E = reorder_matrix_based_on_reference(labels,ftract_labels,SC_W_E)
    SC_matrices.append(("Enigma_dist",np.exp(SC_W_E), None,SC_W_E))

    # Rosen-Halgren dataset
    SC_W_RH_log, SC_L_RH = load_rosen_halgren(ftract_labels)
    SC_W_RH_log, SC_L_RH = keep_val_where_weight(SC_W_RH_log, SC_L_RH)
    SC_matrices.append(("Rosen-Halgren_rh",10**SC_W_RH_log, SC_L_RH,SC_W_RH_log))

    return SC_matrices

def load_set_of_schaefer_matrices_for_pytepfit(ED=None,averaging_methods=None,min_streamlines_count=5):
    """
    Load all matrices available in Schaefer parcellation and reorder
    them to match the ordering of ROIs in PyTepFit TMS-EEG.

    Parameters:
    ED (2D np.array): matrix of Euclidean distances of ROI in Glasser parcellation
        ordered by F-TRACT labels
    averaging_methods (list[str]): list of averaging methods we want to load,
        possible options ["simple","cons","dist","rh"], used when dataset allows
    min_streamlines_count (int): minimal number of streamlines between two ROIs
        in one subject to assume that there is an edge connecting them

    Returns:
    list[(str, 2D np.array, 2D np.array, 2D np.array)]: 
        string in a form <dataset>_<averaging method>
        structural connectivity weights
        structural connectivity lengths
        logarithm of structural connectivity weights    
    """

    SC_matrices = []

    SC_W_pytep, SC_L_pytep = load_pytepfit_sc()
    SC_matrices.append(("PyTepFit_simple",SC_W_pytep, SC_L_pytep,np.log(SC_W_pytep)))

    SC_W_ENIGMA = load_enigma_sc_matrix(parcellation="schaefer_200",reoreder='PyTepFit')
    SC_matrices.append(("Enigma_dist",np.exp(SC_W_ENIGMA), None,SC_W_ENIGMA,))

    parcellation = "schaefer200"
    if averaging_methods is None:
        SC_matrices_domhof = load_matrices_for_specified_preprocessings(load_domhof_matrix_for_pytepfit,"Domhof",parcellation,ED,min_streamlines_count=min_streamlines_count)
        SC_matrices_mica = load_matrices_for_specified_preprocessings(load_mica_matrix_for_pytepfit,"Mica-Mics",parcellation,ED,min_streamlines_count=min_streamlines_count)
    else:
        SC_matrices_domhof = load_matrices_for_specified_preprocessings(load_domhof_matrix_for_pytepfit,"Domhof",parcellation,ED,
                                                                        min_streamlines_count=min_streamlines_count,averaging_methods=averaging_methods)
        SC_matrices_mica = load_matrices_for_specified_preprocessings(load_mica_matrix_for_pytepfit,"Mica-Mics",parcellation,ED,
                                                                        min_streamlines_count=min_streamlines_count,averaging_methods=averaging_methods)
    return SC_matrices+SC_matrices_domhof+SC_matrices_mica

def load_set_of_DKT_matrices_for_ftract(ids_to_delete_in_dkt=[37,7],ED=None,min_streamlines_count=5):
    """
    Load all matrices available in DeskianKilliany parcellation 
    and reorder them to match the ordering of ROIs in F-TRACT.

    Parameters:
    ids_to_delete_in_dkt (list[str]): we found that some datasets indicate 
        that they use DeskianKilliany parcellation, which is supposed to 
        have 68 ROIs, but the mshape of matrices is 70x70, therefore it is 
        necessary to remove two ROIs to have those matrices compatible with 
        the rest, which really has the shape 68x68 (specific indeces that 
        should be deleted were found ad hoc)
    ED (2D np.array): matrix of Euclidean distances of ROI in Glasser parcellation
        ordered by F-TRACT labels
    min_streamlines_count (int): minimal number of streamlines between two ROIs
        in one subject to assume that there is an edge connecting them

    Returns:
    list[(str, 2D np.array, 2D np.array, 2D np.array)]: 
        string in a form <dataset>_<averaging method>
        structural connectivity weights
        structural connectivity lengths
        logarithm of structural connectivity weights    
    """

    SC_matrices = []

    SC_matrices_domhof = load_matrices_for_specified_preprocessings(load_domhof_matrix,
                                                                    "Domhof",
                                                                    "DKT",
                                                                    ED,min_streamlines_count=min_streamlines_count,
                                                                    averaging_methods=["simple","cons","rh"])

    for name, SC_W_D, SC_L_D, SC_W_log_D in SC_matrices_domhof:
        SC_W_D, SC_L_D  = keep_val_where_weight(SC_W_D, SC_L_D)
        for a in [0,1]:
            SC_W_D = np.delete(SC_W_D,ids_to_delete_in_dkt,axis=a)
            SC_L_D = np.delete(SC_L_D,ids_to_delete_in_dkt,axis=a)
            SC_W_log_D = np.delete(SC_W_log_D,ids_to_delete_in_dkt,axis=a)

        SC_matrices.append((name, SC_W_D, SC_L_D, SC_W_log_D))

    SC_W_E = load_enigma_sc_matrix(parcellation="DKT")
    SC_matrices.append(("Enigma_dist",np.exp(SC_W_E), None,SC_W_E))

    return SC_matrices

# ===================================================================
# ============================= F-TRACT =============================
# ===================================================================

def load_ftract(parcellation,short=False):
    """
    Load F-TRACT functional dataset.

    Parameters:
    parcellation (str): string indicating which brain parcellation we
        want to use, possible options based on downloaded data
    short (bool): indicator if we want to use the long (200 ms)
        or short (50 ms) version of F-TRACT responses

    Returns:
    tuple (probability, amplitude, delay_onset, delay_peak, n_stim, n_impl, labels) 
        probability, amplitude, delay_onset, delay_peak: 2D np.arrays 
            characterizing the stimulus response
        n_stim: 2D np.array
            number of stimulations/measurements for each pair of nodes
        n_impl: 
            number of times electrodes were implanted for each pair of nodes
    """

    if short:
        ftract_path = 'external/F-TRACT_short/'
    else:
        ftract_path = 'external/F-TRACT/'
        
    probability = np.loadtxt(path(f'{ftract_path}{parcellation}/probability.txt.gz'))
    amplitude = np.loadtxt(path(f'{ftract_path}{parcellation}/amplitude__median.txt.gz'))
    delay_onset = np.loadtxt(path(f'{ftract_path}{parcellation}/onset_delay__median.txt.gz'))
    delay_peak = np.loadtxt(path(f'{ftract_path}{parcellation}/peak_delay__median.txt.gz'))
    n_stim = np.loadtxt(path(f'{ftract_path}{parcellation}/N_stimulations.txt.gz'))
    n_impl = np.loadtxt(path(f'{ftract_path}{parcellation}/N_implantations.txt.gz'))

    labels = load_ftract_labels(parcellation)

    # this is necessary because files with labels for these parcellations contain many other labels, which are not used
    if parcellation == "DKT" or parcellation == "Destrieux":
        if parcellation == "DKT":
            roi = ['ctx-lh-bankssts','ctx-lh-caudalanteriorcingulate','ctx-lh-caudalmiddlefrontal','ctx-lh-corpuscallosum','ctx-lh-cuneus','ctx-lh-entorhinal','ctx-lh-fusiform','ctx-lh-inferiorparietal','ctx-lh-inferiortemporal','ctx-lh-isthmuscingulate','ctx-lh-lateraloccipital','ctx-lh-lateralorbitofrontal','ctx-lh-lingual','ctx-lh-medialorbitofrontal','ctx-lh-middletemporal','ctx-lh-parahippocampal','ctx-lh-paracentral','ctx-lh-parsopercularis','ctx-lh-parsorbitalis','ctx-lh-parstriangularis','ctx-lh-pericalcarine','ctx-lh-postcentral','ctx-lh-posteriorcingulate','ctx-lh-precentral','ctx-lh-precuneus','ctx-lh-rostralanteriorcingulate','ctx-lh-rostralmiddlefrontal','ctx-lh-superiorfrontal','ctx-lh-superiorparietal','ctx-lh-superiortemporal','ctx-lh-supramarginal','ctx-lh-frontalpole','ctx-lh-temporalpole','ctx-lh-transversetemporal','ctx-lh-insula','ctx-rh-bankssts','ctx-rh-caudalanteriorcingulate','ctx-rh-caudalmiddlefrontal','ctx-rh-corpuscallosum','ctx-rh-cuneus','ctx-rh-entorhinal','ctx-rh-fusiform','ctx-rh-inferiorparietal','ctx-rh-inferiortemporal','ctx-rh-isthmuscingulate','ctx-rh-lateraloccipital','ctx-rh-lateralorbitofrontal','ctx-rh-lingual','ctx-rh-medialorbitofrontal','ctx-rh-middletemporal','ctx-rh-parahippocampal','ctx-rh-paracentral','ctx-rh-parsopercularis','ctx-rh-parsorbitalis','ctx-rh-parstriangularis','ctx-rh-pericalcarine','ctx-rh-postcentral','ctx-rh-posteriorcingulate','ctx-rh-precentral','ctx-rh-precuneus','ctx-rh-rostralanteriorcingulate','ctx-rh-rostralmiddlefrontal','ctx-rh-superiorfrontal','ctx-rh-superiorparietal','ctx-rh-superiortemporal','ctx-rh-supramarginal','ctx-rh-frontalpole','ctx-rh-temporalpole','ctx-rh-transversetemporal','ctx-rh-insula']
        else:
            roi = ['ctx_lh_G_and_S_frontomargin','ctx_lh_G_and_S_occipital_inf','ctx_lh_G_and_S_paracentral','ctx_lh_G_and_S_subcentral','ctx_lh_G_and_S_transv_frontopol','ctx_lh_G_and_S_cingul-Ant','ctx_lh_G_and_S_cingul-Mid-Ant','ctx_lh_G_and_S_cingul-Mid-Post','ctx_lh_G_cingul-Post-dorsal','ctx_lh_G_cingul-Post-ventral','ctx_lh_G_cuneus','ctx_lh_G_front_inf-Opercular','ctx_lh_G_front_inf-Orbital','ctx_lh_G_front_inf-Triangul','ctx_lh_G_front_middle','ctx_lh_G_front_sup','ctx_lh_G_Ins_lg_and_S_cent_ins','ctx_lh_G_insular_short','ctx_lh_G_occipital_middle','ctx_lh_G_occipital_sup','ctx_lh_G_oc-temp_lat-fusifor','ctx_lh_G_oc-temp_med-Lingual','ctx_lh_G_oc-temp_med-Parahip','ctx_lh_G_orbital','ctx_lh_G_pariet_inf-Angular','ctx_lh_G_pariet_inf-Supramar','ctx_lh_G_parietal_sup','ctx_lh_G_postcentral','ctx_lh_G_precentral','ctx_lh_G_precuneus','ctx_lh_G_rectus','ctx_lh_G_subcallosal','ctx_lh_G_temp_sup-G_T_transv','ctx_lh_G_temp_sup-Lateral','ctx_lh_G_temp_sup-Plan_polar','ctx_lh_G_temp_sup-Plan_tempo','ctx_lh_G_temporal_inf','ctx_lh_G_temporal_middle','ctx_lh_Lat_Fis-ant-Horizont','ctx_lh_Lat_Fis-ant-Vertical','ctx_lh_Lat_Fis-post','ctx_lh_Medial_wall','ctx_lh_Pole_occipital','ctx_lh_Pole_temporal','ctx_lh_S_calcarine','ctx_lh_S_central','ctx_lh_S_cingul-Marginalis','ctx_lh_S_circular_insula_ant','ctx_lh_S_circular_insula_inf','ctx_lh_S_circular_insula_sup','ctx_lh_S_collat_transv_ant','ctx_lh_S_collat_transv_post','ctx_lh_S_front_inf','ctx_lh_S_front_middle','ctx_lh_S_front_sup','ctx_lh_S_interm_prim-Jensen','ctx_lh_S_intrapariet_and_P_trans','ctx_lh_S_oc_middle_and_Lunatus','ctx_lh_S_oc_sup_and_transversal','ctx_lh_S_occipital_ant','ctx_lh_S_oc-temp_lat','ctx_lh_S_oc-temp_med_and_Lingual','ctx_lh_S_orbital_lateral','ctx_lh_S_orbital_med-olfact','ctx_lh_S_orbital-H_Shaped','ctx_lh_S_parieto_occipital','ctx_lh_S_pericallosal','ctx_lh_S_postcentral','ctx_lh_S_precentral-inf-part','ctx_lh_S_precentral-sup-part','ctx_lh_S_suborbital','ctx_lh_S_subparietal','ctx_lh_S_temporal_inf','ctx_lh_S_temporal_sup','ctx_lh_S_temporal_transverse','ctx_rh_G_and_S_frontomargin','ctx_rh_G_and_S_occipital_inf','ctx_rh_G_and_S_paracentral','ctx_rh_G_and_S_subcentral','ctx_rh_G_and_S_transv_frontopol','ctx_rh_G_and_S_cingul-Ant','ctx_rh_G_and_S_cingul-Mid-Ant','ctx_rh_G_and_S_cingul-Mid-Post','ctx_rh_G_cingul-Post-dorsal','ctx_rh_G_cingul-Post-ventral','ctx_rh_G_cuneus','ctx_rh_G_front_inf-Opercular','ctx_rh_G_front_inf-Orbital','ctx_rh_G_front_inf-Triangul','ctx_rh_G_front_middle','ctx_rh_G_front_sup','ctx_rh_G_Ins_lg_and_S_cent_ins','ctx_rh_G_insular_short','ctx_rh_G_occipital_middle','ctx_rh_G_occipital_sup','ctx_rh_G_oc-temp_lat-fusifor','ctx_rh_G_oc-temp_med-Lingual','ctx_rh_G_oc-temp_med-Parahip','ctx_rh_G_orbital','ctx_rh_G_pariet_inf-Angular','ctx_rh_G_pariet_inf-Supramar','ctx_rh_G_parietal_sup','ctx_rh_G_postcentral','ctx_rh_G_precentral','ctx_rh_G_precuneus','ctx_rh_G_rectus','ctx_rh_G_subcallosal','ctx_rh_G_temp_sup-G_T_transv','ctx_rh_G_temp_sup-Lateral','ctx_rh_G_temp_sup-Plan_polar','ctx_rh_G_temp_sup-Plan_tempo','ctx_rh_G_temporal_inf','ctx_rh_G_temporal_middle','ctx_rh_Lat_Fis-ant-Horizont','ctx_rh_Lat_Fis-ant-Vertical','ctx_rh_Lat_Fis-post','ctx_rh_Medial_wall','ctx_rh_Pole_occipital','ctx_rh_Pole_temporal','ctx_rh_S_calcarine','ctx_rh_S_central','ctx_rh_S_cingul-Marginalis','ctx_rh_S_circular_insula_ant','ctx_rh_S_circular_insula_inf','ctx_rh_S_circular_insula_sup','ctx_rh_S_collat_transv_ant','ctx_rh_S_collat_transv_post','ctx_rh_S_front_inf','ctx_rh_S_front_middle','ctx_rh_S_front_sup','ctx_rh_S_interm_prim-Jensen','ctx_rh_S_intrapariet_and_P_trans','ctx_rh_S_oc_middle_and_Lunatus','ctx_rh_S_oc_sup_and_transversal','ctx_rh_S_occipital_ant','ctx_rh_S_oc-temp_lat','ctx_rh_S_oc-temp_med_and_Lingual','ctx_rh_S_orbital_lateral','ctx_rh_S_orbital_med-olfact','ctx_rh_S_orbital-H_Shaped','ctx_rh_S_parieto_occipital','ctx_rh_S_pericallosal','ctx_rh_S_postcentral','ctx_rh_S_precentral-inf-part','ctx_rh_S_precentral-sup-part','ctx_rh_S_suborbital','ctx_rh_S_subparietal','ctx_rh_S_temporal_inf','ctx_rh_S_temporal_sup','ctx_rh_S_temporal_transverse'] 

        parcell_ids =  [i for i,label in enumerate(labels) if label in roi]

        probability = probability[np.ix_(parcell_ids, parcell_ids)]
        amplitude  = amplitude[np.ix_(parcell_ids, parcell_ids)]
        delay_onset = delay_onset[np.ix_(parcell_ids, parcell_ids)]
        delay_peak = delay_peak[np.ix_(parcell_ids, parcell_ids)]
        n_stim = n_stim[np.ix_(parcell_ids, parcell_ids)]
        n_impl = n_impl[np.ix_(parcell_ids, parcell_ids)]

        labels = roi

    return probability, amplitude, delay_onset, delay_peak, n_stim, n_impl, labels


# ===================================================================
# ============================= ENIGMA ==============================
# ===================================================================

try:
    from enigmatoolbox.datasets import load_sc
except:
    logging.warning('ENIGMA toolbox not installed, respective data loading will be not available. Install ENIGMA toolbox before further use.')


def load_enigma_sc_matrix(parcellation=None,reoreder=None):
    """
    Load structural connectivity matrix from ENIGMA TOOLBOX based on parcellation
    and eventually reorder based on PyTepFit ROI mapping table.

    Parameters:
    parcellation (str): parcellation name, possble names: Destrieux, DKT, MNI-HCP-MMP1 

    Returns:
    2D numpy array: structural connectivity matrix   
    """
    if parcellation == "DKT":
        SC, _, _, _ = load_sc()
    else:
        SC, _, _, _ = load_sc(parcellation=parcellation)

    if reoreder=='PyTepFit':
        mapping_path = path('interim/schaefer_parcellation_mappings/ROI_MAPPING_pytepfit.csv')
        SC = schaefer_to_schaefer(SC,mapping_path,"idx_csv")       

    return SC 

# ===================================================================
# ============================= Domhof ==============================
# ===================================================================

def load_subjects_3Dmatrix_domhof(path_to_dir,csv_name,n_roi,number_of_subjects=200):
    """
    Load 3D matrix consisting of individual structural connectivity matrices
    from Domhof dataset.

    Parameters:
    path_to_dir (str/path): path to directory where data for individual 
        subjects are stored
    csv_name (str): name of the files  where desired matrices are stored,
        usually Counts.csv or Lengths.csv 
    n_roi (int): number of ROIs, specify based on parcellation, sholud be
        the same as the dimensions of individual matrices we want to load
    number_of_subjects (int): number of subjects we want to load, usually
        the number of subject foledrs in path_to_dir folder

    Returns:
    3D numpy array (number_of_subjects,n_roi,n_roi): structural connectivity matrices
        for all subjects defined by number_of_subjects    
    """
    M = np.zeros((number_of_subjects,n_roi,n_roi))
    for i in range(number_of_subjects):
        counts_file = path_to_dir+f"/{i:03d}/"+csv_name
        with open(counts_file,"r") as cf:
            c = np.genfromtxt(cf)
        M[i] = c

    return M

def load_domhof_matrix(parcellation,min_streamlines_count=5,mode="simple",distances=None):
    """
    Load group averaged matrix from Domhof dataset based on
    averaging method mode.

    Parameters:
    parcellation (str): string indicating which brain parcellation we
        want to use, possible options based on downloaded data
    min_streamlines_count (int): minimal number of streamlines between two ROIs
        in one subject to assume that there is an edge connecting them
    mode (str): group averaging mode, select an option 
        from ["simple","cons","dist","rh"]
            - simple: simple average
            - cons: consensus thresholding
            - dist: distance-dependent consensus thresholding
            - rh: Rosen and Halgren's ageraging method
    distances (2D np.array): ROI distance matrix, i.e. Euclidean
        distances of ROIs, necessary for "dist" mode

    Returns:
    tuple(2D np.array, 2D np.array): 
        structural connectivity weights, structural connectivity lengths
    """
    if parcellation == "DKT":
        n_roi = 70
    elif parcellation == "schaefer200":
        n_roi = 200

    rootdir = f"external/domhof/{parcellation}/"
    scdir = "1StructuralConnectivity/"
    path_to_dir = path(rootdir+scdir)

    matrix_W_name = f"SC_W_min{min_streamlines_count}"
    matrix_L_name = f"SC_L_min{min_streamlines_count}"

    SC_W = load_averaged_matrix(path_to_dir,mode,matrix_W_name)
    SC_L = load_averaged_matrix(path_to_dir,mode,matrix_L_name)
    W = None

    if SC_W is None:
        W = load_subjects_3Dmatrix_domhof(path_to_dir,"Counts.csv",n_roi)
        W_nan = np.where(W>min_streamlines_count,W,np.nan)
        SC_W = create_averaged_matrix_based_on_mode(mode,W_nan,distances)
        save_averaged_matrix(SC_W,path_to_dir,mode,matrix_W_name)

    if SC_L is None:
        if W is None:
            W = load_subjects_3Dmatrix_domhof(path_to_dir,"Counts.csv",n_roi)
        L = load_subjects_3Dmatrix_domhof(path_to_dir,"Lengths.csv",n_roi)

        L_nan = np.where(W>min_streamlines_count,L,np.nan)
        SC_L = np.nanmean(L_nan,axis=0)
        SC_L = np.where(SC_W>0,SC_L,np.nan)

        save_averaged_matrix(SC_L,path_to_dir,mode,matrix_L_name)

    return SC_W, SC_L

def load_domhof_matrix_for_pytepfit(min_streamlines_count=5,parcellation="schaefer200",mode="simple",distances=None):
    """
    Load group averaged matrix from Domhof dataset based on
    averaging method mode and reorder for compatibility with
    TMS-EEG PyTepFit data.

    Parameters:
    min_streamlines_count (int): minimal number of streamlines between two ROIs
        in one subject to assume that there is an edge connecting them
    parcellation (str): string indicating which brain parcellation we
        want to use, it is here only for compatibility and is later 
        set to schaefer200 no matter what
    mode (str): group averaging mode, select an option 
        from ["simple","cons","dist","rh"]
            - simple: simple average
            - cons: consensus thresholding
            - dist: distance-dependent consensus thresholding
            - rh: Rosen and Halgren's ageraging method
    distances (2D np.array): ROI distance matrix, i.e. Euclidean
        distances of ROIs, necessary for "dist" mode

    Returns:
    tuple(2D np.array, 2D np.array): 
        structural connectivity weights, structural connectivity lengths
    """

    SC_W, SC_L = load_domhof_matrix(parcellation,min_streamlines_count=min_streamlines_count,mode=mode,distances=distances)

    mapping_17 = path('interim/schaefer_parcellation_mappings/ROI_MAPPING_7_17.csv')
    mapping_csv = path('interim/schaefer_parcellation_mappings/ROI_MAPPING_pytepfit.csv')

    SC_W = schaefer_to_schaefer(schaefer_to_schaefer(SC_W,mapping_17,"idx_17"),mapping_csv,"idx_csv")
    SC_L = schaefer_to_schaefer(schaefer_to_schaefer(SC_L,mapping_17,"idx_17"),mapping_csv,"idx_csv")

    return SC_W, SC_L

# ===================================================================
# ============================ Mica-Mics ============================
# ===================================================================

def load_subjects_3Dmatrix_mica(path_to_dir,parcellation="glasser360",n_roi=360,number_of_subjects=50,sc_or_lengths='sc'):   
    """
    Load 3D matrix consisting of individual structural connectivity matrices
    from Mica-Mics dataset.

    Parameters:
    path_to_dir (str/path): path to directory where data for individual 
        subjects are stored
    parcellation (str): string indicating which brain parcellation we
        want to use, possible options based on downloaded data
    n_roi (int): number of ROIs, specify based on parcellation, sholud be
        the same as the dimensions of individual matrices we want to load
    number_of_subjects (int): number of subjects we want to load, usually
        the number of subject foledrs in path_to_dir folder
    sc_or_lengths: indicated if we want to load structural connectivity
        weights (sc) or lengths (edgeLength)

    Returns:
    3D numpy array (number_of_subjects,n_roi,n_roi): structural connectivity matrices
        for all subjects defined by number_of_subjects    
    """

    M = np.zeros((number_of_subjects ,n_roi,n_roi))

    # filter out subcortical regions
    index = list(range(15,15+n_roi//2)) + list(range(15+n_roi//2+1,15+n_roi+1))

    for i in range(number_of_subjects):
        counts_file = path(path_to_dir+f"/sub-HC{i+1:03d}_ses-01_space-dwinative_atlas-{parcellation}_desc-{sc_or_lengths}.txt")
        with open(counts_file,"r") as cf:
            c = np.genfromtxt(cf,delimiter=',')
            c = np.take(c,indices=index,axis=0)
            c = np.take(c,indices=index,axis=1)
            cT = c.T.copy()
            np.fill_diagonal(cT,0)
            M[i] = c + cT

    return M

def load_mica_matrix(parcellation="glasser360",min_streamlines_count=5,mode="simple",distances=None):
    """
    Load group averaged matrix from Mica-Mics dataset based on
    averaging method mode.

    Parameters:
    parcellation (str): string indicating which brain parcellation we
        want to use, possible options based on downloaded data
    min_streamlines_count (int): minimal number of streamlines between two ROIs
        in one subject to assume that there is an edge connecting them
    mode (str): group averaging mode, select an option 
        from ["simple","cons","dist","rh"]
            - simple: simple average
            - cons: consensus thresholding
            - dist: distance-dependent consensus thresholding
            - rh: Rosen and Halgren's ageraging method
    distances (2D np.array): ROI distance matrix, i.e. Euclidean
        distances of ROIs, necessary for "dist" mode

    Returns:
    tuple(2D np.array, 2D np.array): 
        structural connectivity weights, structural connectivity lengths
    """

    if parcellation == "glasser360":
        n_roi= 360
    elif parcellation == "schaefer200":
        n_roi = 200

    path_to_dir = path(f"external/mica-mics/{parcellation}/")

    matrix_W_name = f"SC_W_min{min_streamlines_count}"
    matrix_L_name = f"SC_L_min{min_streamlines_count}"

    SC_L = load_averaged_matrix(path_to_dir,mode,matrix_L_name)
    SC_W = load_averaged_matrix(path_to_dir,mode,matrix_W_name)
    W = None

    if SC_W is None:
        W = load_subjects_3Dmatrix_mica(path_to_dir,parcellation,n_roi)
        W_nan = np.where(W>min_streamlines_count,W,np.nan)
        SC_W = create_averaged_matrix_based_on_mode(mode,W_nan,distances)
        save_averaged_matrix(SC_W,path_to_dir,mode,matrix_W_name)

    if SC_L is None:
        if W is None:
            W = load_subjects_3Dmatrix_mica(path_to_dir,parcellation,n_roi)
        L = load_subjects_3Dmatrix_mica(path_to_dir,parcellation,n_roi,sc_or_lengths="edgeLength")

        L_nan = np.where(W>min_streamlines_count,L,np.nan)
        SC_L = np.nanmean(L_nan,axis=0)
        SC_L = np.where(SC_W>0,SC_L,np.nan)
        save_averaged_matrix(SC_L,path_to_dir,mode,matrix_L_name)

    return SC_W, SC_L

def load_mica_matrix_for_pytepfit(min_streamlines_count=5,parcellation="schaefer200",mode="simple",distances=None):
    """
    Load group averaged matrix from Mica-Mics dataset based on
    averaging method mode and reorder for compatibility with
    TMS-EEG PyTepFit data.

    Parameters:
    min_streamlines_count (int): minimal number of streamlines between two ROIs
        in one subject to assume that there is an edge connecting them
    parcellation (str): string indicating which brain parcellation we
        want to use, it is here only for compatibility and is later 
        set to schaefer200 no matter what
    mode (str): group averaging mode, select an option 
        from ["simple","cons","dist","rh"]
            - simple: simple average
            - cons: consensus thresholding
            - dist: distance-dependent consensus thresholding
            - rh: Rosen and Halgren's ageraging method
    distances (2D np.array): ROI distance matrix, i.e. Euclidean
        distances of ROIs, necessary for "dist" mode

    Returns:
    tuple(2D np.array, 2D np.array): 
        structural connectivity weights, structural connectivity lengths
    """
    parcellation="schaefer200"

    SC_W, SC_L = load_mica_matrix(parcellation,min_streamlines_count=min_streamlines_count,mode=mode,distances=distances)

    mapping_csv = path('interim/schaefer_parcellation_mappings/ROI_MAPPING_pytepfit.csv')

    SC_W = schaefer_to_schaefer(SC_W,mapping_csv,"idx_csv")
    SC_L = schaefer_to_schaefer(SC_L,mapping_csv,"idx_csv")

    return SC_W, SC_L

# ===================================================================
# ========================== Rosen-Halgren ==========================
# ===================================================================

def load_rosen_halgren(reference_labels=None):
    """
    Load structural connectivity matrices (weights and lengths) from PyTepFit dataset
    and eventually reorder based on reference labels
    
    Parameters:
    reference_labels (list[str]): labels for reordering

    Returns:
    tuple(2D np.array, 2D np.array)
        2D numpy array: structural connectivity matrix - weights  
        2D numpy array: structural connectivity matrix - lengths
    """

    path_sc_w = path('external/rosen_halgren/averageConnectivity_Fpt.mat')
    path_sc_l = path('external/rosen_halgren/averageConnectivity_tractLengths.mat')
    
    with h5py.File(path_sc_w, 'r') as f:
        SC_W = np.array(f.get('Fpt'))
        labels = ["".join(chr(i) for i in f[f['parcelIDs'][0][j]][:].squeeze()) for j in range(360)]

    with h5py.File(path_sc_l, 'r') as f:
        SC_L = np.array(f.get('tractLengths'))

    if reference_labels is not None:
        SC_W = reorder_matrix_based_on_reference(labels,reference_labels,SC_W)
        SC_L = reorder_matrix_based_on_reference(labels,reference_labels,SC_L)

    return SC_W, SC_L

# ===================================================================
# ========================== PyTepFit SC ============================
# ===================================================================

def load_pytepfit_sc():
    """
    Load structural connectivity matrices (weights and lengths) from PyTepFit dataset
    and reorder based on PyTepFit ROI mapping table to match TMS-EEG ROI ordering.

    Returns:
    tuple(2D np.array, 2D np.array)
        2D numpy array: structural connectivity matrix - weights  
        2D numpy array: structural connectivity matrix - lengths
    """
    pytepfit_path = 'external/pytepfit/'
    
    SC_W = np.loadtxt(path(pytepfit_path+'Schaefer2018_200Parcels_7Networks_count.csv'))
    SC_L = np.loadtxt(path(pytepfit_path+'Schaefer2018_200Parcels_7Networks_distance.csv'))

    SC_L = np.where(SC_W == 0,np.nan,SC_L)

    mapping_path = path('interim/schaefer_parcellation_mappings/ROI_MAPPING_pytepfit.csv')
    return schaefer_to_schaefer(SC_W,mapping_path,"idx_csv"), schaefer_to_schaefer(SC_L,mapping_path,"idx_csv")

# ===================================================================
# ==================== Create new structural matrices ===============
# ===================================================================

if __name__ == "__main__":
    print("DKT")
    ED = dkt_roi_distances()
    SC_matrices = load_set_of_DKT_matrices_for_ftract(ED=ED)

    print("Glasser")
    ED = glasser_roi_distances()
    SC_matrices = load_set_of_glasser_matrices_for_ftract(ED=ED)

    print("Schaefer")
    ED = schaefer_roi_distances()
    SC_matrices = load_set_of_schaefer_matrices_for_pytepfit(ED=ED)  