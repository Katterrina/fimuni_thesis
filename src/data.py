import os 
import h5py
import numpy as np
import pandas as pd

from enigmatoolbox.datasets import load_sc, load_fc

data_root = os.path.abspath(
        os.path.join(
            os.path.dirname(
                os.path.dirname(__file__)
            ),
            'data'
        )
)


def path(relp):
    return os.path.join(data_root, os.path.normpath(relp))

def load_ftract(parcellation):
    # TODO zkusit získat data přímo z f-tract.eu - zatím jsem je tam nenašla, ale v článku se píše, že by měly být
    probability = np.loadtxt(f'../data/external/F-TRACT/{parcellation}/probability.txt.gz')
    amplitude = np.loadtxt(f'../data/external/F-TRACT/{parcellation}/amplitude__median.txt.gz')
    n_stim = np.loadtxt(f'../data/external/F-TRACT/{parcellation}/N_stimulations.txt.gz')
    n_impl = np.loadtxt(f'../data/external/F-TRACT/{parcellation}/N_implantations.txt.gz')

    labels = np.loadtxt(f'../data/external/F-TRACT/{parcellation}/{parcellation}.txt', dtype=str)

    if parcellation == "DKT" or parcellation == "Destrieux":
        if parcellation == "DKT":
            roi = ['ctx-lh-bankssts','ctx-lh-caudalanteriorcingulate','ctx-lh-caudalmiddlefrontal','ctx-lh-corpuscallosum','ctx-lh-cuneus','ctx-lh-entorhinal','ctx-lh-fusiform','ctx-lh-inferiorparietal','ctx-lh-inferiortemporal','ctx-lh-isthmuscingulate','ctx-lh-lateraloccipital','ctx-lh-lateralorbitofrontal','ctx-lh-lingual','ctx-lh-medialorbitofrontal','ctx-lh-middletemporal','ctx-lh-parahippocampal','ctx-lh-paracentral','ctx-lh-parsopercularis','ctx-lh-parsorbitalis','ctx-lh-parstriangularis','ctx-lh-pericalcarine','ctx-lh-postcentral','ctx-lh-posteriorcingulate','ctx-lh-precentral','ctx-lh-precuneus','ctx-lh-rostralanteriorcingulate','ctx-lh-rostralmiddlefrontal','ctx-lh-superiorfrontal','ctx-lh-superiorparietal','ctx-lh-superiortemporal','ctx-lh-supramarginal','ctx-lh-frontalpole','ctx-lh-temporalpole','ctx-lh-transversetemporal','ctx-lh-insula','ctx-rh-bankssts','ctx-rh-caudalanteriorcingulate','ctx-rh-caudalmiddlefrontal','ctx-rh-corpuscallosum','ctx-rh-cuneus','ctx-rh-entorhinal','ctx-rh-fusiform','ctx-rh-inferiorparietal','ctx-rh-inferiortemporal','ctx-rh-isthmuscingulate','ctx-rh-lateraloccipital','ctx-rh-lateralorbitofrontal','ctx-rh-lingual','ctx-rh-medialorbitofrontal','ctx-rh-middletemporal','ctx-rh-parahippocampal','ctx-rh-paracentral','ctx-rh-parsopercularis','ctx-rh-parsorbitalis','ctx-rh-parstriangularis','ctx-rh-pericalcarine','ctx-rh-postcentral','ctx-rh-posteriorcingulate','ctx-rh-precentral','ctx-rh-precuneus','ctx-rh-rostralanteriorcingulate','ctx-rh-rostralmiddlefrontal','ctx-rh-superiorfrontal','ctx-rh-superiorparietal','ctx-rh-superiortemporal','ctx-rh-supramarginal','ctx-rh-frontalpole','ctx-rh-temporalpole','ctx-rh-transversetemporal','ctx-rh-insula']
        else:
            roi = ['ctx_lh_G_and_S_frontomargin','ctx_lh_G_and_S_occipital_inf','ctx_lh_G_and_S_paracentral','ctx_lh_G_and_S_subcentral','ctx_lh_G_and_S_transv_frontopol','ctx_lh_G_and_S_cingul-Ant','ctx_lh_G_and_S_cingul-Mid-Ant','ctx_lh_G_and_S_cingul-Mid-Post','ctx_lh_G_cingul-Post-dorsal','ctx_lh_G_cingul-Post-ventral','ctx_lh_G_cuneus','ctx_lh_G_front_inf-Opercular','ctx_lh_G_front_inf-Orbital','ctx_lh_G_front_inf-Triangul','ctx_lh_G_front_middle','ctx_lh_G_front_sup','ctx_lh_G_Ins_lg_and_S_cent_ins','ctx_lh_G_insular_short','ctx_lh_G_occipital_middle','ctx_lh_G_occipital_sup','ctx_lh_G_oc-temp_lat-fusifor','ctx_lh_G_oc-temp_med-Lingual','ctx_lh_G_oc-temp_med-Parahip','ctx_lh_G_orbital','ctx_lh_G_pariet_inf-Angular','ctx_lh_G_pariet_inf-Supramar','ctx_lh_G_parietal_sup','ctx_lh_G_postcentral','ctx_lh_G_precentral','ctx_lh_G_precuneus','ctx_lh_G_rectus','ctx_lh_G_subcallosal','ctx_lh_G_temp_sup-G_T_transv','ctx_lh_G_temp_sup-Lateral','ctx_lh_G_temp_sup-Plan_polar','ctx_lh_G_temp_sup-Plan_tempo','ctx_lh_G_temporal_inf','ctx_lh_G_temporal_middle','ctx_lh_Lat_Fis-ant-Horizont','ctx_lh_Lat_Fis-ant-Vertical','ctx_lh_Lat_Fis-post','ctx_lh_Medial_wall','ctx_lh_Pole_occipital','ctx_lh_Pole_temporal','ctx_lh_S_calcarine','ctx_lh_S_central','ctx_lh_S_cingul-Marginalis','ctx_lh_S_circular_insula_ant','ctx_lh_S_circular_insula_inf','ctx_lh_S_circular_insula_sup','ctx_lh_S_collat_transv_ant','ctx_lh_S_collat_transv_post','ctx_lh_S_front_inf','ctx_lh_S_front_middle','ctx_lh_S_front_sup','ctx_lh_S_interm_prim-Jensen','ctx_lh_S_intrapariet_and_P_trans','ctx_lh_S_oc_middle_and_Lunatus','ctx_lh_S_oc_sup_and_transversal','ctx_lh_S_occipital_ant','ctx_lh_S_oc-temp_lat','ctx_lh_S_oc-temp_med_and_Lingual','ctx_lh_S_orbital_lateral','ctx_lh_S_orbital_med-olfact','ctx_lh_S_orbital-H_Shaped','ctx_lh_S_parieto_occipital','ctx_lh_S_pericallosal','ctx_lh_S_postcentral','ctx_lh_S_precentral-inf-part','ctx_lh_S_precentral-sup-part','ctx_lh_S_suborbital','ctx_lh_S_subparietal','ctx_lh_S_temporal_inf','ctx_lh_S_temporal_sup','ctx_lh_S_temporal_transverse','ctx_rh_G_and_S_frontomargin','ctx_rh_G_and_S_occipital_inf','ctx_rh_G_and_S_paracentral','ctx_rh_G_and_S_subcentral','ctx_rh_G_and_S_transv_frontopol','ctx_rh_G_and_S_cingul-Ant','ctx_rh_G_and_S_cingul-Mid-Ant','ctx_rh_G_and_S_cingul-Mid-Post','ctx_rh_G_cingul-Post-dorsal','ctx_rh_G_cingul-Post-ventral','ctx_rh_G_cuneus','ctx_rh_G_front_inf-Opercular','ctx_rh_G_front_inf-Orbital','ctx_rh_G_front_inf-Triangul','ctx_rh_G_front_middle','ctx_rh_G_front_sup','ctx_rh_G_Ins_lg_and_S_cent_ins','ctx_rh_G_insular_short','ctx_rh_G_occipital_middle','ctx_rh_G_occipital_sup','ctx_rh_G_oc-temp_lat-fusifor','ctx_rh_G_oc-temp_med-Lingual','ctx_rh_G_oc-temp_med-Parahip','ctx_rh_G_orbital','ctx_rh_G_pariet_inf-Angular','ctx_rh_G_pariet_inf-Supramar','ctx_rh_G_parietal_sup','ctx_rh_G_postcentral','ctx_rh_G_precentral','ctx_rh_G_precuneus','ctx_rh_G_rectus','ctx_rh_G_subcallosal','ctx_rh_G_temp_sup-G_T_transv','ctx_rh_G_temp_sup-Lateral','ctx_rh_G_temp_sup-Plan_polar','ctx_rh_G_temp_sup-Plan_tempo','ctx_rh_G_temporal_inf','ctx_rh_G_temporal_middle','ctx_rh_Lat_Fis-ant-Horizont','ctx_rh_Lat_Fis-ant-Vertical','ctx_rh_Lat_Fis-post','ctx_rh_Medial_wall','ctx_rh_Pole_occipital','ctx_rh_Pole_temporal','ctx_rh_S_calcarine','ctx_rh_S_central','ctx_rh_S_cingul-Marginalis','ctx_rh_S_circular_insula_ant','ctx_rh_S_circular_insula_inf','ctx_rh_S_circular_insula_sup','ctx_rh_S_collat_transv_ant','ctx_rh_S_collat_transv_post','ctx_rh_S_front_inf','ctx_rh_S_front_middle','ctx_rh_S_front_sup','ctx_rh_S_interm_prim-Jensen','ctx_rh_S_intrapariet_and_P_trans','ctx_rh_S_oc_middle_and_Lunatus','ctx_rh_S_oc_sup_and_transversal','ctx_rh_S_occipital_ant','ctx_rh_S_oc-temp_lat','ctx_rh_S_oc-temp_med_and_Lingual','ctx_rh_S_orbital_lateral','ctx_rh_S_orbital_med-olfact','ctx_rh_S_orbital-H_Shaped','ctx_rh_S_parieto_occipital','ctx_rh_S_pericallosal','ctx_rh_S_postcentral','ctx_rh_S_precentral-inf-part','ctx_rh_S_precentral-sup-part','ctx_rh_S_suborbital','ctx_rh_S_subparietal','ctx_rh_S_temporal_inf','ctx_rh_S_temporal_sup','ctx_rh_S_temporal_transverse'] 

        parcell_ids =  [i for i,label in enumerate(labels) if label in roi]

        probability = probability[np.ix_(parcell_ids, parcell_ids)]
        amplitude  = amplitude[np.ix_(parcell_ids, parcell_ids)]
        n_stim = n_stim[np.ix_(parcell_ids, parcell_ids)]
        n_impl = n_impl[np.ix_(parcell_ids, parcell_ids)]

        labels = roi

    return probability, amplitude, n_stim, n_impl, labels

import matplotlib.pyplot as plt

def load_enigma():
    SC, _, _, _ = load_sc()
    FC, _, _, _ = load_fc()

    SC_W = SC
    SC_L = np.where(SC==0.0,0,1/SC)

    return SC_W, SC_L, FC

def load_or_create_mean_matrix(path_to_dir,mean_matrix_name,csv_name,matrix_size,number_of_subjects=200):
    mean_file_path = path_to_dir + mean_matrix_name

    if os.path.isfile(mean_file_path):
        with open(mean_file_path,"rb") as f:
            M = np.load(f)
    else:
        M = np.zeros(matrix_size)
        for i in range(number_of_subjects):
            counts_file = path_to_dir+f"{i:03d}/"+csv_name
            with open(counts_file,"r") as cf:
                c = np.genfromtxt(cf)
            M += c/number_of_subjects
        with open(mean_file_path,"xb") as f:
            np.save(f,M)

    return M

def load_domhof(parcellation,n_roi):
    rootdir = f"../data/external/domhof/{parcellation}/"
    scdir = "1StructuralConnectivity/"
    fcdir = "2FunctionalConnectivity/"
    matrix_size = (n_roi,n_roi)

    SC_W = load_or_create_mean_matrix(rootdir+scdir,"SC_W.npy","Counts.csv",matrix_size)
    SC_L = load_or_create_mean_matrix(rootdir+scdir,"SC_L.npy","Lengths.csv",matrix_size)
    FC = load_or_create_mean_matrix(rootdir+fcdir,"FC.npy","EmpCorrFC_concatenated.csv",matrix_size)

    return SC_W, SC_L, FC

def reorder_matrix_based_on_reference(labels, reference_labels, matrix):
    df_matrix = pd.DataFrame(matrix, index=labels, columns=labels)
    df_matrix = df_matrix.loc[reference_labels,reference_labels]

    return df_matrix.to_numpy()

def load_rosen_halgren(reference_labels):
    path_sc_w = '../data/external/rosen_halgren_sc/public_PLoSBio_final/averageConnectivity_axonCount.mat'
    with h5py.File(path_sc_w, 'r') as f:
        SC_W = np.array(f.get('axonCount'))

    labels = np.loadtxt('../data/external/rosen_halgren_sc/public_PLoSBio_final/roi.txt', dtype=str)

    SC_W = reorder_matrix_based_on_reference(labels,reference_labels,SC_W)
    SC_L = np.where(SC_W==0.0,0,1/SC_W)

    return SC_W, SC_L, None

def glasser_roi_distances(n_roi,reference_labels):
    distances = np.zeros((n_roi,n_roi))

    surf = h5py.File('../data/external/rosen_halgren_sc/public_PLoSBio_final/fsaverage_ico7_pial_surf.mat', 'r')    
    annot = h5py.File('../data/external/rosen_halgren_sc/public_PLoSBio_final/HCP-MMP1.0_fsaverage_annot.mat', 'r')
    
    centroids = np.zeros((n_roi,3))
    labels = []

    for h,hemisphere in [(0,'lh'),(1,'rh')]:
        for i in range(n_roi//2):
            coor = surf[f'srf/{hemisphere}/vertices'][:][:,annot[f'annot/lIdx/{hemisphere}'][:].squeeze()==i+2] 
                # +1 protože číslováno od 1, +1 protože první je '_MedialWall' a to nechci
            labels.append("".join(chr(j) for j in annot[annot[f'annot/labs/{hemisphere}'][0][i+1]][:].squeeze()))
                # +1 protože první je '_MedialWall'
            
            centroids[i+h*n_roi//2] = np.mean(coor,axis=1) 

    for i in range(n_roi):
        for j in range(n_roi):
            distances[i,j] = np.linalg.norm(centroids[i]-centroids[j])

    return reorder_matrix_based_on_reference(labels,reference_labels,distances)

def load_pytepfit_sc():
    SC_W = np.loadtxt('../data/external/pytepfit/Schaefer2018_200Parcels_7Networks_count.csv')
    SC_L = np.loadtxt('../data/external/pytepfit/Schaefer2018_200Parcels_7Networks_distance.csv')

    return SC_W, SC_L, None
                   
def schaefer_roi_distances(n_roi):
    distances = np.zeros((n_roi,n_roi))

    df = pd.read_csv('../data/external/pytepfit/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv')

    centroids = np.zeros((n_roi,3))
    centroids[:,0] = df.R
    centroids[:,1] = df.A
    centroids[:,2] = df.S

    for i in range(n_roi):
        for j in range(n_roi):
            distances[i,j] = np.linalg.norm(centroids[i]-centroids[j])

    return distances