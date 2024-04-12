import os 
import h5py
import numpy as np
import pandas as pd
import logging
import src.roi_remappnig as roi_remappnig

try:
    from enigmatoolbox.datasets import load_sc, load_fc
except:
    logging.warning('ENIGMA toolbox not installed, respective data loading will be not available.')

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

def load_ftract(parcellation,short=False):

    if short:
        ftract_path = 'external/F-TRACT_short/'
    else:
        ftract_path = 'external/F-TRACT/'
    probability = np.loadtxt(path(f'{ftract_path}{parcellation}/probability.txt.gz'))
    amplitude = np.loadtxt(path(f'{ftract_path}{parcellation}/amplitude__median.txt.gz'))
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
        n_stim = n_stim[np.ix_(parcell_ids, parcell_ids)]
        n_impl = n_impl[np.ix_(parcell_ids, parcell_ids)]

        labels = roi

    return probability, amplitude, n_stim, n_impl, labels

def load_ftract_labels(parcellation):
    ftract_path = 'external/F-TRACT/'
    labels = np.loadtxt(path(f'{ftract_path}{parcellation}/{parcellation}.txt'), dtype=str)
    return labels


def load_enigma(parcellation=None,reoreder=False):
    if parcellation == "DKT":
        SC, _, _, _ = load_sc()
    else:
        SC, _, _, _ = load_sc(parcellation=parcellation)

    if reoreder=='PyTepFit':
        mapping_path = path('external/schaefer_parcellation_centroids/ROI_MAPPING_pytepfit.csv')
        SC = roi_remappnig.schaefer_to_schaefer(SC,mapping_path,"idx_csv")       

    return SC, None

def load_subjects_3Dmatrix_domhof(path_to_dir,csv_name,n_roi,number_of_subjects=200):
    M = np.zeros((number_of_subjects,n_roi,n_roi))
    for i in range(number_of_subjects):
        counts_file = path_to_dir+f"/{i:03d}/"+csv_name
        with open(counts_file,"r") as cf:
            c = np.genfromtxt(cf)
        M[i] = c

    return M

def load_subjects_3Dmatrix_mica(path_to_dir,parcellation="glasser360",n_roi=360,number_of_subjects=50,sc_or_lengths='sc'):    
    M = np.zeros((number_of_subjects ,n_roi,n_roi))

    # bacause of subcortical regions
    index = list(range(15,15+n_roi//2)) + list(range(15+n_roi//2+1,15+n_roi+1))

    for i in range(number_of_subjects):
        counts_file = path(path_to_dir+f"/sub-HC{i+1:03d}_ses-01_space-dwinative_atlas-{parcellation}_desc-{sc_or_lengths}.txt")
        with open(counts_file,"r") as cf:
            c = np.genfromtxt(cf,delimiter=',')
            c = np.take(c,indices=index,axis=0)
            c = np.take(c,indices=index,axis=1)
            M[i] = c + c.T

    return M

def create_averaged_matrix_based_on_mode(mode,M,distances=None):
    SC_W = None

    if mode == "mean":
        SC_W = simple_averaging(M)
    elif mode == "dist_dep_thresholding":
        if distances is None:
            print(f"Node distances necessary to calculate distance dependent thresholding!")
        else:
            M = np.transpose(M)
            SC_W = struct_consensus(M,distances,weighted=True)
    elif mode == "rh_averaging":
        SC_W = rosenhalgren_sc_averaging(M)
    else:
        print("Invalid mode!")

    return SC_W
            

def load_domhof(parcellation,n_roi,mode="mean",distances=None):
    rootdir = f"external/domhof/{parcellation}/"
    scdir = "1StructuralConnectivity/"
    sc_path_to_dir = path(rootdir+scdir)

    SC_L = load_averaged_matrix(sc_path_to_dir,"mean","SC_L")
    SC_W = load_averaged_matrix(sc_path_to_dir,mode,"SC_W")
    W = None

    if SC_W is None:
        M = load_subjects_3Dmatrix_domhof(sc_path_to_dir,"Counts.csv",n_roi)
        SC_W = create_averaged_matrix_based_on_mode(mode,M,distances)
        save_averaged_matrix(SC_W,sc_path_to_dir,mode,"SC_W")

    if SC_L is None:
        W = load_subjects_3Dmatrix_domhof(sc_path_to_dir,"Counts.csv",n_roi)
        L = load_subjects_3Dmatrix_domhof(sc_path_to_dir,"Lengths.csv",n_roi)

        L_nan = np.where(W>0,L,np.nan)
        SC_L = np.nanmean(L_nan,axis=0)
        save_averaged_matrix(SC_L,sc_path_to_dir,"mean","SC_L")

    return SC_W, SC_L


def load_mica_for_pytepfit(mode="mean"):
    parcellation = "schaefer200"
    n_roi = 200

    SC_W, SC_L = load_mica(parcellation,n_roi,mode=mode)

    mapping_csv = path('external/schaefer_parcellation_centroids/ROI_MAPPING_pytepfit.csv')

    SC_W = roi_remappnig.schaefer_to_schaefer(SC_W,mapping_csv,"idx_csv")
    SC_L = roi_remappnig.schaefer_to_schaefer(SC_L,mapping_csv,"idx_csv")

    return SC_W, SC_L

def load_domhof_for_pytepfit(mode="mean"):
    parcellation = "schaefer200"
    n_roi = 200

    SC_W, SC_L = load_domhof(parcellation,n_roi,mode=mode)

    mapping_17 = path('external/schaefer_parcellation_centroids/ROI_MAPPING_7_17.csv')
    mapping_csv = path('external/schaefer_parcellation_centroids/ROI_MAPPING_pytepfit.csv')

    SC_W = roi_remappnig.schaefer_to_schaefer(roi_remappnig.schaefer_to_schaefer(SC_W,mapping_17,"idx_17"),mapping_csv,"idx_csv")
    SC_L = roi_remappnig.schaefer_to_schaefer(roi_remappnig.schaefer_to_schaefer(SC_L,mapping_17,"idx_17"),mapping_csv,"idx_csv")

    return SC_W, SC_L

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

def load_mica(parcellation,n_roi,mode="mean",distances=None):
    path_to_dir = path(f"external/mica-mics/{parcellation}/")

    SC_L = load_averaged_matrix(path_to_dir,"mean","SC_L") # TODO mean nonzero
    SC_W = load_averaged_matrix(path_to_dir,mode,"SC_W")
    W = None

    if SC_W is None:
        W = load_subjects_3Dmatrix_mica(path_to_dir,parcellation,n_roi)
        SC_W = create_averaged_matrix_based_on_mode(mode,W,distances)
        save_averaged_matrix(SC_W,path_to_dir,mode,"SC_W")

    if SC_L is None:
        if W is None:
            W = load_subjects_3Dmatrix_mica(path_to_dir,parcellation,n_roi)
        L = load_subjects_3Dmatrix_mica(path_to_dir,parcellation,n_roi,sc_or_lengths="edgeLength")

        L_nan = np.where(W>0,L,np.nan)
        SC_L = np.nanmean(L_nan,axis=0)
        save_averaged_matrix(SC_L,path_to_dir,"mean","SC_L")

    return SC_W, SC_L

def load_rosen_halgren(reference_labels=None):
    path_sc_w = path('external/rosen_halgren/averageConnectivity_Fpt.mat')
    path_sc_l = path('external/rosen_halgren/averageConnectivity_tractLengths.mat')
    
    with h5py.File(path_sc_w, 'r') as f:
        SC_W = np.array(f.get('Fpt'))
        labels = ["".join(chr(i) for i in f[f['parcelIDs'][0][j]][:].squeeze()) for j in range(360)]

    with h5py.File(path_sc_l, 'r') as f:
        SC_L = np.array(f.get('tractLengths'))

    if reference_labels is not None:
        SC_W = roi_remappnig.reorder_matrix_based_on_reference(labels,reference_labels,SC_W)
        SC_L = roi_remappnig.reorder_matrix_based_on_reference(labels,reference_labels,SC_L)

    return SC_W, SC_L

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

def load_glasser_centroids(ftract_labels=None):
    centroids_file = pd.read_csv(path('external/glasser_parcellation_centriods/HCP-MMP1_UniqueRegionList.csv'))   
    
    if ftract_labels is not None:
        labels = ftract_compatible_glasser_labels()
        centroids_file['ftract_labels'] = labels
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

def load_pytepfit_sc():
    pytepfit_path = 'external/pytepfit/'
    
    SC_W = np.loadtxt(path(pytepfit_path+'Schaefer2018_200Parcels_7Networks_count.csv'))
    SC_L = np.loadtxt(path(pytepfit_path+'Schaefer2018_200Parcels_7Networks_distance.csv'))

    SC_L = np.where(SC_W == 0,np.nan,SC_L)

    mapping_path = path('external/schaefer_parcellation_centroids/ROI_MAPPING_pytepfit.csv')
    return roi_remappnig.schaefer_to_schaefer(SC_W,mapping_path,"idx_csv"), roi_remappnig.schaefer_to_schaefer(SC_L,mapping_path,"idx_csv")
                   
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

def get_labels_from_file(centroids_file,label_column):
    df = pd.read_csv(centroids_file)
    return df[label_column]

def rosenhalgren_sc_averaging(M):
    M_mean = np.mean(M,axis=0)
    SC = np.zeros(M_mean.shape)

    for i in range(M_mean.shape[0]):
        for j in range(M_mean.shape[1]):
            SC[i][j] = M_mean[i][j] / (np.sum(M_mean[i,:]) + np.sum(M_mean[:,j]) - M_mean[i][i] - M_mean[j][j])

    np.fill_diagonal(SC,np.nan)
    return SC

def simple_averaging(M):
    return np.mean(M,axis=0)

def keep_val_where_weight(SC_W,SC_L):
    np.fill_diagonal(SC_W,np.nan) # do not consider self-loops
    SC_W = np.where(SC_W==0,np.nan, SC_W) # if weight is 0, there is no connection
    SC_L = np.where(np.isnan(SC_W),np.nan,SC_L) # keep lengths only for edges where is connection (based on weights)
    np.nan_to_num(SC_W, copy=False) # convert nan to 0, because the metrics can not handle nans
    return SC_W,SC_L

def load_set_of_glasser_matrices_for_ftract(ftract_labels,ED=None):
    labels = ftract_compatible_glasser_labels()

    SC_matrices_mica = []

    SC_W_M, SC_L_M = load_mica("glasser360",360,mode="rh_averaging")
    SC_matrices_mica.append(("Mica-Mics_rh",SC_W_M, SC_L_M,np.log(SC_W_M)))

    SC_W_M_mean, SC_L_M_mean = load_mica("glasser360",360,mode="mean")
    SC_matrices_mica.append(("Mica-Mics_simple",SC_W_M_mean, SC_L_M_mean,np.log(SC_W_M_mean)))

    if ED is not None:  
        SC_W_M_dist, SC_L_M_dist = load_mica("glasser360",360,mode="dist_dep_thresholding",distances=ED)
        SC_matrices_mica.append(("Mica-Mics_dist",SC_W_M_dist, SC_L_M_dist,np.log(SC_W_M_dist)))

    # SC_W_E, _ = load_enigma(parcellation="glasser_360")
    # SC_W_E = roi_remappnig.reorder_matrix_based_on_reference(labels,ftract_labels,SC_W_E)
    # SC_matrices.append(("Enigma",np.exp(SC_W_E), None,SC_W_E))

    SC_matrices = []

    for name, SC_W, SC_L, SC_W_log in SC_matrices_mica:
        SC_W, SC_L  = keep_val_where_weight(SC_W, SC_L)
        SC_W = roi_remappnig.reorder_matrix_based_on_reference(labels,ftract_labels,SC_W)
        SC_L = roi_remappnig.reorder_matrix_based_on_reference(labels,ftract_labels,SC_L)
        SC_W_log = roi_remappnig.reorder_matrix_based_on_reference(labels,ftract_labels,SC_W_log)
        SC_matrices.append((name, SC_W, SC_L, SC_W_log))


    SC_W_RH_log, SC_L_RH = load_rosen_halgren(ftract_labels)
    SC_W_RH_log, SC_L_RH = keep_val_where_weight(SC_W_RH_log, SC_L_RH)
    SC_matrices.append(("Rosen-Halgren",10**SC_W_RH_log, SC_L_RH,SC_W_RH_log))

    return SC_matrices

def load_set_of_schaefer_matrices_for_pytepfit(ED=None):
    SC_matrices = []

    SC_W_pytep, SC_L_pytep = load_pytepfit_sc()
    SC_matrices.append(("PyTepFit",SC_W_pytep, SC_L_pytep,np.log(SC_W_pytep)))

    SC_W_ENIGMA, _ = load_enigma(parcellation="schaefer_200",reoreder='PyTepFit')
    # SC_W_ENIGMA = np.where(SC_W_ENIGMA==1,np.nan,SC_W_ENIGMA) # proč tam skra mají všude 1???
    SC_matrices.append(("Enigma",np.exp(SC_W_ENIGMA), None,SC_W_ENIGMA,))

    SC_W_dom, SC_L_dom = load_domhof_for_pytepfit(mode="rh_averaging")
    SC_matrices.append(("Domhof_rh",SC_W_dom, SC_L_dom,np.log(SC_W_dom)))

    SC_W_mica, SC_L_mica = load_mica_for_pytepfit(mode="rh_averaging")
    SC_matrices.append(("Mica-Mics_rh",SC_W_mica, SC_L_mica,np.log(SC_W_mica)))

    #if ED is not None:
    #    SC_W_M_dist, SC_L_M_dist, _ = load_domhof_for_pytepfit(mode="dist_dep_thresholding",ED=ED)
    #    SC_matrices.append(("Mica-Mics_dist",SC_W_M_dist, SC_L_M_dist,np.log(SC_W_M_dist)))

    return SC_matrices

def load_set_of_DKT_matrices_for_ftract(ftract_labels,ids_to_delete_in_dkt):
    SC_matrices = []

    SC_W_E, _ = load_enigma(parcellation="DKT")
    SC_matrices.append(("Enigma",SC_W_E, None,"log"))

    SC_W_D, SC_L_D = load_domhof("DKT",len(ftract_labels))
    SC_W_D, SC_L_D  = keep_val_where_weight(SC_W_D, SC_L_D)
    for a in [0,1]:
        SC_W_D = np.delete(SC_W_D,ids_to_delete_in_dkt,axis=a)
        SC_L_D = np.delete(SC_L_D,ids_to_delete_in_dkt,axis=a)

    SC_matrices.append(("Domhof_simple",SC_W_D, SC_L_D,None))

    SC_W_D_rh, SC_L_D_rh = load_domhof("DKT",len(ftract_labels),mode="rh_averaging")
    SC_W_D_rh, SC_L_D_rh = keep_val_where_weight(SC_W_D_rh, SC_L_D_rh)
    for a in [0,1]:
        SC_W_D_rh = np.delete(SC_W_D_rh,ids_to_delete_in_dkt,axis=a)
        SC_L_D_rh = np.delete(SC_L_D_rh,ids_to_delete_in_dkt,axis=a)

    SC_matrices.append(("Domhof_rh",SC_W_D_rh, SC_L_D_rh,None))

    return SC_matrices

def find_pivot_to_keep_xpercent_edges(matrix,n_roi=200,percent=0.85):
    pivot_id = int((n_roi**2)*percent)
    matrix_flat_sorted = np.sort(np.nan_to_num(matrix.flatten()))
    return matrix_flat_sorted[pivot_id]

def find_pivot_to_keep_x_edges(matrix,x):
    matrix_flat_sorted = np.flip(np.sort(np.nan_to_num(matrix.flatten())))
    return matrix_flat_sorted[x]

# from netneurotools, added hemiid creation

def _ecdf(data):
    """
    Estimate empirical cumulative distribution function of `data`.

    Taken directly from StackOverflow. See original answer at
    https://stackoverflow.com/questions/33345780.

    Parameters
    ----------
    data : array_like

    Returns
    -------
    prob : numpy.ndarray
        Cumulative probability
    quantiles : numpy.darray
        Quantiles
    """
    sample = np.atleast_1d(data)

    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    prob = np.cumsum(counts).astype(float) / sample.size

    # match MATLAB
    prob, quantiles = np.append([0], prob), np.append(quantiles[0], quantiles)

    return prob, quantiles

def struct_consensus(data, distance, hemiid=None,
                     conn_num_inter=None,
                     conn_num_intra=None,
                     weighted=False):
    """
    Calculate distance-dependent group consensus structural connectivity graph.

    Takes as input a weighted stack of connectivity matrices with dimensions
    (N, N, S) where `N` is the number of nodes and `S` is the number of
    matrices or subjects. The matrices must be weighted, and ideally with
    continuous weights (e.g. fractional anisotropy rather than streamline
    count). The second input is a pairwise distance matrix, where distance(i,j)
    is the Euclidean distance between nodes i and j. The final input is an
    (N, 1) vector which labels nodes as belonging to the right (`hemiid==0`) or
    left (`hemiid=1`) hemisphere (note that these values can be flipped as long
    as `hemiid` contains only values of 0 and 1).

    This function estimates the average edge length distribution and builds
    a group-averaged connectivity matrix that approximates this distribution
    with density equal to the mean density across subjects.

    The algorithm works as follows:

    1. Estimate the cumulative edge length distribution,
    2. Divide the distribution into M length bins, one for each edge that will
       be added to the group-average matrix, and
    3. Within each bin, select the edge that is most consistently expressed
       expressed across subjects, breaking ties according to average edge
       weight (which is why the input matrix `data` must be weighted).

    The algorithm works separately on within/between hemisphere links.
    M is the sum of `conn_num_inter` and `conn_num_intra`, if provided.
    Otherwise, M is estimated from the data.

    Parameters
    ----------
    data : (N, N, S) array_like
        Weighted connectivity matrices (i.e., fractional anisotropy), where `N`
        is nodes and `S` is subjects
    distance : (N, N) array_like
        Array where `distance[i, j]` is the Euclidean distance between nodes
        `i` and `j`
    hemiid : (N, 1) array_like
        Hemisphere designation for `N` nodes where a value of 0/1 indicates
        node `N_{i}` is in the right/left hemisphere, respectively
    conn_num_inter : int, optional
        Number of inter-hemispheric connections to include in the consensus
        matrix. If `None`, the number of inter-hemispheric connections will be
        estimated from the data. Default = `None`.
    conn_num_intra : int, optional
        Number of intra-hemispheric connections to include in the consensus
        matrix. If `None`, the number of intra-hemispheric connections will be
        estimated from the data. Default = `None`.
    weighted : bool
        Flag indicating whether or not to return a weighted consensus map. If
        `True`, the consensus will be multiplied by the mean of `data`.

    Returns
    -------
    consensus : (N, N) numpy.ndarray
        Binary (default) or mean-weighted group-level connectivity matrix

    References
    ----------
    Betzel, R. F., Griffa, A., Hagmann, P., & Mišić, B. (2018). Distance-
    dependent consensus thresholds for generating group-representative
    structural brain networks. Network Neuroscience, 1-22.
    """

    num_node, _, num_sub = data.shape      # info on connectivity matrices
    pos_data = data > 0                    # location of + values in matrix
    pos_data_count = pos_data.sum(axis=2)  # num sub with + values at each node

    if hemiid is None:
        hemiid =  np.array([0] * (num_node//2) + [1] * (num_node//2)).reshape(-1, 1)

    with np.errstate(divide='ignore', invalid='ignore'):
        average_weights = data.sum(axis=2) / pos_data_count

    # empty array to hold inter/intra hemispheric connections
    consensus = np.zeros((num_node, num_node, 2))

    for conn_type in range(2):  # iterate through inter/intra hemisphere conn
        if conn_type == 0:      # get inter hemisphere edges
            inter_hemi = (hemiid == 0) @ (hemiid == 1).T
            keep_conn = np.logical_or(inter_hemi, inter_hemi.T)
        else:                   # get intra hemisphere edges
            right_hemi = (hemiid == 0) @ (hemiid == 0).T
            left_hemi = (hemiid == 1) @ (hemiid == 1).T
            keep_conn = np.logical_or(right_hemi @ right_hemi.T,
                                      left_hemi @ left_hemi.T)

        # mask the distance array for only those edges we want to examine
        full_dist_conn = distance * keep_conn
        upper_dist_conn = np.atleast_3d(np.triu(full_dist_conn))

        # generate array of weighted (by distance), positive edges across subs
        pos_dist = pos_data * upper_dist_conn
        pos_dist = pos_dist[np.nonzero(pos_dist)]

        # determine average # of positive edges across subs
        # we will use this to bin the edge weights
        if conn_type == 0:
            if conn_num_inter is None:
                avg_conn_num = len(pos_dist) / num_sub
            else:
                avg_conn_num = conn_num_inter
        else:
            if conn_num_intra is None:
                avg_conn_num = len(pos_dist) / num_sub
            else:
                avg_conn_num = conn_num_intra

        # estimate empirical CDF of weighted, positive edges across subs
        cumprob, quantiles = _ecdf(pos_dist)
        cumprob = np.round(cumprob * avg_conn_num).astype(int)

        # empty array to hold group-average matrix for current connection type
        # (i.e., inter/intra hemispheric connections)
        group_conn_type = np.zeros((num_node, num_node))

        # iterate through bins (for edge weights)
        for n in range(1, int(avg_conn_num) + 1):
            # get current quantile of interest
            curr_quant = quantiles[np.logical_and(cumprob >= (n - 1),
                                                  cumprob < n)]
            if curr_quant.size == 0:
                continue

            # find edges in distance connectivity matrix w/i current quantile
            mask = np.logical_and(full_dist_conn >= curr_quant.min(),
                                  full_dist_conn <= curr_quant.max())
            i, j = np.where(np.triu(mask))  # indices of edges of interest

            c = pos_data_count[i, j]   # get num sub with + values at edges
            w = average_weights[i, j]  # get averaged weight of edges

            # find locations of edges most commonly represented across subs
            indmax = np.argwhere(c == c.max())

            # determine index of most frequent edge; break ties with higher
            # weighted edge
            if indmax.size == 1:  # only one edge found
                group_conn_type[i[indmax], j[indmax]] = 1
            else:                 # multiple edges found
                indmax = indmax[np.argmax(w[indmax])]
                group_conn_type[i[indmax], j[indmax]] = 1

        consensus[:, :, conn_type] = group_conn_type

    # collapse across hemispheric connections types and make symmetrical array
    consensus = consensus.sum(axis=2)
    consensus = np.logical_or(consensus, consensus.T).astype(int)

    if weighted:
        consensus = consensus * np.mean(data, axis=2)
    return consensus
