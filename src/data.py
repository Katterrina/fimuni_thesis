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

def load_ftract(parcellation,DKT_68=False):
    probability = np.loadtxt(f'../data/external/F-TRACT/{parcellation}/probability.txt.gz')
    amplitude = np.loadtxt(f'../data/external/F-TRACT/{parcellation}/amplitude__median.txt.gz')
    n_stim = np.loadtxt(f'../data/external/F-TRACT/{parcellation}/N_stimulations.txt.gz')
    n_impl = np.loadtxt(f'../data/external/F-TRACT/{parcellation}/N_implantations.txt.gz')

    labels = np.loadtxt(f'../data/external/F-TRACT/{parcellation}/{parcellation}.txt', dtype=str)

    if parcellation == "DKT":
        roi = ['ctx-lh-bankssts','ctx-lh-caudalanteriorcingulate','ctx-lh-caudalmiddlefrontal','ctx-lh-corpuscallosum','ctx-lh-cuneus','ctx-lh-entorhinal','ctx-lh-fusiform','ctx-lh-inferiorparietal','ctx-lh-inferiortemporal','ctx-lh-isthmuscingulate','ctx-lh-lateraloccipital','ctx-lh-lateralorbitofrontal','ctx-lh-lingual','ctx-lh-medialorbitofrontal','ctx-lh-middletemporal','ctx-lh-parahippocampal','ctx-lh-paracentral','ctx-lh-parsopercularis','ctx-lh-parsorbitalis','ctx-lh-parstriangularis','ctx-lh-pericalcarine','ctx-lh-postcentral','ctx-lh-posteriorcingulate','ctx-lh-precentral','ctx-lh-precuneus','ctx-lh-rostralanteriorcingulate','ctx-lh-rostralmiddlefrontal','ctx-lh-superiorfrontal','ctx-lh-superiorparietal','ctx-lh-superiortemporal','ctx-lh-supramarginal','ctx-lh-frontalpole','ctx-lh-temporalpole','ctx-lh-transversetemporal','ctx-lh-insula','ctx-rh-bankssts','ctx-rh-caudalanteriorcingulate','ctx-rh-caudalmiddlefrontal','ctx-rh-corpuscallosum','ctx-rh-cuneus','ctx-rh-entorhinal','ctx-rh-fusiform','ctx-rh-inferiorparietal','ctx-rh-inferiortemporal','ctx-rh-isthmuscingulate','ctx-rh-lateraloccipital','ctx-rh-lateralorbitofrontal','ctx-rh-lingual','ctx-rh-medialorbitofrontal','ctx-rh-middletemporal','ctx-rh-parahippocampal','ctx-rh-paracentral','ctx-rh-parsopercularis','ctx-rh-parsorbitalis','ctx-rh-parstriangularis','ctx-rh-pericalcarine','ctx-rh-postcentral','ctx-rh-posteriorcingulate','ctx-rh-precentral','ctx-rh-precuneus','ctx-rh-rostralanteriorcingulate','ctx-rh-rostralmiddlefrontal','ctx-rh-superiorfrontal','ctx-rh-superiorparietal','ctx-rh-superiortemporal','ctx-rh-supramarginal','ctx-rh-frontalpole','ctx-rh-temporalpole','ctx-rh-transversetemporal','ctx-rh-insula']

        parcell_ids =  [i for i,label in enumerate(labels) if label in roi]

        probability = probability[np.ix_(parcell_ids, parcell_ids)]
        amplitude  = amplitude[np.ix_(parcell_ids, parcell_ids)]
        n_stim = n_stim[np.ix_(parcell_ids, parcell_ids)]
        n_impl = n_impl[np.ix_(parcell_ids, parcell_ids)]

        labels = roi

    return probability, amplitude, n_stim, n_impl, labels

def load_enigma(reference_labels):
    SC, sc_ctx_labels, _, _ = load_sc()
    FC, _, _, _ = load_fc()

    i,j = 0,0
    mismatch_ids = []

    while i < len(sc_ctx_labels) and j < len(reference_labels):
        if sc_ctx_labels[i][2:] != reference_labels[j][7:]:
            print(f"Labels {sc_ctx_labels[i]} and {reference_labels[j]} at index {i} do not match!\nAdding nan column/row for {reference_labels[j]}.")
            mismatch_ids.append(i)
            j+=1
        else:
            i+=1
            j+=1
            
    for i in mismatch_ids:
        SC = np.insert(SC,i,np.empty((SC.shape[1])), axis=0)
        SC = np.insert(SC,i,np.empty((SC.shape[0])), axis=1)
        FC = np.insert(FC,i,np.empty((FC.shape[1])), axis=0)
        FC = np.insert(FC,i,np.empty((SC.shape[0])), axis=1)

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

def load_rosen_halgren(reference_labels):
    path_sc_w = '../data/external/rosen_halgren_sc/public_PLoSBio_final/averageConnectivity_axonCount.mat'
    with h5py.File(path_sc_w, 'r') as f:
        SC_W = np.array(f.get('axonCount'))

    labels = np.loadtxt('../data/external/rosen_halgren_sc/public_PLoSBio_final/roi.txt', dtype=str)
    
    # reorder labels to match
    df_SC_W =  pd.DataFrame(SC_W, index=labels, columns=labels)
    df_SC_W = df_SC_W.loc[reference_labels,reference_labels]


    SC_W_correct = df_SC_W.to_numpy()

    SC_W = SC_W_correct
    SC_L = np.where(SC_W_correct==0.0,0,1/SC_W_correct)

    return SC_W, SC_L, None

def glasser_roi_distances(n_roi):
    # Teziste jednotlivych ROI (a nasledne vzdalenosti mezi nima) se daji spocitat tak, 
    # ze se vezmou 3d souradcnice vertexu na povrchu kortexu (soubor fsaverage_ico7_pial_surf.mat, 
    # N=163842 na hemisferu) a namapuji se na ne indexy jednotlivych ROI ze souboru 
    # HCP-MMP1.0_fsaverage_annot.mat (annot/lIdx/lh respektive annot/lIdx/rh ).

    distances = np.zeros((n_roi,n_roi))

    surf = h5py.File('../data/external/rosen_halgren_sc/public_PLoSBio_final/fsaverage_ico7_pial_surf.mat', 'r')    
    annot = h5py.File('../data/external/rosen_halgren_sc/public_PLoSBio_final/HCP-MMP1.0_fsaverage_annot.mat', 'r')

    if len(set(annot['annot/lIdx/lh'][:].squeeze())) != n_roi//2:
        print("POZOR! tahle funkce nefunguje!")
    
    centroids = np.zeros((n_roi,3))

    for h,hemisphere in [(0,'lh'),(1,'rh')]:
        for i in range(n_roi//2):
            coor = surf[f'srf/{hemisphere}/vertices'][:][:,annot[f'annot/lIdx/{hemisphere}'][:].squeeze()==i+1] # +1 protože číslováno od 1

            # tohle by se možná mělo udělat jinak? (zajímá mě těžiště konvexního obalu? ne úplně, protože to nebude konvexní...)
            centroids[i+h*n_roi//2] = np.mean(coor,axis=1) 

    for i in range(n_roi):
        for j in range(n_roi):
            distances[i,j] = np.linalg.norm(centroids[i]-centroids[j])
            
    return distances 
