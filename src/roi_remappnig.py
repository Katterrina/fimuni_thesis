import pandas as pd

def reorder_matrix_based_on_reference(labels, reference_labels, matrix):
    df_matrix = pd.DataFrame(matrix, index=labels, columns=labels)
    df_matrix = df_matrix.loc[reference_labels,reference_labels]

    return df_matrix.to_numpy()

def schaefer_to_freesurfer_schaefer(matrix):
    mapping = pd.read_csv('../data/external/pytepfit/ROI_MAPPING.csv')
    m = mapping.idx_csv

    df_matrix =  pd.DataFrame(matrix)
    df_matrix = df_matrix.loc[m,m]

    return df_matrix.to_numpy()

def schaefer17_to_schaefer7(matrix):
    mapping = pd.read_csv('../data/external/pytepfit/ROI_MAPPING_7_17.csv')
    m = mapping.idx_17

    df_matrix =  pd.DataFrame(matrix)
    df_matrix = df_matrix.loc[m,m]

    return df_matrix.to_numpy()