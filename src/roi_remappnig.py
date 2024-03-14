import pandas as pd

def reorder_matrix_based_on_reference(labels, reference_labels, matrix):
    df_matrix = pd.DataFrame(matrix, index=labels, columns=labels)
    df_matrix = df_matrix.loc[reference_labels,reference_labels]

    return df_matrix.to_numpy()

def schaefer_to_schaefer(matrix,mapping_path,mapping_idx):
    mapping = pd.read_csv(mapping_path)
    m = mapping[mapping_idx]

    df_matrix =  pd.DataFrame(matrix)
    df_matrix = df_matrix.loc[m,m]

    return df_matrix.to_numpy()