import pandas as pd

def reorder_matrix_based_on_reference(labels, reference_labels, matrix):
    """
    Using matrix labels and taget reference labels, reorder the matrix.

    Parameters:
    labels (list[str])
    reference_labels (list[str])
    matrix (2D np.array)
    
    Returns:
    2D np.array
    """
    df_matrix = pd.DataFrame(matrix, index=labels, columns=labels)
    df_matrix = df_matrix.loc[reference_labels,reference_labels]

    return df_matrix.to_numpy()

def schaefer_to_schaefer(matrix,mapping_path,mapping_idx):
    """
    Reorder ROIs of Schaefer200 parcellation using csv file with mapping.

    Parameters:
    matrix (2D np.array)
    mapping_path (path): path to csv file containing ROI indeces in 
        the target order in mapping_idx column
    mapping_idx (str): column in the mapping csv with the desired ordering 
    
    Returns:
    2D np.array
    """
    mapping = pd.read_csv(mapping_path)
    m = mapping[mapping_idx]

    df_matrix =  pd.DataFrame(matrix)
    df_matrix = df_matrix.loc[m,m]

    return df_matrix.to_numpy()