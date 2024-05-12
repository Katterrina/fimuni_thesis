from netneurotools import metrics
    # https://netneurotools.readthedocs.io/en/latest/, pip install netneurotools does not work for me
    # I downloaded the repository and I am using the local copy
import numpy as np

def calculate_communicatin_metrics(ED,SC_L,SC_W):
    metrics_dict = dict()

    metrics_dict["ED"] = ED
    np.fill_diagonal(SC_W,0)
    np.nan_to_num(SC_W,nan=0,copy=False)
    metrics_dict["SC_W"] = SC_W


    if SC_L is not None:
        SC_L = np.where(SC_W==0,np.inf,SC_L)
        np.fill_diagonal(SC_L,0)
        metrics_dict["SC_L"] = SC_L

    # shortest path efficiency
    if SC_L is not None:
        shortest_paths,_ = metrics.distance_wei_floyd(SC_L)
        metrics_dict["SPE"] = np.divide(1,shortest_paths)

    # shortest path efficiency using weights
    shortest_paths,_ = metrics.distance_wei_floyd(1/SC_W)
    metrics_dict["SPE_W"] = np.divide(1,shortest_paths)

    # communicability
    metrics_dict["COM"] = metrics.communicability_wei(np.nan_to_num(SC_W,nan=0))

    # SI
    metrics_dict["SI"] = metrics.search_information(SC_W,ED)

    # SI with length
    if SC_L is not None:
         metrics_dict["SI_L"] = metrics.search_information(SC_W,SC_L)

    # navigation path eff.
    _,_,nav_paths_dist,_,nav_paths = metrics.navigation_wu(ED, SC_W)

    if SC_L is None:
        metrics_dict["NAV"] = np.divide(1,nav_paths_dist)
    # navigation path eff. with L
    else:
        nav_paths = metrics.get_navigation_path_length(nav_paths, SC_L)
        metrics_dict["NAV"] = np.divide(1,nav_paths)


    _, metrics_dict["DIF"] = metrics.diffusion_efficiency(SC_W)
    
    return metrics_dict