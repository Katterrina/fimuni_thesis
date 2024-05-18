import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches as mpatches
from src.paths import *
from src.data import get_labels_from_file
import numpy as np

COLOR_MAPPING_YEO = {'Default':'yellow', 'Limbic':'blue','SalVentAttn':'red',  'DorsAttn':'green','Vis':'purple','Cont':'orange', 'SomMot':'pink'}

def plot_results(d,title):
    plt.figure(figsize=(7,10))
    
    ax = sns.barplot(data=d, x='Y', y='r_abs',hue="dataset")
    ax.set_title(title)  
    ax.set_ylim(bottom=0, top=1)
    ax.tick_params(axis='x', rotation=90)

    plt.xlabel('Communication metric')
    plt.ylabel('Correlation coefficient')

    plt.show()

def plot_results_overlay(d,d_partial,title,fig_dir=None,figsize=(7,10)):
    plt.figure(figsize=figsize)

    datasets = d["dataset"].unique()
    metrics = d["Y"].unique()

    ax = sns.barplot(data=d, x='Y', y='r_abs',hue="dataset", alpha=0.35,legend=False)

    for i,dataset in enumerate(datasets):
        dataset_container = ax.containers[i]
        j = 0
        for metric in metrics:
            if ((d["dataset"] == dataset) & (d['Y'] == metric)).any():
                significance = d[(d["dataset"] == dataset) & (d['Y'] == metric)]["p_sigf"].values[0]
                bar = dataset_container[j]
                bbox = bar.get_bbox().bounds
                x = bbox[0] + (bbox[2]/2)
                y = bbox[3] + 0.005
                plt.text(x,y,significance,horizontalalignment='center')
                j +=1

    sns.barplot(data=d_partial, x='Y', y='r_abs',hue="dataset",ax=ax)

    for i2,dataset in enumerate(datasets):
        dataset_container = ax.containers[i+i2+1]
        j = 0
        for metric in metrics:
            if ((d_partial["dataset"] == dataset) & (d_partial['Y'] == metric)).any():
                significance = d_partial[(d_partial["dataset"] == dataset) & (d_partial['Y'] == metric)]["p_sigf"].values[0]
                bar = dataset_container[j]
                bbox = bar.get_bbox().bounds
                x = bbox[0] + (bbox[2]/2)
                y = bbox[3] + 0.005
                plt.text(x,y,significance,horizontalalignment='center')
                j +=1

    ax.set_title(title)  
    ax.set_ylim(bottom=0, top=1)
    ax.tick_params(axis='x', rotation=90)

    at = AnchoredText("opaque for partiall correlation\ninfluence of ED controlled",frameon=False, loc='upper left')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)

    plt.xlabel('Communication metric')
    plt.ylabel('Correlation coefficient')

    img_name = title.replace("\n"," ")
    img_name = img_name.replace(" ","_")

    image_path = path_figures(fig_dir+img_name+".pdf")
    plt.savefig(image_path,bbox_inches='tight',pad_inches=0)

    plt.show()

def prepare_barcode(n_rows,n_cols,title=None):
    plt.style.use('default')
    pixel_per_bar = 6
    dpi = 100

    fig, ax = plt.subplots(n_rows, 1, figsize=(n_cols * pixel_per_bar / dpi, n_rows/1.5), dpi=dpi, sharex=True)
    fig.suptitle(title)
    fig.subplots_adjust(right=0.7)

    return fig, ax

def plot_one_barcode(ax,data,title=None):

    if not isinstance(data, np.ndarray):
        code = data.to_numpy().reshape(1,-1).astype(float)
    else:
        code = data.reshape(1,-1).astype(float)

    code = np.nan_to_num(code)

    ax.imshow(code, cmap='binary', aspect='auto',
          interpolation='nearest')
    ax.set_title(f"  {title}", loc="right", y=0, ha="left", va="center")
    ax.set_yticks([])

def plot_adjacency_matrix(matrix,ax,title,subnets_idx=False,norm=None,mask=None):

    g =sns.heatmap(matrix,ax=ax,square=True,cbar=False,yticklabels=False, xticklabels=False,cmap='gray',norm=norm,mask=mask)
    g.set_facecolor('k')
    ax.set_title(title)

    if subnets_idx:
        subnet_0,id_start_0 = subnets_idx[0]
        for subnet,id_start in subnets_idx[1:]:
            ax.add_patch(mpatches.Rectangle((id_start_0, id_start_0),
                                              id_start-id_start_0, # Width
                                              id_start-id_start_0, # Height
                                              facecolor="none",
                                              edgecolor=COLOR_MAPPING_YEO[subnet_0],
                                              linewidth=1))
            subnet_0,id_start_0 = subnet,id_start

        ax.add_patch(mpatches.Rectangle((id_start_0, id_start_0),
                                              200-id_start_0, # Width
                                              200-id_start_0, # Height
                                              facecolor="none",
                                              edgecolor=COLOR_MAPPING_YEO[subnet_0],
                                              linewidth=1))

def yeo_legend_patches():
    legend_patches = []
    for c in COLOR_MAPPING_YEO:
        legend_patches.append(mpatches.Patch(color=COLOR_MAPPING_YEO[c], label=c))

    return legend_patches

def plot_structural_matrices_weight_lengths(SC_matrices,fig_dir,parcellation):
    number_of_SC = len(SC_matrices)
    plt.style.use('seaborn-v0_8-white')
    fig, axs = plt.subplots(2,number_of_SC, figsize=(4*number_of_SC,8),tight_layout=True,sharex=True,sharey=True)

    idx_by_yeonet = False

    if parcellation=="schaefer":
        centroids_file = path('interim/schaefer_parcellation_mappings/ROI_MAPPING_pytepfit.csv')
        
        idx_by_yeonet = []
        ROI_colors = []

        legend_patches = yeo_legend_patches()

        labels = get_labels_from_file(centroids_file,"roi_name")

        network = None
        for i,l in enumerate(labels):
            n = l.split('_')[2]
            ROI_colors.append(COLOR_MAPPING_YEO[n])
            if n != network:
                idx_by_yeonet.append((n,i))
                network = n

        fig.legend(handles=legend_patches,loc='lower center', ncols=7,bbox_to_anchor=(0,-0.05,1,1))

    for i in range(number_of_SC):
        name, SC_W, SC_L, SC_W_log = SC_matrices[i]

        plot_adjacency_matrix(SC_W_log,axs[0,i],f"SC {name} - weights (log)",subnets_idx=idx_by_yeonet,mask=SC_W==0)

        if SC_L is not None:
            plot_adjacency_matrix(SC_L,axs[1,i],f"SC {name} - lengths",subnets_idx=idx_by_yeonet,mask=SC_W==0)
    
        else:
            plot_adjacency_matrix(np.zeros(SC_W.shape),axs[1,i],f"SC {name} - NO lengths",subnets_idx=idx_by_yeonet,mask=SC_W==0)


    plt.savefig(path_figures(fig_dir+"sc_matrices.pdf"))