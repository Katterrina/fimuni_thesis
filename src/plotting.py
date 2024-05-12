import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.offsetbox import AnchoredText
from src.paths import *
import numpy as np

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