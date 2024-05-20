import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches as mpatches
from src.paths import *
from src.data import get_labels_from_file
import numpy as np
from adjustText import adjust_text

from matplotlib.colors import LinearSegmentedColormap

COLOR_MAPPING_YEO = {'Default':'yellow', 'Limbic':'blue','SalVentAttn':'red',  'DorsAttn':'green','Vis':'purple','Cont':'orange', 'SomMot':'pink'}

def plot_results_per_roi(rpr,fig_dir,title=None):
    plt.figure(figsize=(7,7))
    
    ax = sns.boxplot(data=rpr, x="Y", y="r_sigf", hue="dataset")
    ax.set_title(title) 
    ax.set_ylim(bottom=-1, top=1)
    ax.tick_params(axis='x', rotation=90)
    plt.xlabel('Communication metric')
    plt.ylabel('Correlation coefficient')
    title_save = (title.replace("\n","_")).replace(" ","_")
    plt.savefig(f'{path_figures(fig_dir+title_save)}.pdf')
    plt.show()

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

def plot_tmseeg_data(data,colors=None,title=None,stimulation_time=None):
    plt.rcParams['figure.figsize'] = [10, 5]
    fig,ax = plt.subplots()
    if colors:
        for i,d in enumerate(data.T):
            ax.plot(d,color=colors[i], alpha=0.7,)
        ax.legend(handles=yeo_legend_patches(),loc='upper left', ncols=1,bbox_to_anchor=(0,-0.05,1,1))
    else:
        ax.plot(data, alpha=0.7)
    plt.xlabel("time [ms]")
    plt.ylabel("EEG [au]")
    if stimulation_time:
        ax.axvline(stimulation_time, ymin=np.min(data), ymax=np.max(data),ls=":",label="stimulation time",color="k")
    plt.title(title)
    

def plot_one_roi_from_tmseeg_data(data,roi_id,labels,title=None,constants_h=[],constants_v=[]):
    plt.rcParams['figure.figsize'] = [10, 5]
    plt.figure()
    plt.plot(data[:,roi_id])
    for c in constants_h:
        plt.plot([c]*data.shape[0],color="k")
    for c in constants_v:
        plt.vlines(c, np.min(data), np.max(data),color="k")
    plt.title(labels[roi_id])
    plt.xlabel("ms")
    plt.show()

def plot_one_roi_response_definitions(selected_curve,label,peak_analysis_result,thr,resp_length,fig_dir):

    amp_id, amp, amp_h_id, amp_h = peak_analysis_result

    auc_color = '#FBC15E'
    first_peak_color = '#348ABD'
    highest_peak_color = '#E24A33'

    plt.subplots(figsize=(10,5))

    plt.plot(selected_curve,color=auc_color)
    plt.fill_between([i for i in range(len(selected_curve))],selected_curve,alpha=0.15,color=auc_color)
    plt.text(resp_length-20, 2, "AUC",color=auc_color,fontsize=20)

    plt.vlines(amp_id, 0, amp,color=first_peak_color)
    plt.text(amp_id+1, amp+0.5, f"first_peak = {amp:.2f}",color=first_peak_color)
    plt.plot([1]*(amp_id+1),color=first_peak_color,ls="--")
    plt.text(amp_id+1, 1+0.3, f"first_peak_latency = {amp_id}",color=first_peak_color)

    plt.vlines(amp_h_id, 0, amp_h,color=highest_peak_color)
    plt.text(amp_h_id+1, amp_h+0.5, f"highest_peak = {amp_h:.2f}",color=highest_peak_color)
    plt.plot([4]*(amp_h_id+1),color=highest_peak_color,ls="--")
    plt.text(amp_h_id+1, 4+0.3, f"highest_peak_latency = {amp_h_id}",color=highest_peak_color)

    ax = plt.gca()
    ax.set_ylim([0,amp_h+3])
    ax.set_xlim([0, len(selected_curve)])

    
    plt.title(f"ROI identifier: {label}")
    plt.plot([thr]*len(selected_curve),color="k",ls=":")
    plt.text(len(selected_curve)-5, thr+0.3, f"threshold = {thr}",color='k',horizontalalignment='right')
    plt.xlabel("time [ms]")
    plt.ylabel("EEG [au]")
    
    path = path_figures(fig_dir+label+"_response_def.pdf")
    plt.savefig(path,bbox_inches='tight',pad_inches=0)
    plt.show()

def plot_df_as_heatmap(df,x_axis,y_axis,value,fig_dir=None,x_label="threshold",y_label="",p=None,title=None,ax=None):
    plt.figure(figsize=(8,8))
    pivot = df.pivot_table(index=x_axis, columns=y_axis, values=value,sort=False)
    if p is not None:
        pivot_p = df.pivot_table(index=x_axis, columns=y_axis, values=p,sort=False)
        pivot = pivot.where(pivot_p < 0.05)

    cmap = LinearSegmentedColormap.from_list('', ['#FF2200', 'white', '#FF2200'])# 'seismic'

    if ax is not None:
        sns.heatmap(pivot, annot=True,center=0,cmap=cmap,vmin=-1, vmax=1,ax=ax,square=True,cbar=False)
    else:
        ax = sns.heatmap(pivot, annot=True,center=0,cmap=cmap,vmin=-1, vmax=1,ax=ax,square=True,cbar=False)
    ax.set_title(title)
    ax.set(xlabel=x_label)
    ax.set(ylabel=y_label)
    plt.yticks(rotation=0) 
    if fig_dir is not None:
        plt.savefig(path_figures(fig_dir+title+".pdf"),bbox_inches='tight',pad_inches=0)
    plt.show()



def scatter_two_columns_from_dataframe(df,col1,col2,label,log_axes=False,labels=False,corr_line=None,title=None,labelx=None,labely=None,fig_dir=None):
    a1 = np.array(df[col1])
    a2 = np.array(df[col2])
        
    plt.style.use('ggplot')
    fig,ax = plt.subplots(figsize=(7,7))

    if corr_line:
        ax.plot(corr_line[0],corr_line[1],c="gray")
    
    ax.scatter(a1,a2,c=df["color7"])
    ax.legend(handles=yeo_legend_patches())

    ax.set_xlabel(col1+"_F-Tract")
    ax.set_ylabel(col2)


    if log_axes:
        ax.set_yscale('log')
        ax.set_xscale('log')
    
    if labels and not log_axes:
        texts = []

        for i, coor in enumerate(zip(a1,a2)):
            x,y = coor
            if np.isnan(x) or np.isnan(y):
                continue

            texts.append(ax.text(x,y,df[label][i],fontsize='x-small'))

        adjust_text(texts, force_text=(0.5,0.5),expand=(1.2,1.2),expand_axes=True,time_lim=5,arrowprops=dict(arrowstyle='-', color='black', lw=0.5))
    
    if labely:
        plt.ylabel(labely)
    if labelx:
        plt.xlabel(labelx)

    if title and fig_dir is not None:
        plt.title(title)
        title_save = (title.replace(" ","_")).replace("\n","_")
        plt.savefig(path_figures(fig_dir+title_save),bbox_inches='tight',pad_inches=0)