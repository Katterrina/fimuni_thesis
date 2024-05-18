import pandas as pd
import numpy as np
import pingouin as pg

def partial_corr_ED(d,x,method):
    result = []
    result.append(pd.DataFrame({"n":[0],"r":[0],"CI95%":[None],"p-val":[1],"Y":["ED"]})) # just to have fitting images

    for k in d.keys():
        if k == "ED" or k == x or np.sum(pd.notna(d[x]))<=2 or np.sum(pd.notna(d[k]))<=2:
            continue
        df_one_result = pg.partial_corr(data=d,x=x,y=k,covar="ED", method=method)
        df_one_result['Y'] = k
        result.append(df_one_result)
    
    return pd.concat(result)

def corr_per_roi(df):
    r_per_roi = dict()
    for label in list(set(df.index.get_level_values(0))):
        relevant_df = df.xs(label, drop_level=False)

        try:
            r_per_roi[label] = pg.pairwise_corr(relevant_df,'response', method='spearman').round(3)
        except:
            pass # we need enoug non-zero values, otherwise the correlation does not make sense
        
    return pd.concat(r_per_roi)

def partial_corr_per_roi(df):
    r_per_roi = dict()
    for label in list(set(df.index.get_level_values(0))):
        relevant_df = df.xs(label, drop_level=False)
        try:
            r_per_roi[label] = partial_corr_ED(relevant_df,'response', method='spearman').round(3)
        except:
            pass # we need enoug non-zero values, otherwise the correlation does not make sense
    return pd.concat(r_per_roi)

def abs_r_to_dataframe(df):
    df['r_abs'] = df['r'].apply(abs)

def sigf_sign_to_dataframe(df):
    if 'p-unc' in df.columns:
        df['p_sigf'] = df['p-unc'].apply(lambda x: "*" if x< 0.05 else "")
    elif'p-val' in df.columns:
        df['p_sigf'] = df['p-val'].apply(lambda x: "*" if x< 0.05 else "")

def sigf_num_to_dataframe(df):
    if 'p-unc' in df.columns:
        df['p_sigf_num'] = df['p-unc'].apply(lambda x: 1 if x< 0.05 else 0)
    elif'p-val' in df.columns:
        df['p_sigf_num'] = df['p-val'].apply(lambda x: 1 if x< 0.05 else 0)

def r_only_sigf_to_dataframe(df):
    df['r_sigf'] = df.apply(lambda x: x["r"] if x["p_sigf"] == '*' else np.nan,axis=1)   