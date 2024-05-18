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

def abs_r_to_dataframe(df):
    df['abs'] = df['r'].apply(abs)

def sigf_sign_to_dataframe(df):
    if 'p-unc' in df.columns:
        df['p_sigf'] = df['p-unc'].apply(lambda x: "*" if x< 0.05 else "")
    elif'p-unc' in df.columns:
        df['p_sigf'] = df['p-val'].apply(lambda x: "*" if x< 0.05 else "")