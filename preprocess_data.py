import os,sys,importlib

import tensorflow as tf
import pandas as pd
import numpy

from PyUtilities.mkdir_p import mkdir_p

# __________________________________________________________________________________________ ||
spec = importlib.util.spec_from_file_location("preprocess_data_cfg", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
cfg = mod.cfg

# __________________________________________________________________________________________ ||
df = pd.read_csv(
        cfg.input_csv_path,
        index_col=0,
        )

# reco pt
#df.iloc[:,0] = (df.iloc[:,0]/df.iloc[:,17] - 1.)*30.
#df.iloc[:,0] = (df.iloc[:,0]/100.)-0.4337
df.iloc[:,0] = df.iloc[:,0]/100.
# reco eta
#df.iloc[:,1] = (df.iloc[:,1]/df.iloc[:,18] - 1.)*100.
# reco phi
#df.iloc[:,2] = (df.iloc[:,2]/df.iloc[:,19] - 1.)*100.
# T
cut_off = 0.8
A = (cut_off+1.)/(10-4.2)
B = -1.-A*4.2
def T_func(x):
    return A*x+B
df.iloc[:,3] = df.iloc[:,3]*1E9
df.iloc[:,3] = df.iloc[:,3].apply(lambda x: T_func(x) if x < 10. else T_func(10.)+(x-10.)/5 )

# isolation var
df.iloc[:,4] = df.iloc[:,4]*10.
# sum pT
df.iloc[:,5] = df.iloc[:,5]
df.iloc[:,6] = df.iloc[:,6]
df.iloc[:,7] = df.iloc[:,7]
# track D0
df.iloc[:,8] = df.iloc[:,8]/0.75
# track DZ
df.iloc[:,9] = df.iloc[:,9]/50.
# track Vx
df.iloc[:,10] = (df.iloc[:,10]+0.2476)/0.02
#df.iloc[:,10] = df.iloc[:,10]
#df.iloc[:,10] = ((df.iloc[:,10]+0.2476)/0.02)+1.
# track Vy
df.iloc[:,11] = (df.iloc[:,11]-0.6923)/0.02
#df.iloc[:,11] = df.iloc[:,11]
#df.iloc[:,11] = (df.iloc[:,11]-0.6923)/0.02-1.
# track Vz
df.iloc[:,12] = (df.iloc[:,12]-7.972)/50.
#df.iloc[:,12] = df.iloc[:,12]/50.-2.
# track outer x
df.iloc[:,13] = df.iloc[:,13]/1300.
# track outer y
df.iloc[:,14] = df.iloc[:,14]/1300.
# track outer z
df.iloc[:,15] = df.iloc[:,15]/3000.

# gen pt
df.iloc[:,17] = df.iloc[:,17]/100

# __________________________________________________________________________________________ ||
df_x = pd.concat(
        [
            df.iloc[:,:5],
            df.iloc[:,8:10],
            df.iloc[:,13:16],
            #df.iloc[:,8:9].abs(),
            #numpy.sign(df.iloc[:,9:10]),
            #df.iloc[:,9:10],
            #df.iloc[:,13:16].abs(),
            #numpy.sign(df.iloc[:,13:16]),
        ],
        axis=1,
        )
df_condition = pd.concat(
        [df.iloc[:,17:],df.iloc[:,10:13],],
        axis=1,
        )

idx_select = (df_condition.iloc[:,4].abs() < 3.).mul(df_condition.iloc[:,5].abs() < 3.)
df_x = df_x.loc[idx_select]
df_condition = df_condition.loc[idx_select]

# __________________________________________________________________________________________ ||
mkdir_p(os.path.abspath(os.path.dirname(cfg.preprocess_df_path)))
df_x.to_csv(cfg.preprocess_df_path.replace(".csv","_x.csv"),index=False,)
df_condition.to_csv(cfg.preprocess_df_path.replace(".csv","_condition.csv"),index=False,)
