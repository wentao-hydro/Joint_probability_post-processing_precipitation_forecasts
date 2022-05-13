


#%%

import xarray as xr
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from numpy import ndarray

#%%
from BJPsimple_class import *

# %% load data of historical fcst series and obs series
file_data = np.load('fcst_obs.npz')
obs=file_data['obs']  
fcst=file_data['fcst']  


#%%
BJP_eg = BJP_pre_simple()


#%% fit parameters 
par_df = BJP_eg.fit(fcst,obs )


# %% make prediction give new fcst series
ens_post = BJP_eg.predict( fcst, 100)



# %% check post-processed  ens mean
ensmean_post = np.mean(ens_post, -1)
print('corr', pearsonr(fcst, obs  )[0], pearsonr(ensmean_post, obs  )[0])
print('bias', np.mean(fcst) / np.mean(obs) - 1  , np.mean(ensmean_post) / np.mean(obs) - 1  ) 



# %%
