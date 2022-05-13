#%%

import xarray as xr
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from numpy import ndarray

#%%
#auxiliary_functions:
from auxiliary_function_transform_map import *
from auxiliary_function_jointprob_map import *
from fcst2ensem_powertransform import *


#%% simplified-BJP (BJPs) for precipitation forecasts, 2022.3.6 written by Wentao Li wentaoli@hhu.edu.cn

class BJP_pre_simple():
    def __init__(self,  threshold_obs = 0.1, threshold_fcst = 0.1) -> None:
        """ Simplified-BJP for precipitation forecasts by MLE for censored data 

        Parameter: 

        threshold_obs: float, default = 0.1mm/day
                the censoring threshold for obs
                
        threshold_fcst: float, default = 0.1mm/day
                the censoring threshold for fcst
        """              
        self.threshold_obs = threshold_obs
        self.threshold_fcst = threshold_fcst
    

    def fit(self, fcst:ndarray, obs:ndarray ) -> pd.DataFrame:
        """ Fit parameters given historical fcst series and obs series

        Input: 

        fcst: ndarray
                raw fcst series 
        obs:  ndarray
                obs series

        Output: dataframe
                fitted parameters
        """        
         
        
        obs_transformed, par_obs = Fit_power_transform_map(obs, self.threshold_obs)
        fcst_transformed, par_fcst = Fit_power_transform_map(fcst, self.threshold_fcst)
        par_corr =   Fit_rho_fixed_map(obs_transformed, fcst_transformed,par_obs, par_fcst)
        
        self.par_obs = par_obs                        
        self.par_fcst = par_fcst  #[par_power, zero_threshold_tranformed, mean,sd ]                       
        self.par_corr = par_corr 
        par_all = np.zeros((11,))            
        par_all[0] =  par_fcst[0]#power par
        par_all[1] =  par_obs[0]
        par_all[2] =  self.threshold_fcst#threshold in original space
        par_all[3] =  self.threshold_obs
        par_all[4] =  par_fcst[1]#threshold in transform space
        par_all[5] =  par_obs[1]
        par_all[6:8] =  par_fcst[2:]
        par_all[8:10] =  par_obs[2:]
        par_all[10] =  par_corr
        self.par_all = par_all 
        parnames = ["power fcst", "power obs", 
         "threshold original fcst", "threshold original obs",
         "threshold transformed fcst", "threshold transformed obs",
         "mu fcst",'sigma fcst',"mu obs",'sigma obs', "corr"   ]
        par_df = pd.DataFrame( {"parameters": par_all}, index=parnames )
        
        return  par_df                     
          


    def predict(self, fcst_new:ndarray, n_mem_set=100 ) -> ndarray:
        """ Prediction given future fcst series

        Input: 

        fcst_new: ndarray
                raw fcst series 
        n_mem_set: int
                number of ensemble members

        Output: ndarray (nday , n_mem_set)
                post-processed ensemble fcst 
        """

        n_day = fcst_new.shape[0]
        ens_fcst_output = np.zeros( (n_day, n_mem_set) )

        if n_mem_set >= 100:   #if ensemble size is large
            n_mem_output=n_mem_set #get random samples from the distribution
            self.n_mem_set = n_mem_set #if ensemble size is large
            for iday in range(n_day): #loop for each day
                x_fcstnew = fcst_new[iday]
                ens_fcst_output[iday,:] = fcst2ensem_powertransform (x_fcstnew ,n_mem_output ,self.par_all )
            

        else:    #if want smaller ensemble size, get quantiles from the random samples
            n_mem_output=500 #get 500 random samples from the distribution
            self.n_mem_set = n_mem_set 
            step_mem = 1/n_mem_set
            quantile_vec = np.arange( step_mem/2,1,step_mem) *100  
            for iday in range(n_day): #loop for each day
                x_fcstnew = fcst_new[iday]
                prediction_1day = fcst2ensem_powertransform (x_fcstnew ,n_mem_output ,self.par_all )
                ens_fcst_output[iday,:] = np.percentile(prediction_1day, quantile_vec )#get 25 quantiles


        return ens_fcst_output






# %%
