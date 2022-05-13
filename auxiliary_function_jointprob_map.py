#%% 

import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal, norm, pearsonr

#import auxiliary_functions:
from auxiliary_function_transform_map import *
from NelderMead_mapping_parameter import NelderMead_mapping_parameter


#%%

def Fit_rho_fixed_map(obs, fcst,  obs_par,fcst_par, optim_method='neldermead_mirror'):
    # fit the parameter of correlation coefficient (rho) in Normal space
    # MLE: maximize log likelihood function, i.e. minimize the negative log likelihood function
    # input: transformed obs and fcst; censoring threshold of obs and fcst;
    #        mean and standard deviation parameter of marginal distribution of obs and fcst
    # output: fitted correlation coefficient parameters 
    
    
    par0 = 0.3 #initial values of parameters
    



    if ( optim_method== 'neldermead_mirror'):
        fitres = minimize (Likelihood_corr_MapParameter , par0, args=(obs, fcst,  obs_par,fcst_par), 
          method = "Nelder-Mead")
        par_unmapped = fitres.x
        par_corr = NelderMead_mapping_parameter ( par_unmapped, [0.001] ,  [0.999]  ) 
    

    elif (optim_method =='L-BFGS-B'):
        fitres = minimize(Likelihood_corr, par0, args=(obs, fcst, obs_par,fcst_par),
          bounds=[(0.001, 0.999)], method='L-BFGS-B')
        par_corr = fitres.x
        
    elif ( optim_method== 'nelder-mead'):
        fitres = minimize(Likelihood_corr, par0, args=(obs, fcst, obs_par,fcst_par), 
          method='Nelder-Mead')
        par_corr = fitres.x

    
    # #output:
    return par_corr  #return the fitted parameter result



 


#%% likelihood for the joint distribution of obs and fcst,follows censored bivariate Normal distribution
def Likelihood_corr(par_corr, obs, fcst,  obs_par,fcst_par):
    #The likelihood function to fit correlation coefficient in transformed space using MLE,
    #consider data less than threshold as censored data
    #fcst_par: [par_power, zero_threshold_tranformed, mean,sd

    #get par
    threshold_fcst = fcst_par[1] #! censoring threshold in tranformed space
    threshold_obs = obs_par[1]
    mu1=fcst_par[2]
    sigma1=fcst_par[3]    
    mu2=obs_par[2]
    sigma2=obs_par[3]
    par_corr = par_corr[0]


    n_sample = len(fcst)
    fcst_obs_data =  np.empty((n_sample, 2))
    fcst_obs_data[:, 0] = fcst
    fcst_obs_data[:, 1] = obs
    ind_pos_fcst = fcst >threshold_fcst
    ind_pos_obs = obs >threshold_obs

    threshold_data = np.empty((1, 2))
    threshold_data[0,0] = threshold_fcst
    threshold_data[0,1] = threshold_obs
    

    # parameter for bivariate normal distribution 
    mu_2D = np.array([mu1,mu2])
    Cov_2D = np.array([[sigma1**2, par_corr*sigma1*sigma2] ,  [par_corr*sigma1*sigma2, sigma2**2] ] )


    #four cases of likelihood functions:  
    func11 = multivariate_normal.pdf( fcst_obs_data, mean=mu_2D , cov=Cov_2D )#fcst>0,obs>0
    func10 = norm.pdf(fcst, mu1, sigma1) * norm.cdf(threshold_obs, loc = (mu2 + par_corr*sigma2/sigma1*(fcst-mu1)), scale = (1-par_corr**2)**(0.5)*sigma2 )#fcst>0,obs=0
    func01 = norm.pdf(obs, mu2, sigma2) * norm.cdf(threshold_fcst, loc = (mu1 + par_corr*sigma1/sigma2*(obs-mu2)), scale = (1-par_corr**2)**(0.5)*sigma1 )#fcst=0,obs>0
    func00 = multivariate_normal.cdf(threshold_data, mean=mu_2D , cov=Cov_2D )#fcst=0,obs=0; only one CDF value



    #if not satisfy the condition in each case, the corresponding f value equals zero
    f1 =  ( func11 * ((ind_pos_fcst==1) * (ind_pos_obs ==1)) )   
    f2 =  ( func10 * ((ind_pos_fcst==1) * (ind_pos_obs ==0)) ) 
    f3 =  ( func01 * ((ind_pos_fcst==0) * (ind_pos_obs ==1)) ) 
    f4 =  ( func00 * ((ind_pos_fcst==0) * (ind_pos_obs ==0)) ) 

    
    

    #negative log likelihood function value for all samples together; 
    # effective cdf or pdf value should be larger than 0
    neg_loglikelihood = -np.nansum( np.log(f1[ f1>0 ]) ) - np.nansum( np.log(f2[ f2>0]) ) - np.nansum( np.log(f3[ f3>0 ]) )  - np.nansum( np.log(f4[ f4>0 ]) )
    

    return neg_loglikelihood #  Return negative log likelihood 




#%% map parameter
def Likelihood_corr_MapParameter(parameters, obs, fcst,  obs_par,fcst_par):
    #  The likelihood function for power transform fitting using MLE
    #  Firstly, transform the parameters to ensure they are within the bounds by Q.J.'s mirroring method,
    #  then optimize by traditional Nelder-Mead method


    par_mapped=NelderMead_mapping_parameter ( parameters,  [0.001] ,  [0.999] )#设置
    neg_loglikelihood=Likelihood_corr(par_mapped, obs, fcst, obs_par,fcst_par)

    return neg_loglikelihood


 
 

 
