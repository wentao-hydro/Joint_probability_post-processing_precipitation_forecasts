
from auxiliary_function_transform_map import *
import numpy as np
from scipy.stats import norm
 

def fcst2ensem_powertransform (x_fcstnew, n_mem_output, par_all ):  
  
    # make prediction given one new forecast 
    # 1.transform
    # 2.compute the conditianal distribution in Normal space: two situations
    # 3.back-transform

        
    # get fitted parameters 
    power_fcst=par_all[0]
    power_obs= par_all[1]   
    threshold_fcst_ori=par_all[2]#threshold in original space
    threshold_obs_ori= par_all[3]
    threshold_fcst_tranformed= par_all[4]#threshold in transformed space
    mu1 = par_all[6]
    sigma1 = par_all[7]    
    mu2 = par_all[8]
    sigma2 = par_all[9] 
    par_corr = par_all[10]


    
    
    #2. compute conditional distribution in Normal space: two situations 
    #2.1 F(y|x=0) if x<=censor threshold 
    if (x_fcstnew <= threshold_fcst_ori):
        threshold_CDF = norm.cdf (threshold_fcst_tranformed, mu1, sigma1)
        rand_CDF = np.random.rand(n_mem_output) * threshold_CDF #generate N CDF value, which follows uniform distribution [0, CDF(Threshold) ]
        x_samples = norm.ppf(rand_CDF, mu1, sigma1)#compute quantiles, so get N  random samples of normal distribution less than the threshold
        
        #the conditional distribution of B(v|u<=u0)
        mu_cond  =  mu2 + par_corr * (sigma2/sigma1) * (x_samples-mu1) #use the generated "x_samples" here to substitute x in the cond. dist. function
        sigma_cond  =   (1-par_corr*par_corr) ** (1/2) * sigma2
        ens_y  =  np.random.randn(n_mem_output) * sigma_cond + mu_cond   #!!   draw random samples of normal distribution here



    #2.2 F(y|x>0)
    else: # if x>censor threshold 
        
        #power transformation for the new fcsts:
        zx_fcstnew = x_fcstnew ** (power_fcst)

        #conditional distribution of B(v|u>u0)
        mu_cond  =  mu2 + par_corr * (sigma2/sigma1) *(zx_fcstnew-mu1)
        sigma_cond  =   (1 - par_corr*par_corr) ** (1/2) * sigma2
        ens_y  =  np.random.randn(n_mem_output) * sigma_cond + mu_cond   #!!   draw random samples of normal distribution here
    
    

    
    #3. back-transform: use marginal parameters for obs 
    ens_y_ori = power_inverse_transform_function(ens_y, power_obs  ,threshold_obs_ori)

    
    return ens_y_ori  #return y in original space
  
 