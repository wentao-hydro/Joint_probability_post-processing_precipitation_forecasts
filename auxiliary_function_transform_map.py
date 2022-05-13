#%%
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr, norm
from NelderMead_mapping_parameter import NelderMead_mapping_parameter
#12.29 modify: in power transform, use x**par_power instead of x**(1/par_power)
#3.6: return all parameters in a vector




def power_transform_function ( x, par_power, zero_threshold ):
    # zero_threshold: threshold in original space!!!
    
    data_transformed = np.zeros_like(x) 
    data_transformed[ x>zero_threshold ] = x[ x>zero_threshold  ] ** (par_power)#set data less than threshold  as  zero 
    
    return data_transformed  
  



def power_inverse_transform_function ( x, par_power, zero_threshold ):
    # zero_threshold: threshold in original space!!!

    data_inv_transformed=np.zeros_like(x) #same size as input
    zero_threshold_tranformed= zero_threshold ** (par_power)
    data_inv_transformed[x > zero_threshold_tranformed ] = (x[x > zero_threshold_tranformed ]) ** (1/par_power)
    data_inv_transformed[x <= zero_threshold_tranformed ]= 0# zero_threshold #set as the censoring threshold, not zero


    return data_inv_transformed 




def Fit_power_transform_map(x, threshold_original, optim_method='neldermead_mirror'):
    #fit power transform parameter by MLE
    #input: x data
    #output: fitted parameters 
    
    #initial values of parameters
    par0 = [.33, .1, .1]  #here modified in 12.19: use power par, not inverse power par





    if ( optim_method== 'neldermead_mirror'):
        fitres_power = minimize (Likelihood_Power_MapParameter , par0, args=(x , threshold_original),
         method = "Nelder-Mead")
        par_unmapped = fitres_power.x
        par = NelderMead_mapping_parameter ( par_unmapped,  [0, -np.inf,-np.inf] ,  [ 0.99, np.inf, np.inf]  ) 


    elif (optim_method =='L-BFGS-B'):
        fitres_power = minimize( Likelihood_Power, par0, args=(x , threshold_original),
         method = "L-BFGS-B", bounds = ((0, None),(None,None),(None,None) )  ) # ensure power paramter > 0
        par = fitres_power.x
        
    elif ( optim_method== 'nelder-mead'):
        fitres_power = minimize ( Likelihood_Power, par0, args=(x , threshold_original), 
         method='Nelder-Mead')
        par = fitres_power.x
    


    #transform
    par_power = par[0]
    sd = np.exp(par[1])
    mean = sd * (par[2])  
    
    
    #apply to data
    zero_threshold_tranformed = (threshold_original) ** (par_power)  
    data_transformed = power_transform_function ( x, par_power, threshold_original )
    # par_distribution = [mean,sd]   


    #all parameters in a vector
    par_vec_all = np.array([par_power, zero_threshold_tranformed, mean,sd ]  )
    
    
    # #output:
    return data_transformed, par_vec_all



#%% likelihood for power transformed variable, follows censored Normal distribution
def Likelihood_Power(par, x, threshold_original):
    #The likelihood function for power transform, using MLE
    p = par[0]
    sd = np.exp(par[1]) #standard deviation
    mean = par[2] * sd
    
    
    ya = x[~np.isnan(x)]  #data in original space
    yb = ya ** (p)  #  data in Normal space
    threshold_transformed =  threshold_original ** (p)

    
    #CASE 1. Likelihood function for non-censored data (data> censored threshold): use density * Jacob
    nsample = ya.shape[0]
    Jacob_value = np.zeros((nsample,1))#Jacob即为导数项
    Jacob_value[np.nonzero(ya),0] = (p) * ( ya[np.nonzero(ya)] ** (p - 1) )   #nonzero
    density_norm = norm.pdf ( yb,  mean,  sd)
    density_yb = np.multiply(density_norm, Jacob_value[:,0])#density * Jacob


    #negative log likelihood for non-censored data
    ign_PDF = np.zeros((nsample,1))
    ign_PDF[np.nonzero(density_yb),0] = np.log(density_yb[np.nonzero(density_yb) ] ) #nonzero
    ign_PDF =  (ign_PDF[:,0] * ( ya > threshold_original ) )
    ign_PDF_sum = np.nansum (-ign_PDF)
    


    #CASE 2 CDF value for censored data (zero precipitation): need to use CDF instead of density
    CDF_censored = norm.cdf ( threshold_transformed, mean, sd)
    #negative log likelihood for censored data
    ign_CDF = -np.log( CDF_censored ) * ( ya <= threshold_original) 
    ign_CDF_sum = np.nansum(ign_CDF )

        
        
    #3. negative log likelihood for all samples together
    neg_loglikelihood = ign_PDF_sum +  ign_CDF_sum
    
    
    # print(par, neg_loglikelihood,'\n')
    return neg_loglikelihood #return negative log likelihood 



def Likelihood_Power_MapParameter(parameters, x , threshold_original):
    #  The likelihood function for power transform fitting using MLE
    #  Firstly, transform the parameters to ensure they are within the bounds by Q.J.'s mirroring method,
    #  then optimize by traditional Nelder-Mead method


    par_mapped=NelderMead_mapping_parameter ( parameters,  [0,-np.inf,-np.inf] ,  [np.inf, np.inf, np.inf] ) #SET THE BOUND
    neg_loglikelihood=Likelihood_Power(par_mapped, x , threshold_original)

    return neg_loglikelihood
 




    