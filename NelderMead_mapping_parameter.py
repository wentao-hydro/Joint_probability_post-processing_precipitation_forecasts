
import numpy as np

def NelderMead_mapping_parameter ( par_original, Params_LowerBound, Params_UpperBound):
    # To map the parameters inside the parameter boundary
    # input: 'par_original' original parameters,
    # output: 'new_par', mapped parameters, within the upper and lower bounds
    # this function follows QJ Wang 's method and Jian Jie's matlab code in https://ww2.mathworks.cn/matlabcentral/fileexchange/73872-mapping-unbounded-parameter-to-bounded-one-for-fminsearch?s_tid=prof_contriblnk
  
  
    nparams=len( par_original)
    new_par=np.zeros_like(par_original)#transformed parameters
    
    for  ipar in range(nparams): 
    
        LowerBound = Params_LowerBound[ipar]
        UpperBound = Params_UpperBound[ipar]

        flag_inf_lowerbound = np.isneginf(LowerBound)
        flag_inf_upperbound = np.isposinf(UpperBound )

        if ( flag_inf_lowerbound & flag_inf_upperbound ):#upper and lower bounds are infinite 
            new_par[ipar] = par_original[ipar] 
        
        elif ( flag_inf_lowerbound & (~flag_inf_upperbound) ):#upper bound is finite
            new_par[ipar] = UpperBound - np.abs(par_original[ipar] - UpperBound )

        elif ( (~flag_inf_lowerbound) & flag_inf_upperbound ):# lower bound is finite
            new_par[ipar] = LowerBound + np.abs(par_original[ipar] - LowerBound )


        else: # both bounds are finite

            # Step 1: checking lower boundary,flip the values
            new_par[ipar] = (2*LowerBound-par_original[ipar]) if (par_original[ipar] < LowerBound)  else   par_original[ipar]
            # Step 2: move
            new_par[ipar] = new_par[ipar] - 2 *(UpperBound-LowerBound) * np.floor( (new_par[ipar]-LowerBound) /(2 *(UpperBound-LowerBound)) ) 
            # Step 3: checking upper boundary,flip the values
            new_par[ipar] = (2 *UpperBound-new_par[ipar]) if (new_par[ipar] > UpperBound)  else  new_par[ipar]
  
     
  
    return new_par 