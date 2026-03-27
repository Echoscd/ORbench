"""
This script serves as the main module for running various financial analysis schemes and simulations.
It allows the user to select different scheme types (e.g., PCA, GPS, LH-PCA, Ledoit-Wolf) and analysis types 
(e.g., Empirical Analysis using Z-scores or Simulation using Volatility Ratios). 

The script interacts with external modules (`analysis_zscore`, `analysis_VR`) to perform computations 
and outputs results in the form of numerical arrays or saved files. It supports multiprocessing for 
efficient computation and handles user inputs to customize the analysis.

Key Features:
- Scheme selection: PCA, GPS, LH-PCA, Ledoit-Wolf.
- Analysis types: Z-score-based empirical analysis or volatility ratio simulations.
- Data options: CRSP or Euro datasets for empirical analysis.
- Multiprocessing support for parallelized computations.
- Outputs results as CSV files for further analysis.

Usage:
Run the script directly and follow the prompts to specify the scheme type, analysis type, and data options.
"""

import warnings
warnings.filterwarnings(action='ignore')

import numpy as np
import time
import datetime
from datetime import timedelta
from multiprocessing import set_start_method, get_context
import analysis
import analysis_zscore
import analysis_VR
import analysis
from analysis_zscore import analysis_zscore
from analysis_VR import analysis_VR

##########################
##		Main module		##
##########################

if __name__ == '__main__':
    version = 'OPRE_Final'

    output_str = ""
    output_str = "".join(( output_str, "Run Date: "+str(datetime.date.today())+"\n" ))

    scheme_type_num = int(input("Specify Scheme type: 1. PCA (1Y),   2. GPS (1Y),   3. LH-PCA (6Y), 4. Lediot-Wolf (2004)\n (1 -- 4) => "))
    if(scheme_type_num == 1):
        print ("*** Scheme type: PCA (1Y) ***\n")
        scheme_type = "PCA"
    elif(scheme_type_num == 2):
        print ("*** Scheme type: GPS (1Y) ***\n")
        scheme_type = "GPS"
    elif(scheme_type_num == 3):
        print ("*** Scheme type: LH-PCA (6Y) ***\n")
        scheme_type = "LHPCA"
    elif(scheme_type_num == 4):
        print ("*** Scheme type: Lediot-Wolf (2004) ***\n")
        scheme_type = "LW"
    else:
        raise RuntimeError('Scheme type is invalid.')

    analysis_type_num = int(input("Specify Analysis type: 1. Empirical Analysis (Z-score),   2. Simulation (Volatility Ratio)\n (1 -- 2) => "))
    if(analysis_type_num==1):
        print("*** Analysis type: Empirical Analysis (Z-score) ***\n")
        data_type_num = int(input("Specify Data type: 1. CRSP 2001-2021,   2. Euro 2001-2021\n (1 -- 2) => "))
        if(data_type_num == 1):
            print ("*** Data type: CRSP 2001-2021 ***\n")
            obj = analysis_zscore("CRSP", scheme_type)
            output_save_name = "CRSP"
        elif(data_type_num == 2):
            print ("*** Data type: Euro 2001-2021 ***\n")
            obj = analysis_zscore("EURO", scheme_type)
            output_save_name = "EURO"
        else:
            raise RuntimeError('Data type is invalid.')

        Ratios = np.zeros((obj.num_processes*obj.window_num_per_process,obj.num_portfolio_for_Zscore,obj.num_method_for_Zscore))

    elif(analysis_type_num==2):
        print("*** Analysis type: Simulation (Volatility Ratio) ***\n")
        obj = analysis_VR("SIM", scheme_type)
        Ratios = np.zeros((obj.number_seed_returns,obj.num_portfolio_for_VR,obj.num_method_for_VR))

    else:
        raise RuntimeError('Analysis type is invalid.')

    set_start_method("spawn")
    with get_context("spawn").Pool(processes=obj.num_processes) as pool:
        results_vec = pool.map(obj.main_analysis, range (obj.pooling_argument))
        pool.close()
        pool.join()

    if(obj.analysis_type=="Zscore"):
        for i in range (obj.pooling_argument):
            Ratios[i*obj.window_num_per_process:(i+1)*obj.window_num_per_process,:,:] = results_vec[i][:]

    elif(obj.analysis_type=="VR"):
        for i in range (obj.pooling_argument):
            Ratios[i,:,:] = results_vec[i][:]

    temp_title = obj.scheme_type
    temp_title += '_method='+str(obj.pf_RCA)+'_'
    temp_title += input('Specify additional title info (e.g., tuning parameters and/or RCA options) to append to output file names: ')

    if(obj.analysis_type=="Zscore"):
        if(obj.output_file==1):
            np.savetxt(output_save_name+'_RV_'+temp_title+'.csv', Ratios[:,:,0], delimiter=",")
            np.savetxt(output_save_name+'_Zscore_'+temp_title+'.csv', Ratios[:,:,1], delimiter=",")
            np.savetxt(output_save_name+'_Pred_vol_'+temp_title+'.csv', Ratios[:,:,2], delimiter=",") 
            np.savetxt(output_save_name+'_num_stock_'+temp_title+'.csv', Ratios[:,:,3], delimiter=",")
            np.savetxt(output_save_name+'_num_PCA_eigenfactors_'+temp_title+'.csv', Ratios[:,:,4], delimiter=",")

    elif(obj.analysis_type=="VR"):
        if(obj.output_file==1):
            np.savetxt('VR_predicted_vol_'+temp_title+'.csv', Ratios[:,:,0], delimiter=",")
            np.savetxt('VR_actual_vol_'+temp_title+'.csv', Ratios[:,:,1], delimiter=",")
            np.savetxt('VR_true_vol_'+temp_title+'.csv', Ratios[:,:,2], delimiter=",")
            np.savetxt('VR_Cov_Norm_'+temp_title+'.csv', Ratios[:,:,3], delimiter=",")
            np.savetxt('VR_num_PCA_eigenfactors_'+temp_title+'.csv', Ratios[:,:,4], delimiter=",")
