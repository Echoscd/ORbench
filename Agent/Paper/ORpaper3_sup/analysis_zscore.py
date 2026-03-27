"""
This module defines the `analysis_zscore` class, which is a child class of `analysis_object`, specifically designed 
for performing Z-score-based empirical analysis. It extends the functionality of the parent class to include methods 
and attributes tailored for Z-score computations, such as factor extraction, portfolio evaluation, and statistical testing.

Key Features:
- Implements Z-score-based analysis for empirical datasets.
- Supports factor extraction using PCA and other methods.
- Provides functionality for evaluating portfolio performance based on Z-scores.
- Includes methods for statistical testing and bias evaluation.
- Utilizes multiprocessing for efficient computation of Z-score results.

Usage:
This module is intended to be used to perform Z-score analysis on empirical data.
"""

from analysis import *
import numpy as np
from importlib import reload
import csv
import pandas as pd
from sklearn.covariance import LedoitWolf 

def columns_ge_k_consecutive_nans(df, k):
    return df.isnull().rolling(window=k, axis=0).sum().ge(k).any(axis=0)

#################################################
## A child analysis class for Z-score analysis ##
#################################################
class analysis_zscore(analysis_object):
    def __init__(self, data_type, method_type):
        super().__init__(data_type, method_type)

        self.analysis_type = "Zscore"
        raw_return = []

        if(self.data_type=="EURO"):
            print ("(Loading Euro 2001-2021 data...)")
            data_filename = 'euro_return_2001_2021.csv'

        elif(self.data_type=="CRSP"):
            print ("(Loading CRSP 2001-2021 data...)")
            data_filename = 'crsp_return_2001_2021.csv'

        with open(data_filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            header = next(reader)   # skip the header row
            raw_return.append([[float(m) for m in row] for row in reader])
            self.R = np.array(raw_return[0])[:,1:]  # skip the date column

        self.T = self.R.shape[0]
        self.N = self.R.shape[1]

        print("T = "+str(self.T)+",     N = "+str(self.N))

        self.init_time_point = self.T_L - self.T_M
        self.pooling_argument = self.num_processes
        self.Number_of_Windows = self.T - self.init_time_point - self.T_M
        self.window_num_per_process = int(np.floor(self.Number_of_Windows /self.num_processes))

        self.num_minvar_portfolio = 1 
        self.num_method_for_Zscore = 5 # 1st column: Realized Variance of Estimated MinVar, 2nd column: Realized Z-score of True MinVar, 3rd column:  Target Variance of Estimated MinVar (NaN unless Simulation), 4th: K_extract
        
        self.num_portfolio_for_Zscore = self.num_minvar_portfolio 
        
    def main_analysis(self, window_id):

        progress_show(window_id, 0.0)

        Zscore_results = np.zeros((self.window_num_per_process,self.num_portfolio_for_Zscore,self.num_method_for_Zscore))
        first_starting_point = self.init_time_point+window_id*self.window_num_per_process
        last_starting_point = self.init_time_point+(window_id+1)*self.window_num_per_process-1


        ##############################
        ##  Moving Window analysis  ##
        ##############################
        for Window in range(self.window_num_per_process):
            progress = (Window+1)/self.window_num_per_process
            progress_show(window_id, progress)

            if(self.data_type=="CRSP"):
                R_history = self.R[first_starting_point + Window - self.T_L + self.T_M : first_starting_point + Window + self.T_M,:]
                R_history_and_nextday = self.R[first_starting_point + Window - self.T_L + self.T_M : first_starting_point + Window + self.T_M + 1,:]

                ## moving window issue 1: stocks not traded
                ## exclude any columns that have NAN (i.e. a stock should be traded throughout the moving window + 1)
                ## to be consistent across methods, select active stocks on T_L + 1 days 
                R_history_active = pd.DataFrame(R_history_and_nextday).dropna(axis = 1)
                active_stocks_index = np.array(R_history_active.columns)

                R_history_short = pd.DataFrame(R_history).iloc[(self.T_L - self.T_M):,:] # DHR 220622: T_M by N_Window log returns matrix (last T_M rows from log_returns1 dataframe)
                R_history_short_zero_log_rtn_dropped = R_history_short.loc[:, (R_history_short != 0).any(axis=0)] # DHR 220612: drop zero log return columns (occasionally arises in 1Y PCA and 1Y GPS)
                idx_zero_log_rtn_dropped = np.array(R_history_short_zero_log_rtn_dropped.columns)

                idx_active_and_no_zero_log_rtn = np.intersect1d(active_stocks_index, idx_zero_log_rtn_dropped)
                R_history_active1 = R_history[:,idx_active_and_no_zero_log_rtn]
                R_Window = R_history_active1[(self.T_L - self.T_M):,:]
                R_next = self.R[first_starting_point + Window + self.T_M, idx_active_and_no_zero_log_rtn]
                
                ## To ensure that BN estimator is calculated using 250 moving window regardless of your choice of PCA or LH-PCA
                R_short_Window_for_consistent_BN_estimation = R_history_active1[-250:, :]

            elif(self.data_type=="EURO"):
                R_history = self.R[first_starting_point + Window - self.T_L + self.T_M : first_starting_point + Window + self.T_M,:]
                R_history_and_nextday = self.R[first_starting_point + Window - self.T_L + self.T_M : first_starting_point + Window + self.T_M + 1,:]

                R_history_active = pd.DataFrame(R_history_and_nextday).dropna(axis = 1)
                active_stocks_index = np.array(R_history_active.columns)

                NAN_index = np.where(~columns_ge_k_consecutive_nans(pd.DataFrame(R_history_and_nextday),10))
                active_and_no_consecutive_NAN_index = np.intersect1d(active_stocks_index, NAN_index)

                R_history_and_nextday = R_history_and_nextday[:,active_and_no_consecutive_NAN_index]

                R_history_active1 = R_history[:,active_and_no_consecutive_NAN_index]
                R_Window = R_history_active1[self.T_L - self.T_M:,:]
                R_next = self.R[first_starting_point + Window + self.T_M, active_and_no_consecutive_NAN_index]
                
                ## To ensure that BN estimator is calculated using 250 moving window regardless of your choice of PCA or LH-PCA
                R_short_Window_for_consistent_BN_estimation = R_history_active1[-250:, :]
                
            if(self.factors_bai_and_ng == False):
                self.K_extract = self.factors_fixed
            elif(self.factors_bai_and_ng == True):
                min_K_extract = 3
                # estimated_K_plus_alpha = Bai_and_Ng_corr_estimator(R_short_Window_for_consistent_BN_estimation, r_max = 100, method = 1) + self.alpha_to_add_or_subtract_from_bai_and_ng
                estimated_K_plus_alpha = Bai_and_Ng_corr_estimator(R_Window, r_max = 100, method = 1) + self.alpha_to_add_or_subtract_from_bai_and_ng
                self.K_extract = max(min_K_extract, estimated_K_plus_alpha)

            self.N_window = R_Window.shape[1]
            self.Portfolio_values = np.zeros((self.num_portfolio_for_Zscore,self.N_window))
            
            
            if(self.covariance_matrix_type == 1):

                factor_return_M = self.estimate_factor_return_M(R_Window, R_history_active1)

                if(self.pf_RCA == 1):
                    pf_responsive = True
                else:
                    pf_responsive = False

                pf_Covariance_Estimate, _, _ = self.get_estimated_cov_mat(R_Window, factor_return_M, pf_responsive)
                vol_pred_Covariance_Estimate = pf_Covariance_Estimate

                self.get_optimized_portfolio(pf_Covariance_Estimate)
            
            elif(self.covariance_matrix_type == 2): ## Ledoit-Wolf
                if(self.pf_RCA == 1):
                    pf_responsive = True
                    R_adjusted = np.multiply(np.sqrt(self.decay_weights), R_Window.T).T
                else:
                    pf_responsive = False
                    R_adjusted = R_Window.copy()

                pf_Covariance_Estimate = self.LW_shrinkage(R_adjusted)
                vol_pred_Covariance_Estimate = pf_Covariance_Estimate

                self.get_optimized_portfolio(pf_Covariance_Estimate)

            for Portfolio_number in range(self.num_portfolio_for_Zscore):
                Portfolio_1 = self.Portfolio_values[Portfolio_number,:].copy()
                Portfolio_1_return_next = np.dot(R_next, Portfolio_1)

                Variance_Portfolio = np.dot(Portfolio_1.T,np.dot(vol_pred_Covariance_Estimate,Portfolio_1))
                
                ## Standard Zscore
                if (np.sqrt(Variance_Portfolio) == 0 or np.sqrt(Variance_Portfolio) == np.nan or Portfolio_1_return_next == 0):
                    Zscore_Portfolio = np.nan
                else:
                    Zscore_Portfolio = Portfolio_1_return_next/np.sqrt(Variance_Portfolio)
                Zscore_results[Window,Portfolio_number,:] = (Portfolio_1_return_next, Zscore_Portfolio, np.sqrt(Variance_Portfolio), np.sum(Portfolio_1!=0.0), self.K_extract)

        return Zscore_results
