"""
This module defines the `analysis_VR` class, which is a child class of `analysis_object`, specifically designed 
for performing Variance Ratio (VR) analysis. It extends the functionality of the parent class to include methods 
and attributes tailored for VR computations, such as portfolio volatility prediction and covariance estimation.

Key Features:
- Implements Variance Ratio analysis for different portfolio types.
- Supports covariance matrix estimation using both standard and Ledoit-Wolf shrinkage methods.
- Provides functionality for generating synthetic return data and simulating portfolio performance.
- Includes methods for factor extraction and portfolio optimization.
- Utilizes multiprocessing for efficient computation of VR results.

Usage:
This module is intended to be used to perform VR analysis on simulated or empirical data.
"""

from analysis import *
import numpy as np

########################################################
## A child analysis class for Variance ratio analysis ##
########################################################
class analysis_VR(analysis_object):

    def __init__(self, data_type, method_type):
        super().__init__(data_type, method_type)

        self.analysis_type = "VR"
        
        self.num_minvar_portfolio = 1
        self.num_portfolio_for_VR = self.num_minvar_portfolio
        self.num_method_for_VR = 5

        self.number_seed_returns = 1000
        self.pooling_argument = self.number_seed_returns
        self.seed_returns_cnt = 0

        self.simulation_setup(N_num=2000, T_total=2000)
        self.Portfolio_values = np.zeros((self.num_portfolio_for_VR, self.N))
        self.prepare_sim_return_data()
        self.check_narrow_factor()
        self.N_window = self.N
        if(self.K1>0):
            self.country_active = self.country_code_list
        if(self.K2>0):
            self.industry_active = self.industry_code_list

    def main_analysis(self, seed_return):

        self.seed_returns_cnt += 1
        progress = self.seed_returns_cnt/(self.number_seed_returns/self.num_processes)*4
        progress_show(seed_return, progress)

        self.set_regime_states(seed_return)

        VR_results = np.zeros((self.num_portfolio_for_VR, self.num_method_for_VR))

        self.generate_sim_return_data(seed_return)
        R_history = self.R[self.T-self.T_L:,:]
        R_Window = self.R[self.T-self.T_M:,:]

        if(self.factors_bai_and_ng == False):
            self.K_extract = self.factors_fixed
        elif(self.factors_bai_and_ng == True):
            self.K_extract = Bai_and_Ng_corr_estimator(R_Window, r_max = 100, method = 1) + self.alpha_to_add_or_subtract_from_bai_and_ng

        if(self.covariance_matrix_type == 1):

            factor_return_M = self.estimate_factor_return_M(R_Window, R_history)

            if(self.pf_RCA==1):
                pf_responsive = True
            else:
                pf_responsive = False

            pf_Covariance_Estimate, _, _ = self.get_estimated_cov_mat(R_Window, factor_return_M, pf_responsive)
            self.get_optimized_portfolio(pf_Covariance_Estimate)

        elif(self.covariance_matrix_type == 2):
            pf_responsive = False
            pf_Covariance_Estimate = self.LW_shrinkage(R_Window)

            self.get_optimized_portfolio(pf_Covariance_Estimate)

        ones_N = np.ones(self.N)
        V_inv = np.linalg.inv(self.True_stock_covariance)
        V_hat_inv = np.linalg.inv(pf_Covariance_Estimate)
        Volatility_True_Portfolio = 1.0/np.sqrt( np.dot(np.dot(ones_N.T, V_inv), ones_N ))


        for Portfolio_number in range(self.Portfolio_values.shape[0]):

            Portfolio = self.Portfolio_values[Portfolio_number,:].copy()

            Volatility_Actual_Portfolio = np.sqrt(np.dot(Portfolio.T,np.dot(self.True_stock_covariance,Portfolio)))

            if(self.covariance_matrix_type == 1):
                Volatility_Predicted_Portfolio = np.sqrt(np.dot(Portfolio.T,np.dot(pf_Covariance_Estimate,Portfolio)))
            
            else:
                Volatility_Predicted_Portfolio = np.sqrt(self.get_adjusted_covariance(np.dot(R_Window, Portfolio), pf_responsive))

            VR_results[Portfolio_number,:] = (Volatility_Predicted_Portfolio, Volatility_Actual_Portfolio, Volatility_True_Portfolio, np.linalg.norm(V_inv - V_hat_inv), self.K_extract)
            
        return VR_results
