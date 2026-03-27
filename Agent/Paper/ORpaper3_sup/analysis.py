"""
This module contains the core classes and functions for performing financial analysis and simulations.
It provides the parent `analysis_object` class, which serves as the foundation for implementing various
analysis schemes such as PCA, GPS, LH-PCA, and Ledoit-Wolf. The module also includes utility functions
for factor extraction, covariance estimation, and portfolio optimization.

Key Features:
- Implements the `analysis_object` class for shared attributes and methods across analysis schemes.
- Supports factor extraction using PCA and Bai-Ng methods.
- Provides methods for covariance matrix estimation, including Ledoit-Wolf shrinkage.
- Includes portfolio optimization techniques for long-only and long-short portfolios.
- Supports simulation setups for generating synthetic return data with regime-switching models.

Usage:
This module is designed to be imported and extended by other scripts or modules.
"""

import numpy as np
from scipy.sparse import csc_matrix
import csv
import sys
from factor import fm_
from scipy.sparse.linalg import eigsh
from sklearn.preprocessing import normalize
from numpy.linalg import eig
from qpsolvers import solve_qp

def Bai_and_Ng_corr_estimator(R, r_max, method): 
    TT = R.shape[0]
    NN = R.shape[1]
    R_normalized = normalize(R,axis=0)   
    cov_mat = np.dot(R_normalized,R_normalized.T)
    eigval, _ = eig(cov_mat) 
    eigval = abs(eigval) # to fix complex eigenvalues issues, if any
    idx = eigval.argsort()[::-1] # sort eigenvalues in descending orders
    eigval_descending_list = eigval[idx] 
    
    r_max = r_max
    store_list = []

    for k in range(0, r_max):
        V_k = eigval_descending_list[(k+1):].sum()/(NN*TT) # k+1 in the equation
        sigma_hat_sq = eigval_descending_list[(r_max+1):].sum()/(NN*TT) #r_max+1 in the equation
        if method == 1 or method == "PC1" or method == "pc1":
            store_list.append(V_k + k * sigma_hat_sq * (NN+TT)/(NN*TT) * np.log((NN*TT)/(NN+TT)))
        elif method == 2 or method == "IC1" or method == "ic1":
            store_list.append(np.log(V_k) + k * (NN+TT)/(NN*TT) * np.log((NN*TT)/(NN+TT)))
        elif method == 3 or method == "BIC" or method == "bic":
            store_list.append(V_k + k * sigma_hat_sq * ((NN+TT-k)*np.log(NN*TT))/(NN*TT))
        else: 
            print('Error! invalid method type entered') 
            break
            
    Bai_Ng_num_of_factors = np.where(store_list == min(store_list))[0][0] + 1 
    return Bai_Ng_num_of_factors

def pca_factors(R, K_=1, corr_=True):
    ## R = TxN security returns sample matrix
    ## K_ = number of PCA factors to extract

    if(corr_==True):
        R_normalized = normalize(R,axis=0)   # divide each of the N columns by its Euclidean norm
        TxT_corr_mat = np.dot(R_normalized,R_normalized.T)      # correlation matrix without demeaning
        eigen_Val, eigen_Vec = eigsh(TxT_corr_mat, K_, which="LA", maxiter=1e6)
    else:
        TxT_cov_mat = np.dot(R,R.T)      # covariance matrix without normalization
        eigen_Val, eigen_Vec = eigsh(TxT_cov_mat, K_, which="LA", maxiter=1e6)

    PCA_phi = eigen_Vec               # A TxK_ factor returns
    PCA_Z = np.dot(PCA_phi.T, R)      # A K_xN factor exposures

    return PCA_Z, PCA_phi, K_

def get_accuracy_metrics(Zscores):

    ## Remove zeros and Inf to compute bias/Qstat/MRAD
    Zscore2 = Zscores**2 
    Zscore2[Zscore2==0.0] = np.nan
    Zscore2[np.isinf(Zscore2)]=np.nan

    bias = np.sqrt( np.nanmean(Zscore2,axis=0) )
    #Qstat = np.nanmean(Zscore2,axis=0) - np.nanmean(np.log(Zscore2),axis=0)

    ## Calculate MRAD with n days of subperiod
    MRAD = np.zeros((Zscore2.shape[1],Zscore2.shape[2]))
    n = 12
    k = 0
    for tau in range(Zscore2.shape[0]-n+1):
        Zscore2_tau = Zscore2[tau:tau+n,:,:].copy()
        bias_tau = np.sqrt( np.nanmean((Zscore2_tau),axis=0) )
        if(np.sum(np.isnan(bias_tau))):
            continue
        else:
            k = k+1
            MRAD = MRAD + np.abs(bias_tau - 1)
            
    MRAD = MRAD / k 

    return bias, MRAD


def progress_show(job_id, progress, bar_length=25):
    hashes = '*' * int(round(progress * bar_length))
    spaces = '-' * (bar_length - len(hashes))
    sys.stdout.write("\rProgress of job #{0}: [{1}] {2}%  ".format(int(job_id), hashes + spaces, int(round(progress * 100))))
    sys.stdout.flush()


###############################
## The parent analysis class ##
###############################
class analysis_object():
    def __init__(self, data_type, scheme_type):
        ## Assign the attributes to be shared by both children
        self.data_type = data_type
        self.scheme_type = scheme_type

        self.T_L = 1500 
        if(scheme_type == "PCA"):
            self.T_M = 250
            self.covariance_matrix_type = 1  
        elif(scheme_type == "GPS"):
            self.T_M = 250
            self.covariance_matrix_type = 1  
        elif(scheme_type == "LHPCA"):
            self.T_M = 1500
            self.covariance_matrix_type = 1  
        elif(scheme_type == "LW"):
            self.T_M = 250
            self.covariance_matrix_type = 2  
        else:
            raise RuntimeError('Your input is invalid.')

        pf_RCA_num = int(input("\nApplying RCA: 1. with RCA,   2. without RCA \n (1 -- 2) => "))
        self.pf_RCA = pf_RCA_num
        # self.vol_pred_RCA = pf_RCA_num


        if(scheme_type != "LW"):  
            factors_type_num = int(input("\nIndicate factor extraction: 1. Bai and Ng + alpha,     2. Fixed\n (1 -- 2) => "))
            if(factors_type_num == 1):
                self.factors_bai_and_ng = True
                self.alpha_to_add_or_subtract_from_bai_and_ng = int(input("\nIndicate alpha \n => "))
            else:
                self.factors_bai_and_ng = False
                self.factors_fixed = int(input("\nHow many fixed factors? \n => "))

        else:
            self.factors_bai_and_ng = False
            self.factors_fixed = 0  ## Not used in LW scheme

        self.qp_solver = 'cvxopt'

        with open('base_parameter.csv', 'r') as f:
            reader = csv.reader(f, delimiter=',')
            variables = ((row[0], row[1]) for row in reader)

            val_eval = lambda s: eval(s) if not set(s).difference('0123456789. *+-/e') else float(s)
            for row in variables:
                varname = row[0]
                varvalue = val_eval(row[1])
                setattr(self,varname,varvalue)

        self.decay_factor = np.exp(np.log(0.5)/self.T_S) #0.9828

        temp_T_M = 250
        self.decay_weights = np.zeros(self.T_M)
        self.decay_weights[self.T_M-1] = 1.0
        for k in range(temp_T_M-1):
            self.decay_weights[self.T_M-k-2] = self.decay_weights[self.T_M-k-1] * self.decay_factor

    ### First try the closed-form solution.
    ### If it does not work properly, we turn to the numerical optimization.
    def longshort_minvar_closed_form (self, N, V):
        try:
            V_inv = np.linalg.inv(V)
            ones_N = np.ones(N)
            w = np.dot(V_inv, ones_N)/np.dot(ones_N.T, np.dot(V_inv, ones_N))
        except np.linalg.LinAlgError: ## V is Not invertible
            print( "V is not invertible...")
            w = self.longshort_minvar (N, V)  ## numerical optimization

        return w

    ### This function will be called when the closed-form solution cannot be numerically evaluated.
    def longshort_minvar(self, N, V):
        P = V
        A = np.ones(N).T

        q = np.zeros(N)
        b = np.array([1.0])

        x = solve_qp(P, q, G=None, h=None, A=A, b=b, solver=self.qp_solver)

        return np.array(x)


    def simulation_setup(self, N_num, T_total):
        self.N = N_num
        self.T = T_total

        self.dist= "Normal"
        self.df = 0
        self.alpha = 0.0

        #### Markov Regime Switching Model (2 regimes) ###
        self.regime_state = np.zeros(self.T+1)
        if(self.MRS):
            self.p_00 = 0.998
            self.p_11 = 0.992
        else:
            self.p_00 = 1.0
            self.p_11 = 1.0

    def prepare_sim_return_data(self):
        self.fm_0 = fm_(N=self.N, K0=self.K0, K1=self.K1, K2=self.K2, seed=self.seed_fm, uniformCorrelation=self.uniformCorrelation, dist=self.dist, df=self.df, alpha=self.alpha)
        get_daily_vol = lambda x: (x*0.01) / np.sqrt(self.fm_0.num_tdays)

        if(self.K0>0):
            self.fm_0.vol0 = list (map (get_daily_vol, self.fm_0.vol0))
        if(self.K1>0):
            self.fm_0.vol1 = list (map (get_daily_vol, self.fm_0.vol1))
        if(self.K2>0):
            self.fm_0.vol2 = list (map (get_daily_vol, self.fm_0.vol2))
        self.fm_0.vols = list (map (get_daily_vol, self.fm_0.vols))

        self.regime_weight_vec = np.array([2.0, 1.5, 1.5, 1.5, 1.25])  # broad market, broad style, country, industry, idiosyncratic


        self.fm_1 = fm_(N=self.N, K0=self.K0, K1=self.K1, K2=self.K2, seed=self.seed_fm, uniformCorrelation=self.uniformCorrelation, dist=self.dist, df=self.df, alpha=self.alpha)
        get_daily_vol = lambda x: (x*0.01) / np.sqrt(self.fm_1.num_tdays)

        if(self.K0>0):
            self.fm_1.vol0 = list (map (get_daily_vol, self.fm_1.vol0))
        if(self.K1>0):
            self.fm_1.vol1 = list (map (get_daily_vol, self.fm_1.vol1))
        if(self.K2>0):
            self.fm_1.vol2 = list (map (get_daily_vol, self.fm_1.vol2))
        self.fm_1.vols = list (map (get_daily_vol, self.fm_1.vols))

        self.fm_1.vol0 = [element*self.regime_weight_vec[1] for element in self.fm_1.vol0]   # global style vols
        self.fm_1.vol0[0] = self.fm_1.vol0[0]*self.regime_weight_vec[0]   # global market vol
        if(self.K1>0):
            self.fm_1.vol1 = [element*self.regime_weight_vec[2] for element in self.fm_1.vol1]   # country vols
        if(self.K2>0):
            self.fm_1.vol2 = [element*self.regime_weight_vec[3] for element in self.fm_1.vol2]   # industry vols
        self.fm_1.vols = [element*self.regime_weight_vec[4] for element in self.fm_1.vols]   # idiosyncratic vols

        if self.K1+self.K2 > 0:
            self.fm_0.Z = np.concatenate((self.fm_0.Y,self.fm_0.X),axis=0)
            true_sparse_exposures_country = self.fm_0.Z[self.fm_0.K0:self.fm_0.K0+self.fm_0.K1].copy()
            true_sparse_exposures_industry = self.fm_0.Z[self.fm_0.K0+self.fm_0.K1:].copy()

            self.fm_1.Z = np.concatenate((self.fm_1.Y,self.fm_1.X),axis=0)

            self.country_code_list = np.zeros(self.N)
            self.industry_code_list = np.zeros(self.N)

            for idx in range(self.N):
                self.country_code_list[idx] = np.where(true_sparse_exposures_country[:,idx]==1)[0][0]
                self.industry_code_list[idx] = np.where(true_sparse_exposures_industry[:,idx]==1)[0][0]
        else:
            self.fm_0.Z = self.fm_0.Y.copy()
            self.fm_1.Z = self.fm_1.Y.copy()

        if self.K1+self.K2==0:
            self.Factor_true_vols_0 = self.fm_0.vol0
            self.Factor_true_vols_1 = self.fm_1.vol0
        else:
            self.Factor_true_vols_0 = np.concatenate((self.fm_0.vol0,self.fm_0.vol1,self.fm_0.vol2))
            self.Factor_true_vols_1 = np.concatenate((self.fm_1.vol0,self.fm_1.vol1,self.fm_1.vol2))

        self.True_Idio_stock_covariance_0 = np.diag(np.array(self.fm_0.vols)**2)
        self.True_Idio_stock_covariance_1 = np.diag(np.array(self.fm_1.vols)**2)
        self.Factor_true_vars_0 = np.array(self.Factor_true_vols_0)**2
        self.Factor_true_vars_1 = np.array(self.Factor_true_vols_1)**2
        self.Factor_true_covariance_0 = np.zeros((self.K0+self.K1+self.K2,self.K0+self.K1+self.K2))
        self.Factor_true_covariance_1 = np.zeros((self.K0+self.K1+self.K2,self.K0+self.K1+self.K2))
        self.Factor_true_covariance_0[:self.K0,:self.K0] = self.fm_0.add_correlation(self.Factor_true_vars_0[:self.K0],self.uniformCorrelation)
        self.Factor_true_covariance_0[self.K0:,self.K0:] = self.fm_0.add_correlation(self.Factor_true_vars_0[self.K0:],self.uniformCorrelation)
        self.Factor_true_covariance_1[:self.K0,:self.K0] = self.fm_1.add_correlation(self.Factor_true_vars_1[:self.K0],self.uniformCorrelation)
        self.Factor_true_covariance_1[self.K0:,self.K0:] = self.fm_1.add_correlation(self.Factor_true_vars_1[self.K0:],self.uniformCorrelation)

    def generate_sim_return_data(self, seed_return):
        self.R = np.zeros((self.T, self.N))

        offset_seed_return = seed_return + self.seed_return_offset
        
        psi_0, phi_0, eps_0 = self.fm_0.generate_returns (T=self.T, seed=offset_seed_return)
        psi_1, phi_1, eps_1 = self.fm_1.generate_returns (T=self.T, seed=offset_seed_return)

        self.true_psi_0 = psi_0.copy()
        self.true_psi_1 = psi_1.copy()
        self.true_psi = np.zeros((self.T, self.K0))

        self.loading_vol = np.array([0.1, 0.05, 0.05, 0.05])*5
        loading_corr = 0.3
        self.loading_cov = np.ones((self.K0,self.K0))*loading_corr
        for i in range(self.K0):
            for j in range(self.K0):
                if(i==j):
                    self.loading_cov[i,i] = self.loading_vol[i]**2
                else:
                    self.loading_cov[i,j] = self.loading_vol[i]*self.loading_vol[j]*loading_corr

        np.random.seed(offset_seed_return)
        self.random_walk_loading_broad = np.zeros((self.K0, self.N))
        self.random_walk_loading_narrow = np.zeros((self.K1+self.K2, self.N))
        dt = 1/250
        kappa = 0.5

        if(self.K0>0):
            if(self.K1+self.K2>0):
                phi_0 = np.concatenate((psi_0,phi_0),axis=1)
                phi_1 = np.concatenate((psi_1,phi_1),axis=1)
            else:
                phi_0 = psi_0.copy()
                phi_1 = psi_1.copy()

            for idx in range(self.T+1):
                if(self.K1+self.K2>0):
                    if(self.regime_state[idx]==0):
                        theta = np.concatenate((self.fm_0.Y,self.fm_0.X),axis=0)
                    else:
                        theta = np.concatenate((self.fm_1.Y,self.fm_1.X),axis=0)
                else:
                    if(self.regime_state[idx]==0):
                        theta = self.fm_0.Y.copy()
                    else:
                        theta = self.fm_1.Y.copy()
                if(idx==0):
                    self.factor_loading = theta.copy()
                else:
                    correlated_random_walk_broad = np.random.multivariate_normal(mean=np.zeros(self.K0),cov=self.loading_cov,size=self.N)#.T
                    correlated_random_walk_total = np.concatenate((correlated_random_walk_broad.T, np.zeros(np.shape(self.fm_0.X))),axis=0)
                    self.factor_loading += kappa*(theta - self.factor_loading)*dt + correlated_random_walk_total*np.sqrt(dt)
                    
                    if(idx<self.T):
                        if(self.regime_state[idx]==0):
                            self.R[idx,:] = np.dot(phi_0[idx,:],self.factor_loading) + eps_0[idx,:]
                            self.true_psi[idx,:] = self.true_psi_0[idx,:]
                        else:
                            self.R[idx,:] = np.dot(phi_1[idx,:],self.factor_loading) + eps_1[idx,:]
                            self.true_psi[idx,:] = self.true_psi_1[idx,:]

            self.X_true_unsorted = self.factor_loading.copy()
            if(self.regime_state[-1]==0):
                self.True_Factor_stock_covariance = np.dot(np.dot(self.X_true_unsorted.T,self.Factor_true_covariance_0),self.X_true_unsorted)
                self.True_Idio_stock_covariance = self.True_Idio_stock_covariance_0.copy()
            else:
                self.True_Factor_stock_covariance = np.dot(np.dot(self.X_true_unsorted.T,self.Factor_true_covariance_1),self.X_true_unsorted)
                self.True_Idio_stock_covariance = self.True_Idio_stock_covariance_1.copy()

            self.True_stock_covariance = self.True_Factor_stock_covariance + self.True_Idio_stock_covariance

        else:
            raise NotImplementedError("K0 must be strictly positive.")


    def check_narrow_factor(self):
        if self.data_type!="SIM" or self.K1 > 0:
            country_code_max = int(np.max(self.country_code_list))
            country_cnt = np.zeros(country_code_max+1)
            for country_idx in range(country_code_max+1):
                country_cnt[country_idx] = (self.country_code_list==country_idx).sum()
            country_cnt = country_cnt[country_cnt != 0]
            print("Number of stocks in each country: ",country_cnt)

        if self.data_type!="SIM" or self.K2 > 0:
            industry_code_max = int(np.max(self.industry_code_list))
            industry_cnt = np.zeros(industry_code_max+1)
            for industry_idx in range(industry_code_max+1):
                industry_cnt[industry_idx] = (self.industry_code_list==industry_idx).sum()
            industry_cnt = industry_cnt[industry_cnt != 0]
            print("Number of stocks in each industry: ",industry_cnt)

        self.num_country_portfolio = 0
        self.num_industry_portfolio = 0
        self.Number_of_Stocks_in_Factors = 0

    def set_regime_states(self,seed_return):
        np.random.seed(seed_return)
        u = np.random.rand()
        if(self.MRS):
            normal_prob = 0.8
        else:
            normal_prob = 1.0
        if(u<=normal_prob):
            self.regime_state[0] = 0
        else:
            self.regime_state[0] = 1

        for idx in range(self.T):
            u = np.random.rand()
            if(self.regime_state[idx]==0):
                if(u<=self.p_00):
                    self.regime_state[idx+1]=0
                else:
                    self.regime_state[idx+1]=1
            else:
                if(u<=self.p_11):
                    self.regime_state[idx+1]=1
                else:
                    self.regime_state[idx+1]=0

    def estimate_factor_return_M(self, R_Window, R_history):
        factor_X,factor_phi, self.n_components = pca_factors(R=R_Window, K_=self.K_extract, corr_=self.corr_mat)   ## Extract K_extract factor returns and exposures by PCA
        if(self.scheme_type=="GPS"):
            ## Python finds the Market factor indexed at "-1"
            factor_X[-1,:], factor_phi[:,-1] = self.GPS_bias_corrected_X_phi(R_est=R_Window, u1=factor_X[-1,:], market_factor=factor_phi[:,-1])
        factor_return_M = np.dot(factor_phi, factor_X)

        return factor_return_M

    def get_estimated_cov_mat(self, R_Window, factor_return_M, responsive):
        Idio_return = R_Window - factor_return_M

        Factor_covariance_factor_X_phi = self.get_adjusted_covariance(factor_return_M, responsive)
        Factor_covariance_Idio = self.get_adjusted_covariance(Idio_return, responsive)

        Covariance_Estimate = Factor_covariance_factor_X_phi + np.diag(np.diag(Factor_covariance_Idio))

        return Covariance_Estimate, Factor_covariance_factor_X_phi, Factor_covariance_Idio


    def get_adjusted_covariance(self, R_est, responsive=False):
        if(responsive==True):
            if(np.ndim(R_est)==1):
                R_adjusted = np.sqrt(self.decay_weights)*R_est
            else:
                R_adjusted = np.multiply(np.sqrt(self.decay_weights), R_est.T).T
            cov_est = np.dot(R_adjusted.T,R_adjusted)/np.sum(self.decay_weights)
        else:
            cov_est = np.dot(R_est.T,R_est)/R_est.shape[0]

        return cov_est

    def get_optimized_portfolio(self, Covariance_Estimate):
        self.Portfolio_values[0,:] = self.longshort_minvar_closed_form (self.N_window, Covariance_Estimate)


    # An abstract method cannot be used here, and it always throws an error.
    def main_analysis(self, abstract_int):
        raise NotImplementedError("Subclass must implement abstract method")

    def cross_validation(self):
        raise NotImplementedError("Subclass must implement abstract method")



    ## Goldberg-Papanicolaou-Shkolnik correction algorithm adjusted to the setting of Anderson-Kim-Ryu
    def GPS_bias_corrected_X_phi(self, R_est, u1, market_factor):
        # R_est: T x N return matrix
        R_T = R_est.shape[0]
        R_N = R_est.shape[1]
        z = np.ones((R_N,1))/np.sqrt(R_N) # N x 1

        u1 = np.reshape(u1,(1,R_N))
        market_factor = np.reshape(market_factor,(R_T,1))

        market_return_PCA = np.dot(market_factor,u1)
        Residual_PCA = R_est-market_return_PCA
        L_hat = self.get_adjusted_covariance(market_return_PCA)
        Residual_PCA_covariance = self.get_adjusted_covariance(Residual_PCA)
        Delta_hat_diag = np.diag(Residual_PCA_covariance)
        S = L_hat + np.diag(Delta_hat_diag)     # PCA-estimated covariance (single-factor model)

        lambda_1 = self.get_adjusted_covariance(market_factor)

        # GPS correction; see Algorithm 1
        beta_hat = np.sign(np.dot(u1, z))*u1.T # beta_hat is an (N x 1) vector
        sigma_hat_sq = lambda_1*np.linalg.norm(beta_hat, ord = 2)**2 # market_variance
        beta_hat = beta_hat/np.linalg.norm(beta_hat, ord = 2) # normalize beta_hat so beta_hat has L2 length of 1

        self.GPS_mode="data-driven"     # Data-driven method is seemingly better in the presence of narrow factors
        if(self.GPS_mode=="oracle"):
            beta = self.X_true_unsorted[0,:].T  # (N x 1) true market exposures
            beta = beta / np.linalg.norm(beta, ord = 2)
            gamma_beta_z = np.dot(beta.T, z)
            gamma_beta_betahat = np.dot(beta.T, beta_hat)
            gamma_betahat_z = np.dot(beta_hat.T, z)
            rho_hat = (gamma_beta_z - gamma_beta_betahat*gamma_betahat_z)/(gamma_beta_betahat - gamma_beta_z*gamma_betahat_z)
        else:
            Delta_hat_inverse_sqrt = np.diag(1.0/np.sqrt(Delta_hat_diag))
            lambda_1_tilde = sigma_hat_sq * np.linalg.norm(np.dot(Delta_hat_inverse_sqrt, beta_hat), ord = 2)**2
            S_tilde = np.dot(np.dot(Delta_hat_inverse_sqrt, S), Delta_hat_inverse_sqrt)
            z_tilde = np.dot(Delta_hat_inverse_sqrt, z)
            z_tilde = z_tilde/np.linalg.norm(z_tilde, ord = 2)
            beta_tilde = np.dot(Delta_hat_inverse_sqrt, beta_hat)
            beta_tilde = beta_tilde/np.linalg.norm(beta_tilde, ord = 2)
            delta_tilde_sq = (np.trace(S_tilde) - lambda_1_tilde)/(R_N - 1.0 - R_N/R_T)
            c1_hat = R_N/(R_T*lambda_1_tilde - R_N*delta_tilde_sq)
            Psi_hat = np.sqrt(1.0 + delta_tilde_sq*c1_hat)
            gamma_tilde = np.dot(beta_tilde.T, z_tilde)
            rho_hat = Psi_hat*gamma_tilde*(Psi_hat - 1.0/Psi_hat)/(1.0 - (Psi_hat*gamma_tilde)**2)

        gamma = np.dot(beta_hat.T, z)
        beta_rho = (beta_hat + rho_hat*z)/np.sqrt(1.0 + 2.0*rho_hat*gamma + rho_hat**2)
        gamma_rho = np.dot(beta_rho.T, z)
        sigma_rho_hat_sq = sigma_hat_sq*(gamma/gamma_rho)**2

        # GPS adjustment
        regression_vec,Gorbash1,Gorbash2,Gorbash3 = np.linalg.lstsq(beta_rho, R_est.T, rcond = -1)
        market_factor_corrected = regression_vec.T*(gamma/gamma_rho)

        return beta_rho.T, market_factor_corrected[:,0]

    def LW_shrinkage(self, returns: np.array):
        """Shrinks sample covariance matrix towards constant correlation unequal variance matrix.

        Ledoit & Wolf ("Honey, I shrunk the sample covariance matrix", Portfolio Management, 30(2004),
        110-119) optimal asymptotic shrinkage between 0 (sample covariance matrix) and 1 (constant
        sample average correlation unequal sample variance matrix).

        Paper:
        http://www.ledoit.net/honey.pdf

        Matlab code:
        https://www.econ.uzh.ch/dam/jcr:ffffffff-935a-b0d6-ffff-ffffde5e2d4e/covCor.m.zip

        Special thanks to Evgeny Pogrebnyak https://github.com/epogrebnyak

        :param returns:
            t, n - returns of t observations of n shares.
        :return:
            Covariance matrix, sample average correlation, shrinkage.
        """
        t, n = returns.shape
        mean_returns = np.mean(returns, axis=0, keepdims=True)
        returns -= mean_returns
        sample_cov = returns.transpose() @ returns / t

        # sample average correlation
        var = np.diag(sample_cov).reshape(-1, 1)
        sqrt_var = var ** 0.5
        unit_cor_var = sqrt_var * sqrt_var.transpose()
        average_cor = ((sample_cov / unit_cor_var).sum() - n) / n / (n - 1)
        prior = average_cor * unit_cor_var
        np.fill_diagonal(prior, var)

        # pi-hat
        y = returns ** 2
        phi_mat = (y.transpose() @ y) / t - sample_cov ** 2
        phi = phi_mat.sum()

        # rho-hat
        theta_mat = ((returns ** 3).transpose() @ returns) / t - var * sample_cov
        np.fill_diagonal(theta_mat, 0)
        rho = (
            np.diag(phi_mat).sum()
            + average_cor * (1 / sqrt_var @ sqrt_var.transpose() * theta_mat).sum()
        )

        # gamma-hat
        gamma = np.linalg.norm(sample_cov - prior, "fro") ** 2

        # shrinkage constant
        kappa = (phi - rho) / gamma
        shrink = max(0, min(1, kappa / t))

        # estimator
        sigma = shrink * prior + (1 - shrink) * sample_cov

        return sigma