'''
16/01/26
Code for "Beyond discounted returns: Robust Markov decision processes with average and Blackwell optimality"
Julien Grand-Clement, Marek Petrik, Nicolas Vieille
Author of the code: Julien Grand-Clement (grand-clement@hec.fr)

This file computes the linear regression slopes presented in Table 3
This code requires to have computed all the numerical results with main.py
'''


#####################################################################################
#############################
#############################     Code for computing regression coefficients in the log-log plots
#############################
#####################################################################################


import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


#####################################################################################
#############################
#############################     Hyperparameters
############################# 
#####################################################################################


'''

    Note: Need to be careful how we choose nb points for linear regression o/w we may see conflicting results,
    since the early parts of the curves are quite hectic 
    (e.g. on the healthcare instance, the performance is hectic until at least iteration 100)
    We use the last 900 iterations here.

'''


########### Hyperparameters for MDPs

#### Radius for ellipsoid:
alpha=0.05
### Radius for cube:
scale = 0.05
# Number of states and actions
S = 20

########### Setup for the paper
# number of iterations in the algorithms (to show on figures)
T=1000
T_algo_1 = T*5
# number of points on plots
nb_points = 20
# number of random instances
nb_random_instances = 25    
# exponents for gamma_t = 1-1/(t+1)^rho
rho = 1

# The number of last points to consider in the regression:
nb_points_regression = 900


# Choice of running a smaller instance:
toy_example = 1
if toy_example:
    # number of iterations in the algorithms (to show on figures)
    T=50
    T_algo_1 = T*5
    # number of points on plots
    nb_points = 20
    # number of random instances
    nb_random_instances = 5
    # stopping criterion for computing optimal gain and for adversarial policy iteration
    eps = 0.001
    # exponents for gamma_t = 1-1/(t+1)^rho
    rho = 1
    
    nb_points_regression = 20
    
# choice of MDP instance
list_mdp_instance = ['Machine','Forest','Healthcare','Garnet']
MDP_instance = list_mdp_instance[3]


# choice of uncertainty
uncertainty_choice = 1
list_uncertainty = ['ellipsoid','cube']
uncertainty = list_uncertainty[uncertainty_choice]


# next line to avoid "if then else" in each np.load:
if uncertainty == 'ellipsoid':
    scale = alpha
    
#####################################################################################
#############################
#############################     Plotting figures
############################# 
#####################################################################################

print('MDP instance:',MDP_instance)
print('Uncertainty:',uncertainty)
print('Exponent rho:',rho)
if MDP_instance != 'Garnet':
    
    # Loading curves
    yvec_algo1 = np.load(f'numerical_results/yvec_algo1_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_rho_{rho}.npy')        
    yvec_algo2 = np.load(f'numerical_results/yvec_algo2_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_rho_{rho}.npy')        
    yvec_algo3 = np.load(f'numerical_results/yvec_algo3_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_rho_{rho}.npy')        
        
    # Algo 1:
    x = range(T-nb_points_regression-2,T-2)
    y = yvec_algo1[-nb_points_regression-2:-2]
    x = np.asarray(np.log10(x))
    y = np.asarray(np.log10(y))
    matrix = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(matrix, y, rcond=None)[0]
    print('slope for algo 1:',np.around(m,4),np.around(c,4)) 
    
     # Algo 2:
    x = range(T-nb_points_regression-2,T-2)
    y = yvec_algo2[-nb_points_regression-2:-2]
    x = np.asarray(np.log10(x))
    y = np.asarray(np.log10(y))
    matrix = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(matrix, y, rcond=None)[0]
    print('slope for algo 2:',np.around(m,4),np.around(c,4)) 
    
    # Algo 3:
    x = range(T-nb_points_regression-2,T-2)
    y = yvec_algo3[-nb_points_regression-2:-2]
    x = np.asarray(np.log10(x))
    y = np.asarray(np.log10(y))
    matrix = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(matrix, y, rcond=None)[0]
    print('slope for algo 3:',np.around(m,4),np.around(c,4)) 
    
if MDP_instance == 'Garnet':
    
    
    
    # Loading curves
    yvec_algo1 = np.load(f'numerical_results/yvec_algo_all_1_mean_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_nb_random_instances_{nb_random_instances}_rho_{rho}.npy')        
    yvec_algo2 = np.load(f'numerical_results/yvec_algo_all_2_mean_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_nb_random_instances_{nb_random_instances}_rho_{rho}.npy')        
    yvec_algo3 = np.load(f'numerical_results/yvec_algo_all_3_mean_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_nb_random_instances_{nb_random_instances}_rho_{rho}.npy')        

    # Algo 1:
    x = range(T-nb_points_regression-2,T-2)
    y = yvec_algo1[-nb_points_regression-2:-2]
    x = np.asarray(np.log10(x))
    y = np.asarray(np.log10(y))
    matrix = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(matrix, y, rcond=None)[0]
    print('slope for algo 1:',np.around(m,4),np.around(c,4)) 
    
     # Algo 2:
    x = range(T-nb_points_regression-2,T-2)
    y = yvec_algo2[-nb_points_regression-2:-2]
    x = np.asarray(np.log10(x))
    y = np.asarray(np.log10(y))
    matrix = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(matrix, y, rcond=None)[0]
    print('slope for algo 2:',np.around(m,4),np.around(c,4)) 
    
    # Algo 3:
    x = range(T-nb_points_regression-2,T-2)
    y = yvec_algo3[-nb_points_regression-2:-2]
    x = np.asarray(np.log10(x))
    y = np.asarray(np.log10(y))
    matrix = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(matrix, y, rcond=None)[0]
    print('slope for algo 3:',np.around(m,4),np.around(c,4)) 
