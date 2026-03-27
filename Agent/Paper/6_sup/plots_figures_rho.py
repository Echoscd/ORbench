'''
16/01/26
Code for "Beyond discounted returns: Robust Markov decision processes with average and Blackwell optimality"
Julien Grand-Clement, Marek Petrik, Nicolas Vieille
Author of the code: Julien Grand-Clement (grand-clement@hec.fr)

This file generates Figure 5 and Figure EC.4
This code requires to have computed all the numerical results with main.py
'''

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['text.usetex'] = False


#####################################################################################
#############################
#############################     Hyperparameters
############################# 
#####################################################################################

axis_log = True

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

# Representing some 1/T**p on the plots
plot_T = 0
p = 1

plot_sqrt_T = 0

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

    
    
# list of MDP instance
list_mdp_instance = ['Machine','Forest','Healthcare','Garnet']

# choice of uncertainty
uncertainty_choice = 1
list_uncertainty = ['ellipsoid','cube']
uncertainty = list_uncertainty[uncertainty_choice]

# next line to avoid "if then else" in each np.load:
if uncertainty == 'ellipsoid':
    scale = alpha

plot_all = 1

# choosing ticks for x axis:
choose_ticks = 1
# choosing boundaries for ylim:
choose_ylim = 1

# choice of algo among ['algo1', 'algo3'] (Algo 2 does not depend on rho)
algo = 'algo3'
if algo == 'algo1':  
    algo_garnet = 'algo_all_1'
elif algo == 'algo3':
    algo_garnet = 'algo_all_3'
else:
    raise ValueError('Misspecified algorithm')
    


#####################################################################################
#############################
#############################     Plotting figures
############################# 
#####################################################################################

if plot_all:


    fig, axes = plt.subplots(nrows=1, ncols=4, sharex=False, sharey=False, figsize=(43, 5))
    
    fig.text(0.5, -0.1, 'Number of iterations', ha='center',fontsize=40)
    fig.text(0.07, 0.5, 'Errors', va='center', rotation='vertical',fontsize=40)
    
    
    cols = ['Machine','Forest','Healthcare','Garnet']
    
    for col in range(len(cols)):
        axes[col].set_title(cols[col],fontsize=40)
        if choose_ylim:
            axes[col].set_ylim([0.5*1e-5,1])
            
    x_axis = np.asarray(range(T-1))
    indices = np.logspace(0.0, np.log10(len(x_axis)), num=nb_points)
    indices=indices.astype(int)
    indices[-1]=indices[-1]-2  
    
    ################################################################################
    ################################################
    ################################################    First row 
    
    ################ 0,0
    
    ### Machine
    
    col = 0

    MDP_instance = list_mdp_instance[col]
    
    axes[col].grid()
    
    axes[col].tick_params(labelsize=35)
    axes[col].set_yscale('log')
    axes[col].set_xscale('log')
    
    if choose_ticks:
        axes[col].set_xticks([10,100,1000])
        axes[col].set_yticks([1e0,1e-1,1e-2,1e-3,1e-4,1e-5])
    
    rho = 1/2
    yvec_algo_rho_1 = np.load(f'numerical_results/yvec_{algo}_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_rho_{rho}.npy')        
    rho = 3/4
    yvec_algo_rho_2 = np.load(f'numerical_results/yvec_{algo}_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_rho_{rho}.npy')        
    rho = 7/8
    yvec_algo_rho_3 = np.load(f'numerical_results/yvec_{algo}_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_rho_{rho}.npy')        
    rho = 1
    yvec_algo = np.load(f'numerical_results/yvec_{algo}_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_rho_{rho}.npy')        
    
    axes[col].plot(x_axis[indices], yvec_algo[indices], color='red', marker='*', linestyle=':',linewidth=5, markersize=18)
    axes[col].plot(x_axis[indices], yvec_algo_rho_3[indices], color='green', marker='p', linestyle='--',linewidth=5, markersize=15)
    axes[col].plot(x_axis[indices], yvec_algo_rho_2[indices], color='blue', marker='D', linestyle='-',linewidth=5, markersize=15)
    axes[col].plot(x_axis[indices], yvec_algo_rho_1[indices], color='grey', marker='o', linestyle='--',linewidth=5, markersize=15)

    ################ 0,1

    ### Forest    
    col = 1
    
    MDP_instance = list_mdp_instance[col]

    axes[col].grid()
    
    axes[col].tick_params(labelsize=35)
    axes[col].set_yscale('log')
    axes[col].set_xscale('log')
    
    if choose_ticks:
        axes[col].set_xticks([10,100,1000])
        axes[col].set_yticks([1e0,1e-1,1e-2,1e-3,1e-4,1e-5])
    
    rho = 1/2
    yvec_algo_rho_1 = np.load(f'numerical_results/yvec_{algo}_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_rho_{rho}.npy')        
    rho = 3/4
    yvec_algo_rho_2 = np.load(f'numerical_results/yvec_{algo}_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_rho_{rho}.npy')        
    rho = 7/8
    yvec_algo_rho_3 = np.load(f'numerical_results/yvec_{algo}_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_rho_{rho}.npy')        
    rho = 1
    yvec_algo = np.load(f'numerical_results/yvec_{algo}_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_rho_{rho}.npy')        
    
    axes[col].plot(x_axis[indices], yvec_algo[indices], color='red', marker='*', linestyle=':',linewidth=5, markersize=18)
    axes[col].plot(x_axis[indices], yvec_algo_rho_3[indices], color='green', marker='p', linestyle='--',linewidth=5, markersize=15)
    axes[col].plot(x_axis[indices], yvec_algo_rho_2[indices], color='blue', marker='D', linestyle='-',linewidth=5, markersize=15)
    axes[col].plot(x_axis[indices], yvec_algo_rho_1[indices], color='grey', marker='o', linestyle='--',linewidth=5, markersize=15)

    ################ 0,2

    ### Healthcare
    
    col = 2
    
    MDP_instance = list_mdp_instance[col]
    
    axes[col].grid()
    
    axes[col].tick_params(labelsize=35)
    axes[col].set_yscale('log')
    axes[col].set_xscale('log')
    
    if choose_ticks:
        axes[col].set_xticks([10,100,1000])
        axes[col].set_yticks([1e0,1e-1,1e-2,1e-3,1e-4,1e-5])
    
    rho = 1/2
    yvec_algo_rho_1 = np.load(f'numerical_results/yvec_{algo}_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_rho_{rho}.npy')        
    rho = 3/4
    yvec_algo_rho_2 = np.load(f'numerical_results/yvec_{algo}_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_rho_{rho}.npy')        
    rho = 7/8
    yvec_algo_rho_3 = np.load(f'numerical_results/yvec_{algo}_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_rho_{rho}.npy')        
    rho = 1
    yvec_algo = np.load(f'numerical_results/yvec_{algo}_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_rho_{rho}.npy')        
    
    axes[col].plot(x_axis[indices], yvec_algo[indices], color='red', marker='*', linestyle=':',linewidth=5, markersize=18)
    axes[col].plot(x_axis[indices], yvec_algo_rho_3[indices], color='green', marker='p', linestyle='--',linewidth=5, markersize=15)
    axes[col].plot(x_axis[indices], yvec_algo_rho_2[indices], color='blue', marker='D', linestyle='-',linewidth=5, markersize=15)
    axes[col].plot(x_axis[indices], yvec_algo_rho_1[indices], color='grey', marker='o', linestyle='--',linewidth=5, markersize=15)

    ################ 0,3

    ### Garnet
    
    col = 3
    
    MDP_instance = list_mdp_instance[col]
    
    axes[col].grid()
    
    axes[col].tick_params(labelsize=35)
    axes[col].set_yscale('log')
    axes[col].set_xscale('log')
    
    if choose_ticks:
        axes[col].set_xticks([10,100,1000])
        axes[col].set_yticks([1e0,1e-1,1e-2,1e-3,1e-4,1e-5])
    
    rho = 1
    yvec_algo = np.load(f'numerical_results/yvec_{algo_garnet}_mean_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_nb_random_instances_{nb_random_instances}_rho_{rho}.npy')        
    yvec_algo_ciup = np.load(f'numerical_results/yvec_{algo_garnet}_ciup_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_nb_random_instances_{nb_random_instances}_rho_{rho}.npy')        
    yvec_algo_cidwn = np.load(f'numerical_results/yvec_{algo_garnet}_cidwn_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_nb_random_instances_{nb_random_instances}_rho_{rho}.npy')        
    
    threshold = 1e-10


    
    rho = 1/2
    yvec_algo_rho_1 = np.load(f'numerical_results/yvec_{algo_garnet}_mean_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_nb_random_instances_{nb_random_instances}_rho_{rho}.npy')        
    yvec_algo_rho_1_ciup = np.load(f'numerical_results/yvec_{algo_garnet}_ciup_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_nb_random_instances_{nb_random_instances}_rho_{rho}.npy')        
    yvec_algo_rho_1_cidwn = np.load(f'numerical_results/yvec_{algo_garnet}_cidwn_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_nb_random_instances_{nb_random_instances}_rho_{rho}.npy')        

    rho = 3/4
    yvec_algo_rho_2 = np.load(f'numerical_results/yvec_{algo_garnet}_mean_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_nb_random_instances_{nb_random_instances}_rho_{rho}.npy')        
    yvec_algo_rho_2_ciup = np.load(f'numerical_results/yvec_{algo_garnet}_ciup_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_nb_random_instances_{nb_random_instances}_rho_{rho}.npy')        
    yvec_algo_rho_2_cidwn = np.load(f'numerical_results/yvec_{algo_garnet}_cidwn_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_nb_random_instances_{nb_random_instances}_rho_{rho}.npy')        
    
    rho = 7/8
    yvec_algo_rho_3 = np.load(f'numerical_results/yvec_{algo_garnet}_mean_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_nb_random_instances_{nb_random_instances}_rho_{rho}.npy')        
    yvec_algo_rho_3_ciup = np.load(f'numerical_results/yvec_{algo_garnet}_ciup_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_nb_random_instances_{nb_random_instances}_rho_{rho}.npy')        
    yvec_algo_rho_3_cidwn = np.load(f'numerical_results/yvec_{algo_garnet}_cidwn_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_nb_random_instances_{nb_random_instances}_rho_{rho}.npy')        



    axes[col].plot(x_axis[indices], yvec_algo[indices], color='red', marker='*', linestyle=':',linewidth=5, markersize=18,label = r'$\rho=1$')
    axes[col].plot(x_axis[indices], yvec_algo_rho_3[indices], color='green', marker='p', linestyle='--',linewidth=5, markersize=15,label = r'$\rho=7/8$')
    axes[col].plot(x_axis[indices], yvec_algo_rho_2[indices], color='blue', marker='D', linestyle='-',linewidth=5, markersize=15,label = r'$\rho=3/4$')
    axes[col].plot(x_axis[indices], yvec_algo_rho_1[indices], color='grey', marker='o', linestyle='--',linewidth=5, markersize=15,label = r'$\rho=1/2$')

    threshold = 1e-10
    axes[col].fill_between(x_axis[indices], np.maximum(yvec_algo_cidwn[indices],threshold),yvec_algo_ciup[indices], color='red', alpha=0.1)
    axes[col].fill_between(x_axis[indices], np.maximum(yvec_algo_rho_3_cidwn[indices],threshold),yvec_algo_rho_3_ciup[indices], color='green', alpha=0.1)
    axes[col].fill_between(x_axis[indices], np.maximum(yvec_algo_rho_2_cidwn[indices],threshold),yvec_algo_rho_2_ciup[indices], color='blue', alpha=0.1)
    axes[col].fill_between(x_axis[indices], np.maximum(yvec_algo_rho_1_cidwn[indices],threshold),yvec_algo_rho_1_ciup[indices], color='grey', alpha=0.1)

    ################################################################################
    ################################################
    ################################################    Axis parameter
        
    
    fig.legend(ncol=8,loc="upper center",bbox_to_anchor=(0.5,1.3,0,0),fontsize=40)
    fig.savefig(f"figures/plots_all_instances_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_nb_random_instances_{nb_random_instances}_rho_{rho}_per_instance_algo_{algo}.pdf", bbox_inches='tight')
    plt.show()
