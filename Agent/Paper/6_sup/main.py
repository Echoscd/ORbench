'''
16/01/26
Code for "Beyond discounted returns: Robust Markov decision processes with average and Blackwell optimality"
Julien Grand-Clement, Marek Petrik, Nicolas Vieille
Author of the code: Julien Grand-Clement (grand-clement@hec.fr)
'''

#####################################################################################
#############################
#############################     Importing MDP Functions
#############################
#####################################################################################

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


# importing MDP instances
from aux_functions import Machine_MDP,Healthcare_MDP,Forest_MDP, Garnet
# Hypercube functions (Linfinity distance)
from aux_functions import form_uncertainty_hypercube
from aux_functions import run_algo_VI_increasing_horizon_hypercube, run_algo_VI_increasing_discounts_hypercube
from aux_functions import run_algo_limit_discounted_returns_hypercube
# Ellipsoid functions
from aux_functions import form_uncertainty_ellipsoid
from aux_functions import run_algo_VI_increasing_horizon_ellipsoid, run_algo_VI_increasing_discounts_ellipsoid
from aux_functions import run_algo_limit_discounted_returns_ellipsoid


#####################################################################################
#############################
#############################     Choice of simulations to run
#############################
#####################################################################################

# computing the optimal gain for each instance and save it for comparisons:
compute_instance_dependent_optimal_gain = 1

# Running each algorithms for T iterations 
# Here, the gain obtained by Algorithm 1 is used for reference as optimal gain
compute_error_to_optimal_gain = 1

#####################################################################################
#############################
#############################     Choice of experiments
#############################
#####################################################################################

# Choosing the type of uncertainty
uncertainty_ellipsoid = 1
uncertainty_cube = 1

axis_log = True


#####################################################################################
#############################
#############################     Hyperparameters for MDPs
#############################
#####################################################################################


#### Radius for ellipsoid:
alpha=0.05
### Radius for cube:
scale = 0.05
# Number of states and actions
S = 20
# Initial distribution: 
p0 = (1/S)*np.ones(S)



# number of iterations in the algorithms (to show on figures)
T=1000
T_algo_1 = T*5
# number of points on plots
nb_points = 20
# number of random instances (for Garnet MDPs)
nb_random_instances = 25
# stopping criterion for computing optimal gain and for adversarial policy iteration
eps = 0.001
# exponents for gamma_t = 1-1/(t+1)^rho
rho = 1

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
    
    
# choice of MDP instance
list_mdp_instance = ['Machine','Forest','Healthcare','Garnet']
MDP_instance = list_mdp_instance[0]

print('MDP instance:',MDP_instance)


if MDP_instance == 'Machine':
    P_nom,rew=Machine_MDP(S)
    rew = rew/np.max(rew)
    A = 2
elif MDP_instance == 'Forest':
    P_nom,rew=Forest_MDP(S)
    rew = rew/np.max(rew)
    A = 2
elif MDP_instance == 'Healthcare':
    P_nom,rew=Healthcare_MDP(S)
    rew = rew/np.max(rew)
    A = 3
elif MDP_instance == 'Garnet':
    seed = 2
    A,N_B = 5,10
    P_nom=Garnet(S,N_B,A,seed)
    np.random.seed(0)
    rew = np.random.rand(S,A)
    rew = rew/np.max(rew)
    
#####################################################################################
#############################
#############################     Running the algorithms
#############################
#####################################################################################

if compute_instance_dependent_optimal_gain:
    
    ''' Computing an estimate of the optimal gain
    by running Algorithm 1 for a large number of periods (=5*T)
    See Section 5.4.1 for the empirical setup
    '''
    
    if MDP_instance != 'Garnet':
        
        if uncertainty_ellipsoid:
            
            uncertainty = 'ellipsoid'
            print('Uncertainty:',uncertainty)
            radius = form_uncertainty_ellipsoid(P_nom,alpha,S,A)
            
            
            print('Running Algo 1 (limit of discounted returns)...')
            result_limit_discounted_return, time_limit_discounted_return = run_algo_limit_discounted_returns_ellipsoid(rew,S,A,T_algo_1,P_nom,radius,p0,eps,rho)

            optimal_gain = result_limit_discounted_return[-1]

            np.save(f'numerical_results/optimal_gain_{MDP_instance}_{uncertainty}_radius_{alpha}_states_{S}_T_algo1_{T_algo_1}_rho_{rho}.npy',optimal_gain)
            
        if uncertainty_cube:
                    
            uncertainty = 'cube'
            print('Uncertainty:',uncertainty)
            
            scaling_factor_up = scale
            scaling_factor_down = scale
        
            P_up,P_down = form_uncertainty_hypercube(P_nom,scaling_factor_up,scaling_factor_down,S,A)
    
            print('Running Algo 1 (limit of discounted returns)...')
            result_limit_discounted_return, time_limit_discounted_return = run_algo_limit_discounted_returns_hypercube(rew,S,A,T_algo_1,P_nom,P_up,P_down,p0,eps,rho)
   
            optimal_gain = result_limit_discounted_return[-1]
  
            np.save(f'numerical_results/optimal_gain_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_algo1_{T_algo_1}_rho_{rho}.npy',optimal_gain)

            
    elif MDP_instance == 'Garnet':
        
        optimal_gain_vec = np.zeros(nb_random_instances)
        
        if uncertainty_ellipsoid:
            
            uncertainty = 'ellipsoid'
            print('Uncertainty:',uncertainty)
            
            for seed in range(nb_random_instances):
                print('Instance number ...',seed)
                
                A,N_B = 5,10
                P_nom=Garnet(S,N_B,A,seed)
                np.random.seed(seed)
                rew = np.random.rand(S,A)
                rew = rew/np.max(rew)
                
                radius = form_uncertainty_ellipsoid(P_nom,alpha,S,A)
            
            
                print('Running Algo 1 (limit of discounted returns)...')
                result_limit_discounted_return, time_limit_discounted_return = run_algo_limit_discounted_returns_ellipsoid(rew,S,A,T_algo_1,P_nom,radius,p0,eps,rho)
                
                optimal_gain_vec[seed] = result_limit_discounted_return[-1]
                np.save(f'numerical_results/optimal_gain_{MDP_instance}_{uncertainty}_radius_{alpha}_states_{S}_T_algo1_{T_algo_1}_rho_{rho}_seed_{seed}.npy',optimal_gain_vec[seed])

        if uncertainty_cube:
            
            uncertainty = 'cube'
            print('Uncertainty:',uncertainty)
            
            for seed in range(nb_random_instances):
                
                print('Instance number ...',seed)
                
                A,N_B = 5,10
                P_nom=Garnet(S,N_B,A,seed)
                np.random.seed(seed)
                rew = np.random.rand(S,A)
                rew = rew/np.max(rew)
     
                scaling_factor_up = scale
                scaling_factor_down = scale
            
                P_up,P_down = form_uncertainty_hypercube(P_nom,scaling_factor_up,scaling_factor_down,S,A)
        
        
                print('Running Algo 1 (limit of discounted returns)...')
                result_limit_discounted_return, time_limit_discounted_return = run_algo_limit_discounted_returns_hypercube(rew,S,A,T_algo_1,P_nom,P_up,P_down,p0,eps,rho)
        
                optimal_gain_vec[seed] = result_limit_discounted_return[-1]
                np.save(f'numerical_results/optimal_gain_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_algo1_{T_algo_1}_rho_{rho}_seed_{seed}.npy',optimal_gain_vec[seed])

# For next piece of code: we need to have already computed/saved the optimal gains, i.e. we need to execute
# compute_instance_dependent_optimal_gain first
if compute_error_to_optimal_gain:
    
    '''
    Run each algorithm (Algos 1-2-3) for T periods
    Plot the difference to the optimal gain for each algorithm
    See Section 5.4.1 for the empirical setup
    '''
    if MDP_instance != 'Garnet':
        
        if uncertainty_ellipsoid:
            
            uncertainty = 'ellipsoid'
            print('Uncertainty:',uncertainty)
            radius = form_uncertainty_ellipsoid(P_nom,alpha,S,A)
            
            
            print('Running Algo 1 (limit of discounted returns)...')
            result_limit_discounted_return, time_limit_discounted_return = run_algo_limit_discounted_returns_ellipsoid(rew,S,A,T,P_nom,radius,p0,eps,rho)
            
            print('Running Algo 2 (VI - increasing horizon)...')
            result_algo_increasing_horizon,time_increasing_horizon = run_algo_VI_increasing_horizon_ellipsoid(rew,P_nom,S,A,T,radius,p0)
            print('Running Algo 3 (VI - increasing discount factor)...')
            result_algo_increasing_discounts,time_increasing_discounts = run_algo_VI_increasing_discounts_ellipsoid(rew,P_nom,S,A,T,radius,p0,rho)
        
            optimal_gain = np.load(f'numerical_results/optimal_gain_{MDP_instance}_{uncertainty}_radius_{alpha}_states_{S}_T_algo1_{T_algo_1}_rho_{rho}.npy')

            yvec_algo1 = abs(result_limit_discounted_return[0:T]-optimal_gain)
            yvec_algo2 = abs(result_algo_increasing_horizon-optimal_gain)
            yvec_algo3 = abs(result_algo_increasing_discounts-optimal_gain)
            
            x_axis = np.asarray(range(T-1))
            indices = np.logspace(0.0, np.log10(len(x_axis)), num=nb_points)
            indices=indices.astype(int)
            indices[-1]=indices[-1]-2  
            
            fig, ax = plt.subplots()
            
            ax.tick_params(labelsize=25)
            
            if axis_log:
                ax.set_yscale('log')
                ax.set_xscale('log')
            
            ax.plot(x_axis[indices], yvec_algo1[indices], color='red', marker='*', linestyle=':',linewidth=2, markersize=10,label = 'Algo. 1')
            ax.plot(x_axis[indices], yvec_algo2[indices], color='green', marker='p', linestyle='--',linewidth=2, markersize=10, label = 'Algo. 2')
            ax.plot(x_axis[indices], yvec_algo3[indices], color='blue', marker='D', linestyle='-',linewidth=2, markersize=10,label = 'Algo. 3')
    
            ax.set_xlabel('Number of iterations',fontsize=20)
            ax.set_ylabel('Error',fontsize=20)
            ax.tick_params(axis='both', which='major', labelsize=20)
    
            ax.grid()
    
            fig.legend(ncol=3,loc="upper center",bbox_to_anchor=(0.5,1.15,0,0),fontsize=20)
            
            np.save(f'numerical_results/yvec_algo1_{MDP_instance}_{uncertainty}_radius_{alpha}_states_{S}_T_{T}_T_algo1_{T_algo_1}_rho_{rho}.npy',yvec_algo1)
            np.save(f'numerical_results/yvec_algo2_{MDP_instance}_{uncertainty}_radius_{alpha}_states_{S}_T_{T}_T_algo1_{T_algo_1}_rho_{rho}.npy',yvec_algo2)
            np.save(f'numerical_results/yvec_algo3_{MDP_instance}_{uncertainty}_radius_{alpha}_states_{S}_T_{T}_T_algo1_{T_algo_1}_rho_{rho}.npy',yvec_algo3)

            fig.savefig(f"figures/errors_{MDP_instance}_{uncertainty}_radius_{alpha}_states_{S}_T_{T}_T_algo1_{T_algo_1}_axis_log_{axis_log}_rho_{rho}.pdf", bbox_inches='tight')
            
            plt.show()
            
        if uncertainty_cube:
                    
            uncertainty = 'cube'
            print('Uncertainty:',uncertainty)
            
            scaling_factor_up = scale
            scaling_factor_down = scale
        
            P_up,P_down = form_uncertainty_hypercube(P_nom,scaling_factor_up,scaling_factor_down,S,A)
    
    
            print('Running Algo 1 (limit of discounted returns)...')
            result_limit_discounted_return, time_limit_discounted_return = run_algo_limit_discounted_returns_hypercube(rew,S,A,T,P_nom,P_up,P_down,p0,eps,rho)
    
            print('Running Algo 2 (VI - increasing horizon)...')
            result_algo_increasing_horizon,time_increasing_horizon = run_algo_VI_increasing_horizon_hypercube(rew,S,A,T,P_up,P_down,p0)
            print('Running Algo 3 (VI - increasing discount factor)...')
            result_algo_increasing_discounts,time_increasing_discounts = run_algo_VI_increasing_discounts_hypercube(rew,S,A,T,P_up,P_down,p0,rho)

            optimal_gain = np.load(f'numerical_results/optimal_gain_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_algo1_{T_algo_1}_rho_{rho}.npy')
            yvec_algo1 = abs(result_limit_discounted_return[:T]-optimal_gain)
            yvec_algo2 = abs(result_algo_increasing_horizon-optimal_gain)
            yvec_algo3 = abs(result_algo_increasing_discounts-optimal_gain)
            
            x_axis = np.asarray(range(T-1))
            indices = np.logspace(0.0, np.log10(len(x_axis)), num=nb_points)
            indices=indices.astype(int)
            indices[-1]=indices[-1]-2  
            
            fig, ax = plt.subplots()
            
            ax.tick_params(labelsize=25)
            
            if axis_log:
                ax.set_yscale('log')
                ax.set_xscale('log')
            
            ax.plot(x_axis[indices], yvec_algo1[indices], color='red', marker='*', linestyle=':',linewidth=2, markersize=10,label = 'Algo. 1')
            ax.plot(x_axis[indices], yvec_algo2[indices], color='green', marker='p', linestyle='--',linewidth=2, markersize=10, label = 'Algo. 2')
            ax.plot(x_axis[indices], yvec_algo3[indices], color='blue', marker='D', linestyle='-',linewidth=2, markersize=10,label = 'Algo. 3')
    
            ax.set_xlabel('Number of iterations',fontsize=20)
            ax.set_ylabel('Error',fontsize=20)
            ax.tick_params(axis='both', which='major', labelsize=20)
    
            ax.grid()
    
            fig.legend(ncol=3,loc="upper center",bbox_to_anchor=(0.5,1.15,0,0),fontsize=20)
            
            np.save(f'numerical_results/yvec_algo1_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_rho_{rho}.npy',yvec_algo1)
            np.save(f'numerical_results/yvec_algo2_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_rho_{rho}.npy',yvec_algo2)
            np.save(f'numerical_results/yvec_algo3_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_rho_{rho}.npy',yvec_algo3)

            fig.savefig(f"figures/errors_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_axis_log_{axis_log}_rho_{rho}.pdf", bbox_inches='tight')
            
            plt.show()
            
    elif MDP_instance == 'Garnet':
        
        yvec_algo_all_1 = np.zeros((nb_random_instances,T))
        yvec_algo_all_2 = np.zeros((nb_random_instances,T))
        yvec_algo_all_3 = np.zeros((nb_random_instances,T))
        
        
        if uncertainty_ellipsoid:
            
            uncertainty = 'ellipsoid'
            print('Uncertainty:',uncertainty)
            
            for seed in range(nb_random_instances):
                print('Instance number ...',seed)
                
                optimal_gain = np.load(f'numerical_results/optimal_gain_{MDP_instance}_{uncertainty}_radius_{alpha}_states_{S}_T_algo1_{T_algo_1}_rho_{rho}_seed_{seed}.npy')

                
                A,N_B = 5,10
                P_nom=Garnet(S,N_B,A,seed)
                np.random.seed(seed)
                rew = np.random.rand(S,A)
                rew = rew/np.max(rew)
                
                radius = form_uncertainty_ellipsoid(P_nom,alpha,S,A)
            
            
                print('Running Algo 1 (limit of discounted returns)...')
                result_limit_discounted_return, time_limit_discounted_return = run_algo_limit_discounted_returns_ellipsoid(rew,S,A,T,P_nom,radius,p0,eps,rho)
                
                print('Running Algo 2 (VI - increasing horizon)...')
                result_algo_increasing_horizon,time_increasing_horizon = run_algo_VI_increasing_horizon_ellipsoid(rew,P_nom,S,A,T,radius,p0)
                print('Running Algo 3 (VI - increasing discount factor)...')
                result_algo_increasing_discounts,time_increasing_discounts = run_algo_VI_increasing_discounts_ellipsoid(rew,P_nom,S,A,T,radius,p0,rho)
            
                yvec_algo_all_1[seed] = abs(result_limit_discounted_return[:T]-optimal_gain)
                yvec_algo_all_2[seed] = abs(result_algo_increasing_horizon-optimal_gain)
                yvec_algo_all_3[seed] = abs(result_algo_increasing_discounts-optimal_gain)
                
            yvec_algo_all_1_mean = np.mean(yvec_algo_all_1,axis = 0)
            yvec_algo_all_2_mean = np.mean(yvec_algo_all_2,axis = 0)
            yvec_algo_all_3_mean = np.mean(yvec_algo_all_3,axis = 0)
            
            yvec_algo_all_1_std = np.std(yvec_algo_all_1,axis = 0)
            yvec_algo_all_2_std = np.std(yvec_algo_all_2,axis = 0)
            yvec_algo_all_3_std = np.std(yvec_algo_all_3,axis = 0)      
            
            yvec_algo_all_1_ciup = yvec_algo_all_1_mean + (1.96/np.sqrt(nb_random_instances))*yvec_algo_all_1_std
            yvec_algo_all_2_ciup = yvec_algo_all_2_mean + (1.96/np.sqrt(nb_random_instances))*yvec_algo_all_2_std
            yvec_algo_all_3_ciup = yvec_algo_all_3_mean + (1.96/np.sqrt(nb_random_instances))*yvec_algo_all_3_std

            yvec_algo_all_1_cidwn = yvec_algo_all_1_mean - (1.96/np.sqrt(nb_random_instances))*yvec_algo_all_1_std
            yvec_algo_all_2_cidwn = yvec_algo_all_2_mean - (1.96/np.sqrt(nb_random_instances))*yvec_algo_all_2_std
            yvec_algo_all_3_cidwn = yvec_algo_all_3_mean - (1.96/np.sqrt(nb_random_instances))*yvec_algo_all_3_std
        
        
            x_axis = np.asarray(range(T-1))
            indices = np.logspace(0.0, np.log10(len(x_axis)), num=nb_points)
            indices=indices.astype(int)
            indices[-1]=indices[-1]-2  
            
            fig, ax = plt.subplots()
            
            ax.tick_params(labelsize=25)
            
            if axis_log:
                ax.set_yscale('log')
                ax.set_xscale('log')
            
            ax.plot(x_axis[indices], yvec_algo_all_1_mean[indices], color='red', marker='*', linestyle=':',linewidth=2, markersize=10,label = 'Algo. 1')
            ax.plot(x_axis[indices], yvec_algo_all_2_mean[indices], color='green', marker='p', linestyle='--',linewidth=2, markersize=10, label = 'Algo. 2')
            ax.plot(x_axis[indices], yvec_algo_all_3_mean[indices], color='blue', marker='D', linestyle='-',linewidth=2, markersize=10,label = 'Algo. 3')
            
            threshold = 1e-10
            ax.fill_between(x_axis[indices], np.maximum(yvec_algo_all_1_cidwn[indices],threshold),yvec_algo_all_1_ciup[indices], color='red', alpha=0.1)
            ax.fill_between(x_axis[indices], np.maximum(yvec_algo_all_2_cidwn[indices],threshold),yvec_algo_all_2_ciup[indices], color='green', alpha=0.1)
            ax.fill_between(x_axis[indices], np.maximum(yvec_algo_all_3_cidwn[indices],threshold),yvec_algo_all_3_ciup[indices], color='blue', alpha=0.1)


    
            ax.set_xlabel('Number of iterations',fontsize=20)
            ax.set_ylabel('Error',fontsize=20)
            ax.tick_params(axis='both', which='major', labelsize=20)
    
            ax.grid()
    
            fig.legend(ncol=3,loc="upper center",bbox_to_anchor=(0.5,1.15,0,0),fontsize=20)
            
            
            np.save(f'numerical_results/yvec_algo_all_1_mean_{MDP_instance}_{uncertainty}_radius_{alpha}_states_{S}_T_{T}_T_algo1_{T_algo_1}_nb_random_instances_{nb_random_instances}_rho_{rho}.npy',yvec_algo_all_1_mean)
            np.save(f'numerical_results/yvec_algo_all_2_mean_{MDP_instance}_{uncertainty}_radius_{alpha}_states_{S}_T_{T}_T_algo1_{T_algo_1}_nb_random_instances_{nb_random_instances}_rho_{rho}.npy',yvec_algo_all_2_mean)
            np.save(f'numerical_results/yvec_algo_all_3_mean_{MDP_instance}_{uncertainty}_radius_{alpha}_states_{S}_T_{T}_T_algo1_{T_algo_1}_nb_random_instances_{nb_random_instances}_rho_{rho}.npy',yvec_algo_all_3_mean)

            np.save(f'numerical_results/yvec_algo_all_1_cidwn_{MDP_instance}_{uncertainty}_radius_{alpha}_states_{S}_T_{T}_T_algo1_{T_algo_1}_nb_random_instances_{nb_random_instances}_rho_{rho}.npy',yvec_algo_all_1_cidwn)
            np.save(f'numerical_results/yvec_algo_all_2_cidwn_{MDP_instance}_{uncertainty}_radius_{alpha}_states_{S}_T_{T}_T_algo1_{T_algo_1}_nb_random_instances_{nb_random_instances}_rho_{rho}.npy',yvec_algo_all_2_cidwn)
            np.save(f'numerical_results/yvec_algo_all_3_cidwn_{MDP_instance}_{uncertainty}_radius_{alpha}_states_{S}_T_{T}_T_algo1_{T_algo_1}_nb_random_instances_{nb_random_instances}_rho_{rho}.npy',yvec_algo_all_3_cidwn)
            
            np.save(f'numerical_results/yvec_algo_all_1_ciup_{MDP_instance}_{uncertainty}_radius_{alpha}_states_{S}_T_{T}_T_algo1_{T_algo_1}_nb_random_instances_{nb_random_instances}_rho_{rho}.npy',yvec_algo_all_1_ciup)
            np.save(f'numerical_results/yvec_algo_all_2_ciup_{MDP_instance}_{uncertainty}_radius_{alpha}_states_{S}_T_{T}_T_algo1_{T_algo_1}_nb_random_instances_{nb_random_instances}_rho_{rho}.npy',yvec_algo_all_2_ciup)
            np.save(f'numerical_results/yvec_algo_all_3_ciup_{MDP_instance}_{uncertainty}_radius_{alpha}_states_{S}_T_{T}_T_algo1_{T_algo_1}_nb_random_instances_{nb_random_instances}_rho_{rho}.npy',yvec_algo_all_3_ciup)

            fig.savefig(f"figures/errors_{MDP_instance}_{uncertainty}_radius_{alpha}_states_{S}_T_{T}_T_algo1_{T_algo_1}_axis_log_{axis_log}_nb_random_instances_{nb_random_instances}_rho_{rho}.pdf", bbox_inches='tight')
            
            plt.show()
            
        if uncertainty_cube:
            
            uncertainty = 'cube'
            print('Uncertainty:',uncertainty)
            
            for seed in range(nb_random_instances):
                
                print('Instance number ...',seed)
                
                optimal_gain = np.load(f'numerical_results/optimal_gain_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_algo1_{T_algo_1}_rho_{rho}_seed_{seed}.npy')
                
                A,N_B = 5,10
                P_nom=Garnet(S,N_B,A,seed)
                np.random.seed(seed)
                rew = np.random.rand(S,A)
                rew = rew/np.max(rew)
     
                scaling_factor_up = scale
                scaling_factor_down = scale
            
                P_up,P_down = form_uncertainty_hypercube(P_nom,scaling_factor_up,scaling_factor_down,S,A)
        
        
                print('Running Algo 1 (limit of discounted returns)...')
                result_limit_discounted_return, time_limit_discounted_return = run_algo_limit_discounted_returns_hypercube(rew,S,A,T,P_nom,P_up,P_down,p0,eps,rho)
        
                print('Running Algo 2 (VI - increasing horizon)...')
                result_algo_increasing_horizon,time_increasing_horizon = run_algo_VI_increasing_horizon_hypercube(rew,S,A,T,P_up,P_down,p0)
                print('Running Algo 3 (VI - increasing discount factor)...')
                result_algo_increasing_discounts,time_increasing_discounts = run_algo_VI_increasing_discounts_hypercube(rew,S,A,T,P_up,P_down,p0,rho)
            
            
                yvec_algo_all_1[seed] = abs(result_limit_discounted_return[:T]-optimal_gain)
                yvec_algo_all_2[seed] = abs(result_algo_increasing_horizon-optimal_gain)
                yvec_algo_all_3[seed] = abs(result_algo_increasing_discounts-optimal_gain)
                
            yvec_algo_all_1_mean = np.mean(yvec_algo_all_1,axis = 0)
            yvec_algo_all_2_mean = np.mean(yvec_algo_all_2,axis = 0)
            yvec_algo_all_3_mean = np.mean(yvec_algo_all_3,axis = 0)
            
            yvec_algo_all_1_std = np.std(yvec_algo_all_1,axis = 0)
            yvec_algo_all_2_std = np.std(yvec_algo_all_2,axis = 0)
            yvec_algo_all_3_std = np.std(yvec_algo_all_3,axis = 0)      
            
            yvec_algo_all_1_ciup = yvec_algo_all_1_mean + (1.96/np.sqrt(nb_random_instances))*yvec_algo_all_1_std
            yvec_algo_all_2_ciup = yvec_algo_all_2_mean + (1.96/np.sqrt(nb_random_instances))*yvec_algo_all_2_std
            yvec_algo_all_3_ciup = yvec_algo_all_3_mean + (1.96/np.sqrt(nb_random_instances))*yvec_algo_all_3_std

            yvec_algo_all_1_cidwn = yvec_algo_all_1_mean - (1.96/np.sqrt(nb_random_instances))*yvec_algo_all_1_std
            yvec_algo_all_2_cidwn = yvec_algo_all_2_mean - (1.96/np.sqrt(nb_random_instances))*yvec_algo_all_2_std
            yvec_algo_all_3_cidwn = yvec_algo_all_3_mean - (1.96/np.sqrt(nb_random_instances))*yvec_algo_all_3_std
        
        
            x_axis = np.asarray(range(T-1))
            indices = np.logspace(0.0, np.log10(len(x_axis)), num=nb_points)
            indices=indices.astype(int)
            indices[-1]=indices[-1]-2  
            
            fig, ax = plt.subplots()
            
            ax.tick_params(labelsize=25)
            
            if axis_log:
                ax.set_yscale('log')
                ax.set_xscale('log')
            
            ax.plot(x_axis[indices], yvec_algo_all_1_mean[indices], color='red', marker='*', linestyle=':',linewidth=2, markersize=10,label = 'Algo. 1')
            ax.plot(x_axis[indices], yvec_algo_all_2_mean[indices], color='green', marker='p', linestyle='--',linewidth=2, markersize=10, label = 'Algo. 2')
            ax.plot(x_axis[indices], yvec_algo_all_3_mean[indices], color='blue', marker='D', linestyle='-',linewidth=2, markersize=10,label = 'Algo. 3')
            
            threshold = 1e-10
            ax.fill_between(x_axis[indices], np.maximum(yvec_algo_all_1_cidwn[indices],threshold),yvec_algo_all_1_ciup[indices], color='red', alpha=0.1)
            ax.fill_between(x_axis[indices], np.maximum(yvec_algo_all_2_cidwn[indices],threshold),yvec_algo_all_2_ciup[indices], color='green', alpha=0.1)
            ax.fill_between(x_axis[indices], np.maximum(yvec_algo_all_3_cidwn[indices],threshold),yvec_algo_all_3_ciup[indices], color='blue', alpha=0.1)


    
            ax.set_xlabel('Number of iterations',fontsize=20)
            ax.set_ylabel('Error',fontsize=20)
            ax.tick_params(axis='both', which='major', labelsize=20)
    
            ax.grid()
    
            fig.legend(ncol=3,loc="upper center",bbox_to_anchor=(0.5,1.15,0,0),fontsize=20)
            
            np.save(f'numerical_results/yvec_algo_all_1_mean_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_nb_random_instances_{nb_random_instances}_rho_{rho}.npy',yvec_algo_all_1_mean)
            np.save(f'numerical_results/yvec_algo_all_2_mean_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_nb_random_instances_{nb_random_instances}_rho_{rho}.npy',yvec_algo_all_2_mean)
            np.save(f'numerical_results/yvec_algo_all_3_mean_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_nb_random_instances_{nb_random_instances}_rho_{rho}.npy',yvec_algo_all_3_mean)

            np.save(f'numerical_results/yvec_algo_all_1_cidwn_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_nb_random_instances_{nb_random_instances}_rho_{rho}.npy',yvec_algo_all_1_cidwn)
            np.save(f'numerical_results/yvec_algo_all_2_cidwn_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_nb_random_instances_{nb_random_instances}_rho_{rho}.npy',yvec_algo_all_2_cidwn)
            np.save(f'numerical_results/yvec_algo_all_3_cidwn_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_nb_random_instances_{nb_random_instances}_rho_{rho}.npy',yvec_algo_all_3_cidwn)
            
            np.save(f'numerical_results/yvec_algo_all_1_ciup_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_nb_random_instances_{nb_random_instances}_rho_{rho}.npy',yvec_algo_all_1_ciup)
            np.save(f'numerical_results/yvec_algo_all_2_ciup_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_nb_random_instances_{nb_random_instances}_rho_{rho}.npy',yvec_algo_all_2_ciup)
            np.save(f'numerical_results/yvec_algo_all_3_ciup_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_nb_random_instances_{nb_random_instances}_rho_{rho}.npy',yvec_algo_all_3_ciup)

            fig.savefig(f"figures/errors_{MDP_instance}_{uncertainty}_radius_{scale}_states_{S}_T_{T}_T_algo1_{T_algo_1}_axis_log_{axis_log}_rho_{rho}.pdf", bbox_inches='tight')
            
            plt.show()
