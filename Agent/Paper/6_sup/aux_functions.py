'''
16/01/26
Code for "Beyond discounted returns: Robust Markov decision processes with average and Blackwell optimality"
Julien Grand-Clement, Marek Petrik, Nicolas Vieille
Author of the code: Julien Grand-Clement (grand-clement@hec.fr)

This file is the auxiliary functions: MDP instances, algorithms (Alg. 1-2-3),
value iteration, policy iteration
'''


#####################################################################################
#############################
#############################     Defining MDP Functions
#############################
#####################################################################################

from __future__ import division
import numpy as np
import random
import hiive.mdptoolbox.example
import time


#####################################################################################
#############################
#############################     MDP Instances
#############################
#####################################################################################


def Healthcare_MDP(S):
    
    ''' 
    Build the healthcare MDP with S states
    Input:
        S: number of states
    Output:
        P: transition probabilities
        rew: instantaneous rewards
    '''
    
    P=np.zeros((S,3,S))
    rew=np.zeros((S,3))
    for s in range(1,S-1):
        
        rew[s,0]=5
        rew[s,1]=4
        rew[s,2]=3
        
        P[s,0,s]=0.4
        P[s,0,s+1]=0.3
        P[s,0,s-1]=0.3
        
        P[s,1,s]=0.4
        P[s,1,s+1]=0.2
        P[s,1,s-1]=0.4
        
        P[s,2,s]=0.4
        P[s,2,s+1]=0.1
        P[s,2,s-1]=0.5

    # s=0
    rew[0,0]=5
    rew[0,1]=4
    rew[0,2]=3
    
    P[0,0,0]=0.7
    P[0,0,1]=0.3
    
    P[0,1,0]=0.8
    P[0,1,1]=0.2
    
    P[0,2,0]=0.9
    P[0,2,1]=0.1

    # s=m  

    P[S-1,0,S-1]=1      
    P[S-1,1,S-1]=1      
    P[S-1,2,S-1]=1  
    
    
    return P,rew

def Forest_MDP(S):
    
    ''' 
    Build the forest MDP with S states from the mdptoolbox [Cordwell et al. 2015]
    Input:
        S: number of states
    Output:
        P0: transition probabilities
        R: instantaneous rewards
    '''
    
    P, R = hiive.mdptoolbox.example.forest(S,0.05)
    
    P0=np.zeros((S,2,S))
    for s in range(S):
        P0[s,0,:]=P[0,s,:]
        P0[s,1,:]=P[1,s,:]
    return P0,R

def Garnet(N_S,N_B,N_A,i):
    
    ''' 
    Build Garnet MDPs [Archibald et al. 1995]
    Input:
        N_S: number of states
        N_B: branching factor
        N_A: number of actions
        i: seed
    Output:
        P: transition probabilities
        (the instantaneous rewards are sampled in the main file)
    '''

    random.seed(N_S+N_B+i)
    np.random.seed(N_S+N_B+i)
    P=np.zeros((N_S,N_A,N_S))
    
    for s in range(N_S):
        for a in range(N_A):
            
            # sample Nb - 1 random numbers between 0 & 1 
            
            sub=np.random.rand(N_B-1)
            sub=np.sort(sub)
            sub_vec=np.zeros(N_B+1)
            sub_vec[0]=0
            sub_vec[N_B]=1
            sub_vec[1:(N_B)]=sub
            
            # Sample the N_B next states
            
            next_state=random.sample(range(N_S), N_B)
            
            for s2 in range(N_B):
                
                P[s,a,next_state[s2]]=sub_vec[s2+1]-sub_vec[s2]
                
                
    return P

def Machine_MDP(S):
    P=np.zeros((S,2,S))
    rew=np.zeros((S,2))
    for s in range(S-3):
        rew[s,0]=(1/(s+1))
        rew[s,1]=(0.1)/(s+1)
        P[s,0,s]=0.2
        P[s,0,s+1]=0.8
        P[s,1,s+1]=0.3
        P[s,1,S-2]=0.1
        P[s,1,S-1]=0.6
        
    # state " 8 " (last non-absorbing states)
    
    rew[S-3,0]=(1)/(S-2)
    rew[S-3,1]=(0.1)/(S-2)
    
    P[S-3,0,S-3]=1.0
    P[S-3,1,S-3]=0.3
    P[S-3,1,S-2]=0.1
    P[S-3,1,S-1]=0.6
    
    # state R2
    
    rew[S-2,0]=(1)/(S-1)
    rew[S-2,1]=(0.05)/(S-1)
    
    P[S-2,0,S-2]=1.0
    P[S-2,1,S-2]=0.4
    P[S-2,1,S-1]=0.6
    
    # state R1
    
    rew[S-1,0]=(0.5)/(S)
    rew[S-1,1]=(0.08)/(S)
    
    P[S-1,0,S-1]=0.2
    P[S-1,0,0]=0.8
    P[S-1,1,S-1]=1
    
    return P,rew

def L_infinity(v):
    return np.max(np.abs(v))
#####################################################################################
#############################
#############################     VALUE ITERATION - Ellipsoid setup
#############################
#####################################################################################

def U_basis(n):
    
    ''' 
    Orthonormal basis for the hyperplane {x | <x,e> = 1}
    '''

    U=np.zeros((n,n-1))
    
    for i in range(1,n):
        U[0:i,i-1]=(1/i)
        U[i,i-1]=-1
        U[:,i-1]=np.sqrt(i/(i+1))*U[:,i-1]
        
    return U

def form_uncertainty_ellipsoid(P_nom,alpha,S,A):
    
    '''
    
    Create radiuses for ellipsoid around each transition probabilities P
    We want a singleton anytime that we have absorbing state
    
    '''
    
    radius = np.zeros((S,A))
    for s in range(S):
        for a in range(A):
            if np.max(P_nom[s,a])==1:
                radius[s,a]=0
            else:
                radius[s,a]=alpha

    return radius

def robust_bellman_ellipsoid(rew,P_nom,gamma,V,S,A,radius):
    
    '''
    
    Robust Bellman operator T applied at point V
    Uncertainty set is ellipsoid
    U = orthogonal basis of { x | <x,e> = 1 }
    
    Input:
        rew,P_nom,gamma,S,A: parameters of the MDP instance
        V: vector to apply the Bellman operator
    
    '''
    
    v_next = np.zeros(S)
    
    for s in range(S):
        Q_sa = [ellipsoid_update(rew[s,a],P_nom[s,a],gamma,V,S,A,radius[s,a]) for a in range(A)]
        v_next[s] = max(Q_sa)
    
    return v_next

def ellipsoid_update(rew,P_nom,gamma,V,S,A,radius):

    '''
    
    Return min_{P_{sa} \in U_{sa}} r[s,a] + gamma*<P,V>
    when U_{sa} = { p \in \Delta(S) | || p - p_nom ||_2 <= radius}
    
    Following Reduction from Section D.3 in 
    Conic Blackwell Algorithm: Parameter-free Convex-Concave Saddle-point solving
    
    '''    
    if radius == 0.0:
        return rew+gamma*np.dot(P_nom,V)
    
    else:
        
        
        ## Extract v_loc: vector with components where transitions are non-zero
        index_non_zero = np.nonzero(P_nom)[0]
        v_loc = V[index_non_zero]
        
        # Define basis of { x | <x,e> = 1 }
        U = U_basis(len(index_non_zero))
        
        proj_v = np.dot(np.transpose(U),v_loc)
        
        if np.linalg.norm(proj_v) == 0.0:
            return rew+gamma*v_loc[0]
        
        else:
            
            scalar = (radius)/np.linalg.norm(proj_v)
            z_opt = -scalar*np.dot(U,proj_v)

            P_next = P_nom.copy()
            P_next[index_non_zero] += z_opt
            
            return rew+gamma*(np.dot(P_nom,V) - radius*np.linalg.norm(np.dot(np.transpose(U),v_loc)))

def run_algo_VI_increasing_horizon_ellipsoid(rew,P_nom,S,A,T,radius,p0):
    
    '''
    
    Running Algorithm 2 for T iterations
    Input:
        Robust MDP instance: parameters of the MDP instance
        T: number of iterations
    Output:
        result: array of current <p0,vt>
        
    '''    

    result = np.zeros(T)
    result_time = np.zeros(T)
    start = time.time()

    # Note: due to the division in ellipsoid_update
    # we can not start at V = 0
    # So we start at a small constant on every dimension
    V = 0.01*np.ones(S)
    
    for t in range(T):
        # print('Iteration:',t)
        V = robust_bellman_ellipsoid(rew,P_nom,1,V,S,A,radius)
        result[t] = (1/(t+1))*np.dot(p0,V)
        result_time[t]= time.time()- start
        
    return result,result_time

def run_algo_VI_increasing_discounts_ellipsoid(rew,P_nom,S,A,T,radius,p0,rho):
    
    '''
    
    Running Algorithm 3 for T iterations
    Input:
        Robust MDP instance: parameters of the MDP instance
        T: number of iterations
    Output:
        result: array of current <p0,vt>

    '''    

    result = np.zeros(T)
    result_time = np.zeros(T)
    start = time.time()

    # Note: due to the division in ellipsoid_update
    # we can not start at V = 0
    # So we start at a small constant on every dimension
    V = 0.01*np.ones(S)

    for t in range(T):
        # print('Iteration:',t)
        gamma_t = 1-(1/((t+2)**rho))
        V = robust_bellman_ellipsoid((1-gamma_t)*rew,P_nom,gamma_t,V,S,A,radius)
        result[t] = np.dot(p0,V)
        
        result_time[t]= time.time()- start
        
    return result,result_time

#####################################################################################
#############################
#############################     STRATEGY ITERATION - Ellipsoid setup
#############################
#####################################################################################

def run_algo_limit_discounted_returns_ellipsoid(rew,S,A,T,P_nom,radius,p0,eps,rho):
    
    '''
    Compute the sequence of optimal discounted return with gamma_t -> 1
    Input:
        Robust MDP instance: parameters of the MDP instance
        T: number of iterations

    '''    

    result = np.zeros(T)
    
    result_time = np.zeros(T)
    start = time.time()
    
    pi_warm = np.asarray([0] * S)
    
    for t in range(T):
        gamma_t = 1-(1/((t+2)**rho))
        if t % 50 == 0:
            print('Iteration:',t)
        # next pi warm start = output of function
        result_loc, pi_warm = run_policy_iteration_ellipsoid(rew,S,A,T,P_nom,radius,p0,gamma_t,pi_warm,(1-gamma_t)*eps)
        result[t] = (1-gamma_t)*result_loc
        
        result_time[t]= time.time()- start
        
    return result,result_time

def run_policy_iteration_ellipsoid(rew,S,A,T,P_nom,radius,p0,gamma,pi_warm,eps):
    
    '''
    
    Runs 2-player Strategy-Iteration to compute optimal discount return
    Stop at optimality and return optimal return
    Use warm-start for policies
    
    '''
    
    # initial strategy and value function
    pi = pi_warm
    P_warm = [P_nom[s,pi[s]] for s in range(S)]

    v = np.zeros(S)
    # next P warm start = output of function
    v_next,P_warm = robust_value_function_ellipsoid(rew,S,A,T,P_nom,radius,gamma,pi,P_warm,eps)

    # if keep track of number of iterations in the loop
    k=0
    
    # print('Starting Policy Iteration...')
    
    while L_infinity(v - v_next)>eps:        
        
        v = v_next.copy()
        pi = robust_greedy_update_ellipsoid(rew,S,A,T,P_nom,radius,gamma,v)
        # print('computing robust value function...')
        # next P warm start = output of function
        v_next,P_warm = robust_value_function_ellipsoid(rew,S,A,T,P_nom,radius,gamma,pi,P_warm,eps)
        
        k+=1
        # print('iteration in the loop:',k)
        
    return np.dot(p0,v),pi

def robust_greedy_update_ellipsoid(rew,S,A,T,P_nom,radius,gamma,v):
    
    '''
    
    Return robust greedy policy at vector v
    subroutine -> Robust Bellman operator T applied at point V
    Uncertainty set is ellipsoid
    Input:
        rew,P_nom,gamma,S,A: parameters of the MDP instance
        V: vector to apply the Bellman operator
    Output: greedy policy
    
    '''
    
    pi_next = np.asarray([0] * S)
    
    for s in range(S):
        loc_Q_sa = np.asarray([ellipsoid_update(rew[s,a],P_nom[s,a],gamma,v,S,A,radius[s,a]) for a in range(A)])
        pi_next[s]=np.argmax(loc_Q_sa)
    
    return pi_next   

def robust_value_function_ellipsoid(rew,S,A,T,P_nom,radius,gamma,pi,P_warm,eps):
    
    '''
    
    Return the robust value function for a fixed pi
    Runs policy iteration in the adversarial MDP
    
    '''
    
    # initial P
    P = P_warm
    v_worst = np.zeros(S)
    v_next_worst = nominal_value_function(rew,S,A,gamma,pi,P)
    
    # if keep track of iterations
    k=0
    # print('Starting Adversarial Policy Iteration...')
    
    while L_infinity(v_worst - v_next_worst)>eps:
        v_worst = v_next_worst.copy()
        
        P = adversarial_greedy_update_ellipsoid(rew,S,A,T,P_nom,radius,gamma,v_worst,pi)
        
        v_next_worst = nominal_value_function(rew,S,A,gamma,pi,P)
    
        k+=1
    
    return v_worst,P


def adversarial_greedy_update_ellipsoid(rew,S,A,T,P_nom,radius,gamma,v_worst,pi):
    
    '''
    
    Return the transition probabilities in arg min_{P_{sa} \in U_{sa}} <P,V>,
    for each state s
        when U_{sa} = { p \in \Delta(S) | || p - p_nom ||_2 <= radius}
    
        
    '''    
    P_next = np.zeros((S,S))
    
    for s in range(S):
        
        if radius[s,pi[s]] == 0.0:
            P_next[s]=P_nom[s,pi[s]]
        
        else:
            
            ## Extract v_loc: vector with components where transitions are non-zero
            index_non_zero = np.nonzero(P_nom[s,pi[s]])[0]
            v_loc = v_worst[index_non_zero]
            
            # Define basis of { x | <x,e> = 1 }
            U = U_basis(len(index_non_zero))
            proj_v = np.dot(np.transpose(U),v_loc)
            
            if np.linalg.norm(proj_v) == 0:
                P_next[s] = P_nom[s,pi[s]]
            
            else:

                scalar = (radius[s,pi[s]])/np.linalg.norm(proj_v)
                z_opt = -scalar*np.dot(U,proj_v)
                
                P_next[s] = P_nom[s,pi[s]]
                P_next[s,index_non_zero] += z_opt
                
                if abs(sum(P_next[s]) - 1) > 1e-6:
                    
                    print('Error: P_next[s] not a proper distribution')
                    print('P_next:',P_next[s])
                    print('Sum P_next:',sum(P_next[s]))
                    print('Index_non_zero:',index_non_zero)
                    print('z_opt:',np.around(z_opt,4))
                    
                    
    return P_next

#####################################################################################
#############################
#############################     VALUE ITERATION - L_infinity setup
#############################
#####################################################################################

def form_uncertainty_hypercube(P_nom,scaling_factor_up,scaling_factor_down,S,A):
    
    '''
    
    Create P_up and P_down by scaling P_nom by scaling factor
    
    '''
    
    P_up = np.zeros((S,A,S))
    P_down = np.zeros((S,A,S))
    for s in range(S):
        for a in range(A):
            for s_prime in range(S):
                
                if P_nom[s,a,s_prime] == 0:
                    P_up[s,a,s_prime] = 0
                    P_down[s,a,s_prime] = 0
                elif P_nom[s,a,s_prime] == 1:
                    P_up[s,a,s_prime] = 1
                    P_down[s,a,s_prime] = 1
                else:
                    P_up[s,a,s_prime]=(1-scaling_factor_up)*P_nom[s,a,s_prime] + scaling_factor_up
                    P_down[s,a,s_prime]=(1-scaling_factor_down)*P_nom[s,a,s_prime]
            
    return P_up,P_down

def robust_bellman_hypercube(rew,gamma,V,S,A,P_up,P_down):
    
    '''
    
    Robust Bellman operator T applied at point V
    Uncertainty set is ell_infinity
    Input:
        rew,P_nom,gamma,S,A: parameters of the MDP instance
        V: vector to apply the Bellman operator    
    '''
    
    v_next = np.zeros(S)
    
    for s in range(S):
        loc_Q_sa = np.asarray([hypercube_update(rew[s,a],gamma,V,S,A,P_up[s,a],P_down[s,a]) for a in range(A)])
        v_next[s] = np.max(loc_Q_sa)
    
    return v_next
            
def hypercube_update(rew,gamma,V,S,A,P_up,P_down):

    '''
    
    Return min_{P_{sa} \in U_{sa}} r[s,a] + gamma*<P_{s,a},V>
    when U_{sa} = { p \in \Delta(S) | || p - p_nom ||_{\infinity} <= radius}
    
    Follow proposition 3 in Data Uncertainty in Markov Chains, Goh et al.
    
    '''  
    
    if np.max(np.abs(P_up - P_down))==0:
        # no need for optimizing, we are at an absorbing state
        return rew + gamma*np.dot(P_up,V)
        
    else:
    
    
        L = [ (V[i],i) for i in range(len(V)) ]
        L.sort()
        sorted_V,permutation = zip(*L)
        
        # example:
           # v = [591, 760, 940, 764, 391]
           # sorted_v = (391, 591, 760, 764, 940)
           # permutation = (4, 0, 1, 3, 2)
           
        index,run_sum = find_index(permutation, P_up, P_down,S)
        
        P_next = np.zeros(S)
        for s in range(index):
            P_next[permutation[s]] = P_up[permutation[s]]
        for s in range(index+1,S):
            P_next[permutation[s]] = P_down[permutation[s]]
        P_next[permutation[index]] = 1 - run_sum
        
        return rew + gamma*np.dot(P_next,V)

def find_index(permutation, P_up, P_down,S):
    index = 0
    run_sum = P_up[permutation[0]]+sum([P_down[permutation[j]] for j in range(1,S)])

    while run_sum < 1 :
        index +=1
        run_sum += P_up[permutation[index]] - P_down[permutation[index]]

    return index, run_sum - P_up[permutation[index]]

def run_algo_VI_increasing_horizon_hypercube(rew,S,A,T,P_up,P_down,p0):
    
    '''
    Running VI with increasing horizon
    Input:
        Robust MDP instance: parameters of the MDP instance
        T: number of iterations
    Output:
        v_t: error-approximate value function
        last error to v*
    '''    

    result = np.zeros(T)
    result_time = np.zeros(T)
    start = time.time()

    V=0*np.ones(S)
    
    for t in range(T):
        # print('Iteration:',t)
        V = robust_bellman_hypercube(rew,1,V,S,A,P_up,P_down)
        result[t] = (1/(t+1))*np.dot(p0,V)
        
        result_time[t]= time.time() - start
        
    return result,result_time

def run_algo_VI_increasing_discounts_hypercube(rew,S,A,T,P_up,P_down,p0,rho):
    
    '''
    
    Running VI with increasing discount factor
    Input:
        Robust MDP instance: parameters of the MDP instance
        T: number of iterations
    Output:
        result: array of current <p0,vt>

    '''    

    result = np.zeros(T)
    result_time = np.zeros(T)
    start = time.time()
    
    V=0*np.ones(S)
    
    for t in range(T):
        gamma_t = 1-(1/((t+2)**rho))
        V = robust_bellman_hypercube((1-gamma_t)*rew,gamma_t,V,S,A,P_up,P_down)
        result[t] = np.dot(p0,V)
        
        result_time[t]= time.time() - start
        
    return result,result_time  

#####################################################################################
#############################
#############################     STRATEGY ITERATION - L_infinity setup
#############################
#####################################################################################

def run_algo_limit_discounted_returns_hypercube(rew,S,A,T,P_nom,P_up,P_down,p0,eps,rho):
    
    '''
    Compute the sequence of optimal discounted return with gamma_t -> 1
    Input:
        Robust MDP instance: parameters of the MDP instance
        T: number of iterations

    '''    

    result = np.zeros(T)
    result_time = np.zeros(T)
    start = time.time()
    
    pi_warm = np.asarray([0] * S)
    
    for t in range(T):
        if t % 50 == 0:
            print('Iteration:',t)
        gamma_t = 1-(1/((t+2)**rho))
        # next pi warm start = output of function
        result_loc, pi_warm = run_policy_iteration_hypercube(rew,S,A,T,P_nom,P_up,P_down,p0,gamma_t,pi_warm,(1-gamma_t)*eps)
        result[t] = (1-gamma_t)*result_loc
        result_time[t]=  time.time()- start
        
    return result,result_time 

def run_policy_iteration_hypercube(rew,S,A,T,P_nom,P_up,P_down,p0,gamma,pi_warm,eps):
    
    '''
    
    Runs 2-player Strategy-Iteration to compute optimal discount return
    Stop at optimality and return optimal return
    
    '''
    
    # initial strategy and value function
    pi = pi_warm
    P_warm = [P_nom[s,pi[s]] for s in range(S)]
    v = np.zeros(S)
    v_next,P_warm = robust_value_function_hypercube(rew,S,A,T,P_nom,P_up,P_down,gamma,pi,P_warm,eps)

    # if keep track of number of iterations in the loop
    k=0
    
    # print('Starting Policy Iteration...')
    
    while L_infinity(v - v_next) > eps:
        
        v = v_next.copy()
        pi = robust_greedy_update_hypercube(rew,S,A,T,P_up,P_down,gamma,v)

        v_next,P_warm = robust_value_function_hypercube(rew,S,A,T,P_nom,P_up,P_down,gamma,pi,P_warm,eps)

        k+=1
        
        
    return np.dot(p0,v),pi

def robust_greedy_update_hypercube(rew,S,A,T,P_up,P_down,gamma,v):
    
    '''
    
    Return robust greedy policy at vector v
    subroutine -> Robust Bellman operator T applied at point V
    Uncertainty set is ell_infinity
    Input:
        rew,P_nom,gamma,S,A: parameters of the MDP instance
        V: vector to apply the Bellman operator
    Output: greedy policy
    
    '''
    
    pi_next = np.asarray([0] * S)
    
    for s in range(S):
        loc_Q_sa = np.asarray([hypercube_update(rew[s,a],gamma,v,S,A,P_up[s,a],P_down[s,a]) for a in range(A)])
        pi_next[s]=np.argmax(loc_Q_sa)
    
    return pi_next   

def robust_value_function_hypercube(rew,S,A,T,P_nom,P_up,P_down,gamma,pi,P_warm,eps):
    
    '''
    
    Return the robust value function for a fixed pi
    Runs policy iteration in the adversarial MDP
    Using warm-start for the adversary
    '''
    
    # initial P
    P = P_warm
    v_worst = np.zeros(S)
    v_next_worst = nominal_value_function(rew,S,A,gamma,pi,P)
    
    # if keep track of iterations
    k=0
    # print('Starting Adversarial Policy Iteration...')
    # choice of thresholds 1e-9 (otherwise we may have numerical issues)
    while L_infinity(v_worst - v_next_worst)>1e-9:
        
        v_worst = v_next_worst.copy()
        
        P = adversarial_greedy_update_hypercube(rew,S,A,T,P_up,P_down,gamma,v_worst,pi)
        v_next_worst = nominal_value_function(rew,S,A,gamma,pi,P)

    
        k+=1
    
    return v_worst,P

def nominal_value_function(rew,S,A,gamma,pi,P):
    
    '''
    
    Return the nominal value function associated with pi and P
    
    '''
    
    rew_pi = np.asarray([rew[s,pi[s]] for s in range(S)])
    P_loc = np.asarray(P)
    return np.dot(np.linalg.inv(np.eye(S)-gamma*P_loc),rew_pi)

def adversarial_greedy_update_hypercube(rew,S,A,T,P_up,P_down,gamma,v_worst,pi):
    
    '''
    
    Return the transition probabilities in arg min_{P_{sa} \in U_{sa}} <P,V>,
    for each state s
    Follow proposition 3 in Data Uncertainty in Markov Chains, Goh et al.

    
    '''
    
    P_next = np.zeros((S,S))
    for s in range(S):
        

        if np.max(np.abs(P_up[s,pi[s]] - P_down[s,pi[s]]))==0:
            # no need for optimizing, we are at an absorbing state
            P_next[s]=P_up[s,pi[s]]
            
        else:
        
            L = [ (v_worst[i],i) for i in range(len(v_worst)) ]
            L.sort()
            sorted_v_worst,permutation = zip(*L)

            
            # example:
               # v = [591, 760, 940, 764, 391]
               # sorted_v = (391, 591, 760, 764, 940)
               # permutation = (4, 0, 1, 3, 2)
               
            index,run_sum = find_index(permutation, P_up[s,pi[s]], P_down[s,pi[s]],S)
            
            P_next_loc = np.zeros(S)
            for s_next in range(index):
                P_next_loc[permutation[s_next]] = P_up[s,pi[s],permutation[s_next]]
            for s_next in range(index+1,S):
                P_next_loc[permutation[s_next]] = P_down[s,pi[s],permutation[s_next]]
            P_next_loc[permutation[index]] = 1 - run_sum
            
            P_next[s] = P_next_loc
        
    return P_next
    
