"""
This module defines the `factor_model` class, which is used to generate synthetic financial data based on 
a factor model framework. The factor model simulates returns as a combination of global factors, sparse factors, 
and idiosyncratic noise. It provides functionality for generating factor exposures, volatilities, and returns 
for use in financial analysis and simulations.

Key Features:
- Generates synthetic return data based on a factor model: R = psi * Y + phi * X + eps.
- Supports global (broad) factors, sparse factors, and idiosyncratic returns.
- Allows customization of the number of securities, factors, and volatilities.
- Provides methods for generating covariance matrices and simulating return streams.
- Includes utilities for adding correlations and seeding random number generators.

Usage:
This module is designed to be instantiated as a `factor_model` object, which can then be used to generate 
synthetic data for testing and validating various methods.
"""

import numpy as np

multivariate_normal = np.random.multivariate_normal
normal = np.random.normal
choice = np.random.choice
uniform = np.random.uniform

get_vols = lambda list4choice, num2choose: np.random.choice(list4choice,num2choose,replace=False)

country_vols_euro = [13.00, 12.03, 11.86, 12.38, 13.79, 18.53, 18.29,
    14.31, 13.33, 15.91, 17.57, 16.43, 24.93, 22.59, 24.41, 18.14]

industry_vols_euro = [13.11, 13.99, 9.80, 12.22, 12.74, 11.51, 17.80,
    21.23, 14.07, 15.08, 13.00]

###################################################
#                                                 #
# Generates a factor model of the form            #
# R = psi Y + phi X + eps                         #
#                                                 #
# Y - K0 by N matrix of global factor exposures   #
# X - (K1 + K2 + ...) by N matrix of sparse       #
#     factor exposures.                           #
# psi - global factor returns                     #
# phi - sparse factor returns                     #
# eps - idiosyncratic returns                     #
#                                                 #
# Instantiated as fm = factor_model(N = 32, ...)  #
# or fm = fm_(N = 32, ...) for shorthand. Every   #
# argument has a default value and so is not      #
# required to be specified.                       #
#                                                 #
# @N - total number of securities                 #
# @K0 - number of global (broad) factors          #
# @K1 - number of sparse factors of type 1        #
# ... K2, K3, K4 and so on.                       #
# @vol0 - global factor volatilities              #
# @vol1 - type 1 sparse factor volatilities       #
# ... vol2, vol3 and so on.                       #
# @vols - idiosyncratic volatilities              #
#                                                 #
###################################################

class factor_model:

    machine_tol = 1e-15
    num_tdays = 250
    pavol2var = lambda x: (x/100)**2 / factor_model.num_tdays

    def __init__ (self, **kwargs):

        dict_of_defaults = dict (N=32, K0=0, K1=0, seed=1)
        keys = kwargs.keys()

        # notify user of defaults used
        for key, val in dict_of_defaults.items():
            if key not in keys:
                print ("Defaulting %s to %.3f." % (key,val))
            #@ if
        #@ for

        # overwrite defaults with new arguments
        dict_of_defaults.update (kwargs)
        kwargs = dict_of_defaults
        keys = kwargs.keys()

        # set supplied arguments to attributes
        # also parse the factor types: K0, K1, K2, ...
        # the list KS must have at least K0 and K1
        self.KS = []
        for key, val in kwargs.items():
            setattr (self, key, val)

            label = 'K' + str ( len (self.KS) )
            if label in keys:
                self.KS.append ( kwargs[label] )
            #@ if
        #@ for
        #print('KS:', self.KS)
        # set the random numbers seed for self
        self.signature = [self.seed, self.N] + self.KS
        np.random.seed ( self.signature )

        # generate volatilities unless provided
        for i in range (len (self.KS)):

            # generate even if provided to keep consistent
            # random number generation

            vols = self.generate_volatilities (i, self.KS[i])

            # if not supplies assign
            key = 'vol' + str (i)
            if ( key not in keys ):
               setattr (self, key, vols)
            #@ if
            # if(self.KS[i]>0):
            #     print('Getting (annualized) volatilities (in percent) '+'[%s]' % ', '.join('%.5f' % val for val in vols)+' for factor_type=%i' % i)
            # #@ if
        #@ for

        # generate idiosyncratic volatilities
        if ( 'vols' not in keys ):
            self.vols = self.generate_volatilities (-1, self.N)
        #@ if

        # generate exposures
        self.Y = self.generate_global_exposures()
        self.X = self.generate_sparse_exposures()

        self.seed_return_generating_process()

    #@ __init__


    ######################################################
    # Create a random number generator for each of the 3 #
    # return streams: phi, psi, eps                      #
    #                                                    #
    # @seed - random number generator seed               #
    # can be either an int or a 3-tuple                  #
    ######################################################
    def seed_return_generating_process (self, seed = 0):

        seed = (np.int_([1,1,1]) + np.int_ (seed)).tolist()

        # set up the random number generators for the returns
        # len (rngs)  must be <= len (self.signature) for now
        ret = ['psi', 'phi', 'eps']

        sign = self.signature
        sign = [0,0,0] + sign

        # see each return type in rngs with a unique seed
        for i,r in enumerate (ret):
            sign[i] += seed[i]
            rng = r + '_rng'
            setattr (self, rng, np.random.RandomState (sign))
        #@ for

    def srgp (self, seed):
      return self.seed_return_generating_process (seed)
    #@ def


    ###################################################
    # Generates T realization of the returns          #
    # [ psi Y, phi X, eps ]                           #
    #                                                 #
    # @T - number of observations (int > 0)           #
    # @seed - random number generator seed            #
    ###################################################
    def generate_returns (self, T, seed = None):

        if ( seed is not None ):
            self.seed_return_generating_process (seed)
        #@ if

        psi = self.generate_global_factor_returns (T)
        phi = self.generate_sparse_factor_returns (T)
        eps = self.generate_idiosyncratic_returns (T)

        return [psi, phi, eps]

    ####################################################
    # Computes T realizations of returns to securities #
    # R = psi Y + phi X + eps                          #
    #                                                  #
    # @T - number of observations                      #
    ####################################################

    ## Fix this part so that when K1 = 0 and K2 = 0 (so that sum(KS[1:]) == 0), R = psi.dot (self.Y) + eps
    def R (self, T, seed = None):

        psi, phi, eps = self.generate_returns (T, seed)

        if (sum(KS[1:]) == 0):
            return psi.dot (self.Y) + eps
        else:
            return psi.dot (self.Y) + phi.dot (self.X) + eps
    #@ def


    ####################################################
    # Construct data covariance matrices based on T    #
    # T observations. T = infinity indicated the exact #
    # covariance is sought.                            #
    #                                                  #
    # @T - number of observations                      #
    #                                                  #
    # *use SLD for shorthand                           #
    ####################################################
    def covariance (self, T = np.inf, seed = 1):

        # population covariances
        if np.isinf (T):

            if self.K0 <= 0:
                L = np.zeros ( (self.N, self.N) )
            else:
                fvars = list (map (fm_.pavol2var, self.vol0))
                L = self.Y.T.dot ( np.diag (fvars) )
                L = L.dot (self.Y)
            #@ if

            if self.K1 <= 0:
                S = np.zeros ( (self.N, self.N) )
            else:
                population_sizes = self.KS[1:]
                mean = np.zeros ( np.sum (population_sizes) )

                fvars = self.get_sparse_factor_volatilities()
                fvars = list (map (fm_.pavol2var, fvars))

                S = self.X.T.dot ( np.diag (fvars) )
                S = S.dot (self.X)

            svars = list (map (fm_.pavol2var, self.vols))
            D = np.diag (svars)
        else: # sample covariance
            psi, phi, eps = self.generate_returns (T, seed)

            RY = psi.dot (self.Y)
            L = RY.T.dot (RY) / T

            RX = phi.dot (self.X)
            S = RX.T.dot (RX) / T

            D = eps.T.dot (eps) / T

        return [S, L, D]

    def SLD (self, T = np.inf, seed = 1):
        return self.covariance (T, seed)
    #@ def



    ########################################################
    # Methods below are in a sense private (used in init)  #
    ########################################################

    def generate_global_exposures (self):

        np.random.seed(self.seed)

        # check if there are global factors
        if self.K0 <= 0:
            return np.zeros (1) # or something else?

        # market factor (Revised by BHK 20190331)
        Y0 = normal (1.0, np.sqrt(19/81), (1,self.N))

        # number of remaining factors
        K = self.K0 - 1

        if K <= 0:
            return Y0

        # style factors
        mean = np.zeros (K)
        vcov = np.diag (np.ones (K))*0.75

        Y1 = multivariate_normal (mean, vcov, self.N).T

        return np.vstack ((Y0, Y1))
    #@ def


    def generate_sparse_exposures (self):

        np.random.seed(self.seed)
        # if no sparse factors are present return
        if self.K1 <= 0:
            return np.zeros (1) # or something else?

        # enumerates numbers of factors of each type
        population_sizes = self.KS[1:]
        # counts size of factor
        sizes = np.ones (self.N) / self.N

        # iterate over each type of sparse factor
        z = []
        for K in population_sizes:
            # matrix of exposures to this sparse factor
            Z = np.zeros ( (K, self.N) )

            # order the indices in proportion to populations
            indices = choice (self.N, self.N, p=sizes, replace=False)
            indices = indices.tolist()

            # initialize initial assignments to the K factors at
            # the end each factor will have exactly 1 security.
            init = choice (indices, K, replace=False)
            for i in range (K):
                Z[i, init[i]] = 1
                indices.remove ( init[i] )
            #@ for

            for j in indices:
                # assignment probabilities
                prob = np.maximum(np.sum (Z, 1),5)
                # pick uniformly with prob from unselected factors
                prob = prob/prob.sum()

                # choose a sparse factor in proportion to size
                i = choice (K, 1, p = prob)[0]
                # add security to this sparse factor
                Z[i,j] = 1
            #@ for

            # for each security store the size of the population
            # of the factor it was assigned to.
            for j in range (self.N):
                i = Z[ : , j].tolist().index(1)
                sizes[j] = np.sum ( Z [i, ] )
            #@ for
            sizes = sizes / np.sum (sizes)

            # store the set (type) of sparse factors
            z.append (Z)
        #@ for

        return np.vstack (z)
    #@ def


    def generate_volatilities (self, t, K):

        if K <= 0:
            return 0 

        # idiosyncratic volatilities
        if t < 0:
            vols = uniform (32, 64, K).tolist() 
        # global factor volatilities
        elif t == 0:
            # generate style factor volatilities
            vols = []
            if (K >= 2):
                for nn in range(K-2):
                    vols.insert(0, uniform (4, 4)) 
                # add beta style factor
                vols.insert (0, 8.0)
            # add market factor volatility
            vols.insert (0, 16.0)
        # sparse factor volatilities (scale with 1/t)
        elif t == 1:
            vols = country_vols_euro#country_vols_dev + country_vols_emg
            vols = get_vols(vols, K).tolist()
        elif t == 2:
            vols = industry_vols_euro
            vols = get_vols(vols, K).tolist()
        else:
            cent = 20 / np.sqrt (t)
            mdev = 10 / np.log (1+t)
            vols = uniform (cent-mdev, cent+mdev, K).tolist()
        return vols

    def gv (self, t, K):
       self.generate_volatilities (t, K)
    #@ def



    def generate_global_factor_returns (self, T):

        if self.K0 <= 0:
            return np.zeros (1) # force dimension to T x N?

        mean = np.zeros (self.K0)
        # vcov = np.diag (self.vol0)

        rng = self.psi_rng

        return rng.multivariate_normal (mean, self.add_correlation(np.array(self.vol0)**2, self.uniformCorrelation), T)


    def psi (self, T):
        return self.generate_global_factor_returns (T)
    #@ def



    def generate_sparse_factor_returns (self, T):

        if self.K1 <= 0:
            return np.zeros (1) 

        population_sizes = self.KS[1:]
        mean = np.zeros ( np.sum (population_sizes) )

        vols = []
        for t in range (len (population_sizes)):
             vols.extend (getattr (self, 'vol' + str (t+1)))

        rng = self.phi_rng

        return rng.multivariate_normal (mean, self.add_correlation(np.array(vols)**2, self.uniformCorrelation), T)

    def phi (self, T):
        return self.generate_sparse_factor_returns (T)
    #@ def


    def generate_idiosyncratic_returns (self, T):

        if self.vols == None:
            return np.zeros (1) # force dimension to T x N?

        mean = np.zeros (self.N)
        rng = self.eps_rng

        return rng.multivariate_normal (mean, self.add_correlation(np.array(self.vols)**2, 0.0), T)


    def eps (self, T):
        return self.generate_idiosyncratic_returns (T)
    #@ def

    def add_correlation(self,var,correlation = 0.0):
        vol = np.sqrt(var).reshape(1,len(var))
        cov = vol.T.dot(vol)*correlation
        np.fill_diagonal(cov,0.0)
        cov += np.diag(var)
        return cov

#@ class
fm_ = factor_model
