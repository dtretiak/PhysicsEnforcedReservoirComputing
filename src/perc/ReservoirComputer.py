import numpy as np
import warnings
import scipy
import copy
import scipy
import scipy.linalg

class ReservoirComputer():
    '''
    Basic Reservoir Computer Implementation

    METHODS:
        update_state(u): update the reservoir state with input u (of shape Nu x 1), returns the next reservoir state r
        train(U): train the reservoir with input U (of shape Nu x N), returns the history of reservoir states R (of shape Nr x N-1)
        forecast(n): forecast n time steps into the future, returns the forecasted state U_pred (of shape Nu x n)
    '''

    def __init__(self, Nu, Nr = 1000, alpha = 0.6, beta = 8e-8, rho_A = 0.1, lambda_max = 0.8, sigma=0.084, bias = 1.6, spinup = 500, f = np.tanh):
        '''
        Intialize basic Reservoir Computer

        INPUTS:
            Nu: dimension of input (i.e. Nu = 3 for Lorenz63)
            Nr: number of nodes in reservior
            alpha: leak rate
            beta: regularization parameter for ridge regression. Increase if solver gives ill-conditioned warning
            rho_A: density of reservoir matrix
            lambda_max: spectral radius of reservoir matrix (maximum eigenvalue)
            sigma: bounds on distribution of W_in
            bias: magnitude bias term
            spinup: number of time steps to synchronize reservoir
            f: function used when evolving reservoir states. Default is tanh
        '''
        # dimension of input
        self.Nu = Nu

        # assign reservoir hyperparameters
        self.Nr = Nr
        self.alpha = alpha
        self.beta = beta 
        self.rho_A = rho_A
        self.lambda_max = lambda_max
        self.spinup = spinup
        self.f = f

        # build reservoir
        self.Wr = scipy.sparse.random(self.Nr, self.Nr, density = self.rho_A, data_rvs=np.random.randn).tocsr()
        try: 
            max_eig = np.abs(scipy.sparse.linalg.eigs(self.Wr, k=1, which='LM', return_eigenvectors=False)[0]) # doesn't always converge
        except Exception:
            print("Sparse eigenvalue solver failed to converge. Switching to dense eigenvalue solver.")
            Wr_dense = self.Wr.todense() # convert to dense for eigvals
            max_eig = np.max(np.abs(np.linalg.eigvals(Wr_dense)))
        self.Wr =  self.lambda_max * (self.Wr/max_eig)

        self.r = np.zeros((Nr,1))
        self.W_in = np.random.uniform(-sigma, sigma, size=(self.Nr, self.Nu))
        self.b = bias*np.ones([self.Nr,1])

    def update_state(self, u):
        '''
        Updates internal reservoir state r with input u

        INPUTS:
            u: input of shape Nu x 1
        OUTPUTS:
            r: updated reservoir state of shape Nr x 1
        '''
        u = u.reshape(-1,1) # u should be a col vector
        r_next = self.f(self.Wr@self.r + self.W_in@u + self.b)
        self.r = (1-self.alpha)*self.r + self.alpha*r_next
        return self.r
    
    def train(self, U, method = 'lstsq'):
        '''
        Trains reservoir by fitting output weights W_out with ridge regression

        INPUTS:
            U is Nu x N matrix where Nu is the number of state variables and N is the number of samples in time
            method: linear solver to use. Options are 'direct_solve', 'lstsq'

        OUTPUTS:
            R: history of reservoir states of shape Nr x N-1
        '''

        # Error check spinup
        if self.spinup > U.shape[1]:
            warnings.warn('Spinup is greater than the number of samples in U. Setting spinup to U.shape[1]/2')
            self.spinup = U.shape[1]//2

        # synchronize the reservior
        for i in range(self.spinup):
            _  = self.update_state(U[:, i])
        U = U[:, self.spinup:]

        # Loop thru time
        R = np.zeros((self.Nr, U.shape[1]-1))   # Keep track of reservoir states thru training loop
        for i in range(U.shape[1]-1): # -1 because we need to predict the NEXT state
            R[:,i] = (self.update_state(U[:,i])).reshape(-1)
        U = U[:,1:] # make sure R is aligned with the NEXT state of U

        # optimize with ridge regression || AX - B ||^2 + beta ||X||^2
        A = (R@R.T + self.beta * np.eye(self.Nr)) 
        B = R@U.T 
        
        # solve for W_out (X)
        if method == 'lstsq':
            X = scipy.linalg.lstsq(A, B, cond=None)[0].T
        elif method == 'direct_solve':
            X = scipy.linalg.solve(A, B, assume_a='sym').T
        self.W_out = X

        return R #optional return for debugging
    
    def getR(self, U):
        '''
        Drives Reservoir with input U, and returns the history of reservoir states R for gradient based training

        INPUTS:
            U: input of shape Nu x N
        OUTPUTS:
            U: trimmed input of U to account for spinup
            R: history of reservoir states of shape Nr x N-1
        '''

        # synchronize the reservior
        for i in range(self.spinup):
            _  = self.update_state(U[:, i])
        U = U[:, self.spinup:]

        # Loop thru time
        R = np.zeros((self.Nr, U.shape[1]-1))   # Keep track of res states thru training loop
        for i in range(U.shape[1]-1): # -1 because we need to predict the NEXT state
            R[:,i] = (self.update_state(U[:,i])).reshape(-1)
        U = U[:,1:] # make sure R is aligned with the NEXT state of U

        return R
    
    def forecast(self, n):
        '''
        Forecast n time steps into the future. Starts with the last reservior state from training.

        INPUTS:
            n: The number of time steps to forecast
        OUTPUTS:
            U_pred: forecasted state of shape Nu x n
        ''' 
        # Feed predictions on u back into the RC for forecasting
        U_pred = np.zeros((self.Nu, n))
        for i in range(n):
            U_pred[:,i] = (self.W_out @ self.r).reshape(-1)
            self.update_state(U_pred[:,i])
        return U_pred
    
    def deepcopy(self):
        '''Returns a deepcopy of the ReservoirComputer'''
        return copy.deepcopy(self)

