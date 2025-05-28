import warnings
import copy

import numpy as np
import scipy
import scipy.linalg
import cvxpy as cp

class ReservoirComputer():
    '''
    Basic Reservoir Computer Implementation

    METHODS:
        update_state(u): update the reservoir state with input u (of shape Nu x 1), returns the next reservoir state r
        train(U): train the reservoir with input U (of shape Nu x N), returns the history of reservoir states R (of shape Nr x N-1)
        forecast(n): forecast n time steps into the future, returns the forecasted state U_pred (of shape Nu x n)
    '''

    def __init__(self,
                 Nu,
                 Nr = 1000,
                 alpha = 0.6,
                 beta = 8e-8,
                 rho_A = 0.1,
                 lambda_max = 0.8,
                 sigma=0.084,
                 bias = 1.6,
                 spinup = 500,
                 f = np.tanh,
                 quadratic = False):
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
            quadratic: if True, will add quadratic terms to the reservoir states (i.e. r^2)
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
        self.quadratic = quadratic

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
    
    def train(self, U, method = 'direct_solve'):
        '''
        Trains reservoir by fitting output weights W_out with ridge regression

        INPUTS:
            U: Nu x N matrix where Nu is the number of state variables and N is the number of samples in time
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
        R = np.zeros((self.Nr, U.shape[1]-1))   
        for i in range(U.shape[1]-1): 
            R[:,i] = (self.update_state(U[:,i])).reshape(-1)
        _ = self.update_state(U[:, -1]) 
        U = U[:,1:] # make sure R is aligned with the NEXT state of U

        # quadratic readout 
        if self.quadratic:
            R = np.concatenate((R, R**2), axis=0)

        # optimize with ridge regression || AX - B ||^2 + beta ||X||^2
        A = (R@R.T + self.beta * np.eye(R.shape[0]))
        B = R@U.T 
        
        # solve for W_out (X)
        if method == 'lstsq':
            X = scipy.linalg.lstsq(A, B, cond=None)[0].T
        elif method == 'direct_solve':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", scipy.linalg.LinAlgWarning)
                X = scipy.linalg.solve(A, B, assume_a='sym').T
        self.W_out = X

        return R #optional return for debugging
    
    def train_lh(self, U, C):
        '''
        Trains a Physics-Enforced Reservoir Computer (PERC) )by fitting output weights W_out with ridge regression
        constrained via linear homogeneous constraints CU = 0.

        INPUTS:
            U: Nu x N matrix where Nu is the number of state variables and N is the number of samples in time
            C: Nc x Nu matrix of linear homogeneous constraints; CU = 0, where Nc is the number of constraints

        OUTPUTS:
            R: history of reservoir states of shape Nr x N-1
        '''
        # synchronize the reservior
        for i in range(self.spinup):
            _  = self.update_state(U[:, i])
        U = U[:, self.spinup:]

        # Loop thru time
        R = np.zeros((self.Nr, U.shape[1]-1))   
        for i in range(U.shape[1]-1):
            R[:,i] = (self.update_state(U[:,i])).reshape(-1)
        _ = self.update_state(U[:, -1]) 
        U = U[:,1:] # make sure R is aligned with the NEXT state of U

        # quadratic readout
        if self.quadratic:
            R = np.concatenate((R, R**2), axis=0)

        # optimize with ridge regression || AX - B ||^2 + beta ||X||^2, s.t. CX^T = 0
        A = R.T
        B = U.T

        # use x in null space of C -> use silent part of QR to get basis for null space 
        r_C = np.linalg.matrix_rank(C)
        Q, _, _ = scipy.linalg.qr(C.T, pivoting = True)
        Q_silent = Q[:, r_C:]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", scipy.linalg.LinAlgWarning)
            Y_opt = scipy.linalg.solve(A.T @ A + self.beta * np.eye(A.shape[1]), A.T @ B @ Q_silent, assume_a='sym')
        X_opt = Y_opt @ Q_silent.T
        self.W_out = X_opt.T
        return  R
    
    def train_lih(self, U, C, d,
                      **cvx_kwargs):
        '''
        Trains a Physics-Enforced Reservoir Computer (PERC) by fitting output weights W_out with ridge regression
        constrained via linear inhomogeneous constraints CU = d.
        
        INPUTS:
            U: Nu x N matrix where Nu is the number of state variables and N is the number of samples in time
            C: Nc x Nu matrix of linear inhomogeneous constraints; CU = d, where Nc is the number of constraints
            d: Nc x 1 vector of inhomogeneous terms
            cvx_kwargs: additional keyword arguments for cvxpy solver
                defaults to:
                {
                    'solver': cp.CLARABEL, # cp.CLARABEL or cp.SCS are recommended
                    'warm_start': False,
                    'verbose': False
                }

        OUTPUTS:
            R: history of reservoir states of shape Nr x N-1
        '''       
        # cvx_kwarg defaults
        cvx_kwargs.setdefault('solver', cp.CLARABEL)
        cvx_kwargs.setdefault('warm_start', False)
        cvx_kwargs.setdefault('verbose', False)
        
        # synchronize the reservior
        for i in range(self.spinup):
            _  = self.update_state(U[:, i])
        U = U[:, self.spinup:]

        # Loop thru time
        R = np.zeros((self.Nr, U.shape[1]-1))  
        for i in range(U.shape[1]-1): 
            R[:,i] = (self.update_state(U[:,i])).reshape(-1)
        _ = self.update_state(U[:, -1]) 
        U = U[:,1:] # make sure R is aligned with the NEXT state of U

        # quadratic readout
        if self.quadratic:
            R = np.concatenate((R, R**2), axis=0)

        # new vars 
        R_tild = np.concatenate((R, np.ones((1,R.shape[1]))))
        C_tild = np.concatenate((C,-d), axis=1)
        U_tild = np.concatenate([U, np.ones((1, U.shape[1]))], axis = 0)
        C_2 = np.zeros((1,self.Nu+1))
        C_2[:,-1] = 1
        d_2 = np.zeros((1,R.shape[0]+1))
        d_2[:,-1] = 1

        # use x in null space of C -> use silent part of QR to get basis for null space 
        r_C = np.linalg.matrix_rank(C_tild)
        Q, _, _ = scipy.linalg.qr(C_tild.T, pivoting = True)
        Q_silent = Q[:, r_C:]

        # setup opt problem
        Y = cp.Variable((R.shape[0] + 1, self.Nu + 1 - r_C)) 
        objective = cp.Minimize(cp.norm(R_tild.T @ Y - U_tild.T@Q_silent, 'fro')**2 \
                                + self.beta*cp.norm(Y, 'fro')**2)
        constraints = [C_2 @ Q_silent @ Y.T == d_2]

        # warmstart with linear homogeneous solution 
        if cvx_kwargs['warm_start']:
            A = R_tild.T
            B = U_tild.T 
            Y_opt = scipy.linalg.solve(A.T @ A + self.beta * np.eye(A.shape[1]), A.T @ B @ Q_silent, assume_a='sym')
            Y.value = Y_opt

        # solve 
        prob = cp.Problem(objective, constraints)
        prob.solve(**cvx_kwargs)

        self.W_out = Q_silent @ Y.value.T
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
            if self.quadratic:
                r = np.concatenate((self.r, self.r**2), axis=0)
            else:
                r = self.r
            U_pred[:,i] = (self.W_out @ r).reshape(-1)
            self.update_state(U_pred[:,i])
        return U_pred
    
    def forecast_from_IC(self, n, IC):
        '''
        Forecast n time steps into the future. Obtain reservoir initial condition by spinning it 
        up with IC.

        INPUTS:
            n: The number of time steps to forecast
            IC: initial condition of shape Nu x eps. Will use all columns of IC to initialize the reservoir.
        OUTPUTS:
            U_pred: forecasted state of shape Nu x n
        ''' 
        # synchronize the reservior
        for i in range(IC.shape[1]):
            _  = self.update_state(IC[:, i])

        # Feed predictions on u back into the RC for forecasting
        U_pred = np.zeros((self.Nu, n))
        for i in range(n):
            if self.quadratic:
                r = np.concatenate((self.r, self.r**2), axis=0)
            else:
                r = self.r
            U_pred[:,i] = (self.W_out @ r).reshape(-1)
            self.update_state(U_pred[:,i])
        return U_pred
    
    def forecast_lih(self, n):
        '''
        Forecast n time steps into the future. Starts with the last reservior state from training.
        Applies linear inhomogeneous constraints CU = d learned from train_lih().

        INPUTS:
            n: The number of time steps to forecast
        OUTPUTS:
            U_pred: forecasted state of shape Nu x n
        ''' 
        # Feed predictions on u back into the RC for forecasting
        U_pred = np.zeros((self.Nu+1, n))
        for i in range(n):
            if self.quadratic:
                r = np.concatenate((self.r, self.r**2), axis=0)
            else:
                r = self.r
            r_tild = np.concatenate((r, np.ones((1,1))))
            U_pred[:,i] = (self.W_out @ r_tild).reshape(-1)
            self.update_state(U_pred[:-1,i])
        return U_pred
    
    def forecast_from_IC_lih(self, n, IC):
        '''
        Forecast n time steps into the future. Obtain reservoir initial condition by spinning it 
        up with IC. Applies linear inhomogeneous constraints CU = d learned from train_lih().

        INPUTS:
            n: The number of time steps to forecast
            IC: initial condition of shape Nu x eps. Will use all columns of IC to initialize the reservoir.
        OUTPUTS:
            U_pred: forecasted state of shape Nu x n
        ''' 
        # synchronize the reservior
        for i in range(IC.shape[1]):
            _  = self.update_state(IC[:, i])

        # Feed predictions on u back into the RC for forecasting
        U_pred = np.zeros((self.Nu+1, n))
        for i in range(n):
            if self.quadratic:
                r = np.concatenate((self.r, self.r**2), axis=0)
            else:
                r = self.r
            r_tild = np.concatenate((r, np.ones((1,1))))
            U_pred[:,i] = (self.W_out @ r_tild).reshape(-1)
            self.update_state(U_pred[:-1,i])
        return U_pred
    
    def deepcopy(self):
        '''Returns a deepcopy of the ReservoirComputer'''
        return copy.deepcopy(self)

