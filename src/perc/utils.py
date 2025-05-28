import numpy as np
import scipy.integrate

### overall utils
def get_benchmarks(U_test, U_pred):
    benchmarks = {}
    benchmarks['mse'] = np.mean((U_test - U_pred) ** 2, axis=0)
    benchmarks['rmse'] = np.sqrt(benchmarks['mse'])
    benchmarks['mse_cumul'] = np.cumsum(benchmarks['mse'])
    benchmarks['rmse_cumul'] = np.cumsum(benchmarks['rmse'])
    return benchmarks

### Kol Flow Functions
def flatten_vel_data(U):
    '''
    Flatten 2D velocity data into 2D array of shape (n_samples, n_features). u velocity data is concatenated with v velocity data. 

    INPUTS:
        U: 4D array of velocity data with shape (n_samples, 2, N, N) where N is the number of spatial points
    OUTPUTS:
        U_flat: 2D array of flattened velocity data with shape (n_samples, 2*N**2)
    '''
    u = U[:, 0, :, :]
    v = U[:, 1, :, :]
    u_flat = u.reshape(-1, u.shape[1]*u.shape[2])
    v_flat = v.reshape(-1, v.shape[1]*v.shape[2])
    return np.concatenate((u_flat, v_flat), axis=1)

def unflatten_vel_data(U_flat):
    '''
    Unflatten 2D velocity data from 2D array of shape (n_samples, n_features) to 4D array of shape (n_samples, 2, N, N)

    INPUTS:
        U_flat: 2D array of flattened velocity data with shape (n_samples, 2*N**2)
    OUTPUTS:
        U: 4D array of velocity data with shape (n_samples, 2, N, N) where N is the number of spatial points
    '''
    N = int(np.sqrt(U_flat.shape[1]//2))
    u_flat = U_flat[:, :N**2]
    v_flat = U_flat[:, N**2:]
    u = u_flat.reshape(-1, 1, N, N)
    v = v_flat.reshape(-1, 1, N, N)
    return np.concatenate((u, v), axis=1)

def unflatten_scalar_data(div):
    '''
    Unflatten scalar data from 1D array of shape (n_samples, N**2) to 3D array of shape (n_samples, N, N)

    INPUTS:
        div: 1D array of scalar data with shape (n_samples, N**2)
    OUTPUTS:
        div_3D: 3D array of scalar data with shape (n_samples, N, N) where N is the number of spatial points
    '''
    N = int(np.sqrt(div.shape[1]))
    return div.reshape(-1, N, N)

### Heat Equation Utils
def calc_total_heat(U, L=1.0):
    """
    Calculate the total heat content at each time step.
    
    Parameters:
    U : np.ndarray (size (Nx, Nt))
        Temperature distribution at each time step.
    L : float
        Domain length (default is 1.0).
    
    Returns:
    total_heat : np.ndarray
        Total heat content at each time step.
    """
    Nx, Nt = U.shape
    x = np.linspace(0, L, Nx)    
    total_heat = np.zeros(Nt)
    for i in range(Nt):
        total_heat[i] = scipy.integrate.trapezoid(U[:, i], x)
    return total_heat

def get_adiab_IC(params, total_heat, Nx, L=1.0):
    """
    Generate Gaussian-like initial conditions with fixed total heat content.
    
    Parameters:
    params : dict or array-like
        Parameters that control the shape of the initial condition.
        If dict, should contain:
            - 'center': center position (0 to 1)
            - 'width': width parameter (controls spread)
            - 'shape': shape parameter (controls peakedness)
        If array-like, should be [center, width, shape]
    total_heat : float
        Desired total heat content (integral over the domain).
    Nx : int
        Number of spatial points.
    L : float, optional
        Domain length (default is 1.0).
    
    Returns:
    u0 : np.ndarray (length Nx)
        Initial condition with the specified total heat content.
    """
    # handle parameters
    if isinstance(params, dict):
        center = params.get('center', 0.5)
        width = params.get('width', 0.1)
        shape = params.get('shape', 2.0)
    else:
        center = params[0] if len(params) > 0 else 0.5
        width = params[1] if len(params) > 1 else 0.1
        shape = params[2] if len(params) > 2 else 2.0
    
    # Ensure parameters are in valid ranges
    center = np.clip(center, 0.05, 0.95)  # Keep away from boundaries
    width = np.clip(width, 0.01, 0.5)     # Reasonable width range
    shape = np.clip(shape, 0.5, 10.0)     # Shape parameter range
    
    # Create spatial grid
    x = np.linspace(0, L, Nx)
    
    # Generate Gaussian-like distribution
    u_unnormalized = np.exp(-np.abs(x - center * L)**shape / (2 * width**shape))
    
    # Calculate current integral using trapezoidal rule
    current_integral = scipy.integrate.trapezoid(u_unnormalized, x)
    
    # Scale to achieve desired total heat
    if current_integral > 0:
        scaling_factor = total_heat / current_integral
        u0 = scaling_factor * u_unnormalized
    return u0