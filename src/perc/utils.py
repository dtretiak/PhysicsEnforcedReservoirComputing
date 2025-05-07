import numpy as np

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