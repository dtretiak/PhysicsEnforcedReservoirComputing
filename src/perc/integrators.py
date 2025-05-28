import numpy as np 
import scipy
import scipy.integrate


def lorentz63(x0, dt, t0, tN, sigma=10, rho=28, beta=8/3, rtol = 1e-12, atol = 1e-12, method = 'RK45', dense_output = True):
    '''
    Integrate the Lorentz 63 system with Scipy Integration Methods

    Inputs:
        x0: initial condition iterable of length 3
        sigma, rho, beta: parameters
        dt: time step in output time series. NOT integration time step (determined with Scipy Integration Methods)
        tmax: maximum time to integrate
        rtol: relative tolerance for scipy integrators
        atol: absolute tolerance for scipy integrators
        method: integration method (see scipy.integrate.solve_ivp), default is 'RK45'
        dense_output: if True, return dense output. If False, return sparse output determined by adaptive timestepping


    Outputs:
        t: time series of shape (tN/dt)
        x_t: time series of the solution of shape (3, tN/dt) in the form (x,y,z)

    Sample Usage:
        t, x_t = lorentz63(x0 = [1,1,1], dt=0.001, t0 = 0, tN=25)
        x,y,z = x_t
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x,y,z)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
    '''

    # xdot = f(t,x) defines the system. x is a vector of 3 state variables
    def f(t,x):
        x1, x2, x3 = x
        dx1dt = sigma*(x2 - x1)
        dx2dt = x1*(rho - x3) - x2
        dx3dt = x1*x2 - beta*x3
        return np.array([dx1dt, dx2dt, dx3dt])

    # integrate the system
    if dense_output == True:
        sol = scipy.integrate.solve_ivp(f, t_span=(t0,tN), y0=x0, dense_output=True, rtol = rtol, atol = atol, method = method)
        t = np.arange(t0,tN, dt)
        x_t = sol.sol(t)
    else:
        sol = scipy.integrate.solve_ivp(f, t_span=(t0,tN), y0=x0, rtol = rtol, atol = atol, method = method)
        t = sol.t
        x_t = sol.y
    return x_t, t

def KS_1D_PBC(u0, tN, dt=0.1, domain=(0, 100), Nx=200):
    '''
    Solve 1D Kuramoto-Sivashinsky Equation with Periodic Boundary Conditions using ETDRK4 scheme.
    Incorporates dealiasing and enforces conservation of mean (integral over domain).
    
    MODEL:
        u_t + u*u_x + u_xx + nu*u_xxxx = 0

    INPUTS:
        u0: initial condition of shape (N,)
        tN: final time
        dt: time step
        domain: (x0, xN) left and right boundaries of the domain
        Nx: number of spatial points (including periodic point, which is removed)

    OUTPUTS:
        t: array of time points
        U: solution array of shape (K, N)
    '''
    # Setup spatial grid
    Nx = Nx - 1  # remove duplicate periodic point
    u0 = u0[:-1]
    x = np.linspace(domain[0], domain[1], Nx, endpoint=False)
    dx = x[1] - x[0]
    
    Nt = int(tN / dt)
    U = np.zeros((Nx, Nt))
    U[:, 0] = u0

    # Wavenumbers and operators
    k = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
    k2 = k**2
    k4 = k**4
    L_op = k2 - k4

    # Dealiasing (2/3 rule)
    def dealias(f_hat):
        cutoff = Nx // 3
        f_hat[cutoff:-cutoff] = 0
        return f_hat

    N_op_u = lambda u: dealias(1j * k * np.fft.fft(-0.5 * u ** 2))
    N_op_uhat = lambda u_hat: dealias(1j * k * np.fft.fft(-0.5 * np.real(np.fft.ifft(u_hat)) ** 2))

    # ETDRK4 coefficients (Kassam & Trefethen 2005)
    E1 = np.exp(L_op * dt)
    E2 = np.exp(L_op * dt / 2)
    M = 16
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * np.column_stack([L_op]*M) + np.row_stack([r]*Nx)
    Q = dt * np.mean((np.exp(LR/2) - 1) / LR, axis=1)
    f1 = dt * np.mean((-4 - LR + np.exp(LR)*(4 - 3*LR + LR**2)) / LR**3, axis=1)
    f2 = dt * np.mean((2 + LR + np.exp(LR)*(-2 + LR)) / LR**3, axis=1)
    f3 = dt * np.mean((-4 - 3*LR - LR**2 + np.exp(LR)*(4 - LR)) / LR**3, axis=1)

    # Time stepping loop
    for i in range(Nt - 1):
        u = U[:, i]
        u_hat = np.fft.fft(u)

        a = E2 * u_hat + Q * N_op_u(u)
        b = E2 * u_hat + Q * N_op_uhat(a)
        c = E2 * a + Q * (2*N_op_uhat(b) - N_op_u(u))

        u_hat = E1 * u_hat + f1 * N_op_u(u) + f2 * (N_op_uhat(a) + N_op_uhat(b)) + f3 * N_op_uhat(c)

        # Enforce conservation by zeroing the mean mode
        u_hat[0] = 0.0
        U[:, i+1] = np.real(np.fft.ifft(u_hat, n=Nx))

    # Add back periodic boundary point for output
    U = np.vstack((U, U[0, :]))
    t = np.arange(0, tN, dt)

    return U, t

def heat_eq_1D_adiab(u0, tN, dt=0.1, Nx=50, alpha=0.1):
    """
    Solve the 1D heat equation with insulating boundary conditions (Neumann).
    
    INPUTS:
    u0 : np.ndarray
        Initial condition (temperature distribution).
    tN : float
        Final time.
    dt : float, optional
        Time step size (default is 0.1).
    Nx : int, optional
        Number of spatial points (default is 50).
    alpha : float, optional
        Thermal diffusivity (default is 0.1).
    
    OUTPUTS:
    U : np.ndarray (size (Nx, len(t_eval)))
        Temperature distribution at each time step.
    """
    # Spatial domain parameters
    L = 1.0  # Domain length
    dx = L / (Nx - 1)
    x = np.linspace(0, L, Nx)
    
    # Set up initial condition
    if len(u0) == Nx:
        u_init = u0.copy()
    else:
        # Interpolate initial condition to match grid
        x_init = np.linspace(0, L, len(u0))
        u_init = np.interp(x, x_init, u0)
    
    def heat_rhs_adiabatic(t, u):
        """
        RHS for the heat equation discretized with finite differences.
        du/dt = alpha * d²u/dx²

        BC: du/dx = 0 at x=0 and x=L
        """
        dudt = np.zeros_like(u)
        
        # Interior points: second derivative using central difference
        dudt[1:-1] = alpha * (u[2:] - 2*u[1:-1] + u[:-2]) / (dx**2)
        
        # left BC
        dudt[0] = alpha * 2 * (u[1] - u[0]) / (dx**2)
        
        # right BC
        dudt[-1] = alpha * 2 * (u[-2] - u[-1]) / (dx**2)
        
        return dudt
    
    # Time evaluation points
    t_eval = np.arange(0, tN + dt, dt)
    
    sol = scipy.integrate.solve_ivp(
            heat_rhs_adiabatic,
            [0, tN],
            u_init,
            t_eval=t_eval,
            method='RK45',  
            rtol=1e-12,
            atol=1e-12
        )
    
    U = sol.y  
    return U