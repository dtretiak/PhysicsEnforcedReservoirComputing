import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection

import perc.utils as utils


def imshow_1D_spatiotemp(U, TN, domain, figsize=(20, 6), title = None, save_name = None, show = True, save_dpi = 300, cbar_label = r"$u$", xlabel = r'$t$',**kwargs):
    '''
    Plot 1D spatiotemporal data using imshow. 

    INPUTS:
        U: 2D array of shape (Nx, NT) where Nx is the number of spatial points and NT is the number of time points
        TN: final time
        domain: tuple of length 2 containing the bounds of the spatial domain
        figsize: tuple containing the size of the figure (default is (20, 6))
        title: string containing the title of the plot, if None no title is shown
        save_name: string containing the name of the file to save the plot, if None the plot is not saved
        **kwargs: additional arguments to pass to imshow
    '''

    #set defaults for imshow
    kwargs.setdefault('aspect', 'auto')
    kwargs.setdefault('origin', 'lower')
    kwargs.setdefault('cmap', 'icefire')
    kwargs.setdefault('interpolation', 'bicubic')
    kwargs.setdefault('extent', [0, TN, domain[0], domain[1]])

    plt.figure(figsize=figsize, dpi=150)
    plt.imshow(U, **kwargs)
    plt.ylabel(r'$x$')
    plt.xlabel(xlabel)
    if title is not None:
        plt.title(title)
    plt.colorbar(pad = 0.01, label = cbar_label)
    if save_name is not None:
        plt.savefig(save_name, transparent = True, dpi = save_dpi)
    if show:
        plt.show()

def imshow_1D_spatiotemp_3plots(U_list, TN, domain, figsize=(20, 18), colorbar_labels=None, colormaps=None, show = True, save_name = None, title = None, **kwargs):
    '''
    Plot 1D spatiotemporal data using imshow with 3 stacked subplots sharing the x-axis.

    INPUTS:
        U_list: List of 2D arrays [U1, U2, U3] each of shape (Nx, NT) where Nx is the number of spatial points and NT is the number of time points
        TN: Final time
        domain: Tuple of length 2 containing the bounds of the spatial domain
        figsize: Tuple containing the size of the figure (default is (20, 18))
        colorbar_labels: List of strings for the colorbar labels for each subplot
        colormaps: List of colormaps for each subplot
        **kwargs: Additional arguments to pass to imshow
    '''
    
    # Set defaults for imshow
    kwargs.setdefault('aspect', 'auto')
    kwargs.setdefault('origin', 'lower')
    kwargs.setdefault('interpolation', 'bicubic')
    kwargs.setdefault('extent', [0, TN, domain[0], domain[1]])
    
    # Create figure and 3 subplots sharing the same x-axis
    fig, axes = plt.subplots(3, 1, figsize=figsize, dpi=300, sharex=True)
    
    # Iterate over the subplots, data, and colormaps
    for i, (U, ax) in enumerate(zip(U_list, axes)):
        # Set the colormap for each subplot if provided
        cmap = colormaps[i] if colormaps and i < len(colormaps) else 'icefire'
        im = ax.imshow(U, cmap=cmap, **kwargs)
        ax.set_ylabel(r'$x$')
        
        # Add a colorbar with the specified label if provided
        cbar_label = colorbar_labels[i] if colorbar_labels and i < len(colorbar_labels) else r'$u$'
        fig.colorbar(im, ax=ax, pad=0.01, label=cbar_label)
    
    # Set the common x-axis label at the bottom
    axes[-1].set_xlabel(r'$t$')
    
    # title
    if title is not None:
        plt.suptitle(title)

    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name, transparent = True, dpi = 300)
    if show:
        plt.show()


def plot_time_series(U,t, labels=None, t_lim = None, figsize = (20,8), title = None, **kwargs):
    '''
    Plots time series data with separate panels for each state variable.

    INPUTS:
        U: 2D array of shape (Nu, Nt) where Nu is the number of state variables and Nt is the number of time points
        t: 1D array of time points
        labels: list of strings containing the names of the state variables
        t_lim: limit for the x-axis
        **kwargs: additional arguments to pass to plot
    '''
    if t_lim is None:
        t_lim = t[-1]
    kwargs.setdefault('linewidth', 2)
    fig, axs = plt.subplots(U.shape[0], figsize = figsize)
    for i in range(U.shape[0]):
        axs[i].plot(t, U[i,:], **kwargs)
        axs[i].set_xlim([0,t_lim])
        if labels is not None:
            axs[i].set(ylabel=labels[i])
    axs[-1].set(xlabel=r't')
    if title is not None:
        axs[0].set_title(title, fontsize=14)
    plt.show()

def plot_time_series_pred(U, U_pred, t, labels=None, t_lim = None, figsize = (20,8), **kwargs):
    '''
    Plots time series data with separate panels for each state variable. Also plots the predicted data.

    INPUTS:
        U: 2D array of shape (Nu, Nt) where Nu is the number of state variables and Nt is the number of time points
        U_pred: 2D array of shape (Nu, Nt) containing the predicted data
        t: 1D array of time points
        labels: list of strings containing the names of the state variables
        t_lim: limit for the x-axis
        **kwargs: additional arguments to pass to plot
    '''
    if t_lim is None:
        t_lim = t[-1]
    kwargs.setdefault('linewidth', 2)
    fig, axs = plt.subplots(U.shape[0], figsize = figsize)
    for i in range(U.shape[0]):
        axs[i].plot(t, U[i,:], label = 'True', **kwargs)
        axs[i].plot(t, U_pred[i,:], '--r', label = 'Pred', **kwargs)
        axs[i].set_xlim([0,t_lim])
        if labels is not None:
            axs[i].set(ylabel=labels[i])
    axs[-1].set(xlabel=r'$\lambda t$')
    plt.legend(loc = 'lower right')
    plt.show()


### Kol Flow Functions
def plot_vel_data(U, dt, num_cols = 5, show = True, save_name = None, **kwargs):
    '''
    Plot velocity magnitude at different time points

    INPUTS:
        U: Array of velocity data. Can be 4D of shape (n_samples, 2, N, N) or 2D of shape (n_samples, 2*N**2) where N is the number of spatial points
        num_cols: number of columns in plot
        **kwargs: additional arguments to pass to imshow

    '''
    # default kwargs 
    kwargs.setdefault('cmap', 'icefire')

    # handle flattened inputs
    if U.ndim == 2:
        U = utils.unflatten_vel_data(U)

    _, axs = plt.subplots(1, num_cols, figsize=(15, 5))
    for i in range(num_cols):
        idx = int(len(U) * (i/num_cols))
        U_mag = np.linalg.norm(U[idx, :, :, :], axis=0)
        axs[i].imshow(U_mag, **kwargs)
        axs[i].set_title(f'time {idx*dt:.2f}')
    plt.tight_layout()

    if save_name is not None:
        plt.savefig(save_name, transparent = True, dpi = 300)
    if show:
        plt.show()

def plot_vel_grid(U, t, n_rows, n_cols, spacing = "even", save_name = None, show = True, **imshow_kwargs):
    """
    Plot the velocity field U at different time points

    INPUTS:
        U: Array of velocity data. Can be 4D of shape (n_samples, 2, N, N) or 2D of shape (n_samples, 2*N**2) where N is the number of spatial points
        t: array, time points
        n_rows: int, number of rows in the plot
        n_cols: int, number of columns in the plot
        spacing: str or int, interval between time points to plot
            "even": plot the velocity field at even time points from t[0] to t[-1]
            "first_few": plot the velocity field at the first few time points up to t[n_rows * n_cols]
    """
    # handle flattened inputs
    if U.ndim == 2:
        U = utils.unflatten_vel_data(U)
    U_mag = np.linalg.norm(U, axis=1)
    plot_grid(U_mag, t, n_rows, n_cols, spacing, save_name, show **imshow_kwargs)



def plot_grid(U, t, n_rows, n_cols, spacing = "even", save_name = None, show = True, **imshow_kwargs):
    """
    Plot the of scalar field U at different time points

    INPUTS:
        U: array, scalar field of shape (t, grid_size, grid_size)
        t: array, time points
        n_rows: int, number of rows in the plot
        n_cols: int, number of columns in the plot
        spacing: str or int, interval between time points to plot
            "even": plot the scalar field at even time points from t[0] to t[-1]
            "first_few": plot the scalar field at the first few time points up to t[n_rows * n_cols]
    """
    # set defaults
    imshow_kwargs.setdefault('cmap', 'icefire')
    imshow_kwargs.setdefault('origin', 'lower')
    imshow_kwargs.setdefault('aspect', 'auto')
    imshow_kwargs.setdefault('interpolation', 'bicubic')

    if spacing == "even":
        time_points = np.linspace(0, len(t)-1, n_rows * n_cols, dtype=int)
    elif spacing == "first_few":
        time_points = np.arange(0, n_rows * n_cols)
    else:
        time_points = np.arange(0, len(t), spacing)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    for i, ax in zip(time_points, axs.flat):
        ax.set_aspect('equal', adjustable='box')
        im = ax.imshow(U[i], **imshow_kwargs)
        ax.set_title(f"t = {t[i]:.2f}")
        # ax.axis('off')
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name, transparent = True, dpi = 300)
    if show:
        plt.show()

def animate_grid(U, t, gif_name, frame_interval_factor=200, dpi=100, **imshow_kwargs):
    """
    Produces an animation of scalar field U
    """
    # Set defaults for imshow
    imshow_kwargs.setdefault('cmap', 'icefire')
    imshow_kwargs.setdefault('origin', 'lower')
    imshow_kwargs.setdefault('aspect', 'auto')
    imshow_kwargs.setdefault('interpolation', 'bicubic')

    dt = t[1] - t[0]
    fig, ax = plt.subplots()
    
    num_frames = len(U)
    interval = int(num_frames * frame_interval_factor)

    timestamp = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center')

    def update_frame(frame):
        # clear and update image
        ax.clear()
        cax = ax.imshow(U[frame], **imshow_kwargs)
        
        # Update the timestamp
        timestamp.set_text(f'Time: {t[frame]:.2f}')
        ax.set_title(f'Time: {t[frame]:.2f}')
        
        return cax, timestamp

    ani = animation.FuncAnimation(fig, update_frame, frames=num_frames, interval=interval)
    ani.save(f'{gif_name}.mp4', writer=FFMpegWriter(fps=1/dt), dpi=dpi, savefig_kwargs={'transparent': True})
    plt.show()


def plot_scalar_row(U, dt, cbar_label, show = True, show_title = False, save_name = None, save_dpi = 300, **kwargs):
    # U shape is n x 32 x 32
    kwargs.setdefault('cmap', 'icefire')
    kwargs.setdefault('origin', 'lower')
    # kwargs.setdefault('aspect', 'auto')
    kwargs.setdefault('interpolation', 'bicubic')

    # plot in a column
    num_cols = U.shape[0] 
    fig, axs = plt.subplots(1, num_cols, figsize=(5*num_cols, 5))
    for i in range(num_cols):
        idx = int(len(U) * (i/num_cols))
        im = axs[i].imshow(U[idx,:], **kwargs)
        if show_title:
            axs[i].set_title(f'time {idx*dt:.2f}')
        axs[i].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        if i == num_cols - 1:
            divider = make_axes_locatable(axs[i])
            cax = divider.append_axes("right", size="5%", pad=.05)
            plt.colorbar(im, cax=cax, label = cbar_label)

    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name, transparent = True, dpi = save_dpi)
    if show:
        plt.show()
    else:
        plt.close()


### Heat Equation Functions
def visualize_heat_equation(U, tN, L=1.0):
    """
    Visualize the results of the 1D heat equation simulation.
    
    INPUTS:
    U : np.ndarray
        Temperature distribution at each time step from heat_eq_1D_dirch.
    tN : float
        Final time.
    L : float, optional
        Domain length (default is 1.0).
    save_animation : bool, optional
        Whether to save animation as gif (default is False).
    filename : str, optional
        Filename for saved animation (default is 'heat_equation.gif').
    """
    Nx, Nt_plus_1 = U.shape
    Nt = Nt_plus_1 - 1
    x = np.linspace(0, L, Nx)
    t = np.linspace(0, tN, Nt + 1) 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # left panel: temperature evolution at different time points
    time_indices = [0, Nt//4, Nt//2, 3*Nt//4, Nt]
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))
    
    for i, idx in enumerate(time_indices):
        ax1.plot(x, U[:, idx], color=colors[i], label=f't = {t[idx]:.2f}')
    
    ax1.set_xlabel('Position x')
    ax1.set_ylabel('Temperature u(x,t)')
    ax1.set_title('Temperature Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # right panel: space-time heatmap
    T_mesh, X_mesh = np.meshgrid(t, x)
    im = ax2.contourf(T_mesh, X_mesh, U, levels=50, cmap='hot')
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Position x')
    ax2.set_title('Temperature Distribution (Space-Time)')
    plt.colorbar(im, ax=ax2, label='Temperature')
    plt.tight_layout()    
    plt.show()
    
    return fig

### MSD Visualization
def plot_phase_space(U):
    ''' Plot the phase space trajectory from a 2D array U.'''

    # Extract the x and y coordinates from U which is 2 x Nt
    x = U[0, :]
    y = U[1, :]
    points = np.array([x, y]).T
    segments = np.concatenate([points[:-1, None], points[1:, None]], axis=1)
    
    # set the colors according to time (based on segment order)
    lc = LineCollection(segments, cmap='plasma', norm=plt.Normalize(0, len(segments)))
    lc.set_array(np.arange(len(segments)))
    lc.set_linewidth(2)
    
    # plot
    fig, ax = plt.subplots(dpi = 150)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$p$")
    ax.set_title("Phase Space Trajectory")
    
    # add a colorbar to indicate time progression
    cbar = fig.colorbar(lc, ax=ax)
    cbar.set_label("Time")
    
    plt.show()
