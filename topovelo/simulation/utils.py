from typing import List, Tuple
import numpy as np
import scipy as sp


####################################################################################################
# Utility functions for spatial coordinates
####################################################################################################
def rotate(coords: np.ndarray, angle: float) -> np.ndarray:
    """ Rotate (x, y) by angle counter-clockwise

    Args:
        coords (np.ndarray): 1-D array of shape (2,) containing (x, y) coordinates
        angle (float): rotation angle in radian

    Returns:
        np.ndarray: rotated coordinates
    """
    rot_mtx = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return np.einsum('ij, kj->ki', rot_mtx, coords)


def rotate_cov(cov: np.ndarray, angle: float) -> np.ndarray:
    """ Rotate covariance matrix by angle counter-clockwise

    Args:
        cov (np.ndarray): 2-D array of shape (2, 2) containing covariance matrix
        angle (float): rotation angle in radian

    Returns:
        np.ndarray: rotated covariance matrix
    """
    rot_mtx = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return np.matmul(rot_mtx, np.matmul(cov, rot_mtx.T))


def draw_poisson_event(n: int, lamb: float, t_max: float, geom_param: float = 0.8, seed: int = 42) -> np.ndarray:
    """Draw cell time following a Poisson process with multiple descendants.

    Args:
        n (int): number of samples
        lamb (float): exponential rate
        t_max (float): maximum time
        geom_param (float, optional): parameter of the Geometric distribution controling number of descendants.
            Defaults to 0.8.
        seed (int, optional): random seed. Defaults to 42.

    Returns:
        np.ndarray: sampled cell time
    """
    np.random.seed(seed)
    count = 1
    t = [0]
    n_par = 1
    while count < n:
        n_des = sp.stats.geom.rvs(geom_param, size=n_par)
        n_par_next = np.sum(n_des)
        # Handle the last level to make sure exactly n cells are generated
        if count + n_par_next > n:
            n_par = n - count
            n_par_next = n - count
            n_des = np.array([1]*n_par_next)
        uniform_samples = np.random.uniform(size=(n_par_next,))
        # Apply inverse of CDF
        delta_t = -1/lamb*np.log(1-uniform_samples)
        start = 0
        for i in range(n_par):
            t.extend(list(t[-i-1]+delta_t[start:start+n_des[i]]))
            start += n_des[i]
        count += n_par_next
        n_par = n_par_next
    t = np.array(t)
    t = t / t.max() * t_max
    assert len(t) == n, f"Number of cells {len(t)} is not equal to {n}"
    return t


def draw_init_pos(n: int,
                  style: str = 'linear',
                  d: float = 1.0,
                  seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Draw initial positions of cells

    Args:
        n (int): number of cells
        style (str, optional): type of spatial growth. Defaults to 'linear'.
        d (float, optional): . Defaults to 1.0.
        seed (int, optional): random seed. Defaults to 42.

    Returns:
        Tuple[np.ndarray, np.ndarray]: initial positions either in cartesian or polar coordinates
    
    Raises:
        NotImplementedError: if style is not implemented
    """
    np.random.seed(seed)
    if style == 'linear':  # uniformly draw cells on a line
        return np.random.normal(scale=d*0.01, size=(n,)), np.random.uniform(-d/2, d/2, size=(n,))
    elif style == 'disk':  # uniformly draw cells on a disk in polar coordinates
        theta = np.random.uniform(0, 2*np.pi, size=(n))
        r = np.random.uniform(0, d/2, size=(n))
        return r, theta
    raise NotImplementedError(f"style {style} is not implemented.")


def cart2polar(coords: np.ndarray) -> np.ndarray:
    """Convert Cartesian coordinates to polar coordinates

    Args:
        coords (np.ndarray): 2-D array of shape (n, 2) containing (x, y) coordinates

    Returns:
        np.ndarray: 2-D array of shape (n, 2) containing (r, theta) coordinates
    """
    r = np.linalg.norm(coords, axis=1)
    theta = np.arctan2(coords[:, 1], coords[:, 0])
    theta[np.isnan(theta)] = 0
    return np.stack([r, theta], 1)


def polar2cart(coords: np.ndarray) -> np.ndarray:
    """Convert polar coordinates to Cartesian coordinates

    Args:
        coords (np.ndarray): 2-D array of shape (n, 2) containing (r, theta) coordinates

    Returns:
        np.ndarray: 2-D array of shape (n, 2) containing (x, y) coordinates
    """
    return np.stack([coords[:, 0]*np.cos(coords[:, 1]),
                     coords[:, 0]*np.sin(coords[:, 1])], 1)


####################################################################################################
# Velocity model
####################################################################################################
def const_v(t_max: float, d: float) -> float:
    """Constant velocity model

    Args:
        t_max (float): total time
        d (float): distance

    Returns:
        float: constant velocity
    """
    return d / t_max


def linear_v(t: np.ndarray, k: float, v0: float) -> np.ndarray:
    """Linear velocity model, e.g. constant acceleration

    Args:
        t (np.ndarray): time
        k (float): acceleration
        v0 (float): initial velocity

    Returns:
        np.ndarray: velocity
    """
    return k*t + v0


####################################################################################################
# Displacement
####################################################################################################
def disp_const_v(t: np.ndarray, v: float) -> np.ndarray:
    """Displacement under constant velocity

    Args:
        t (np.ndarray): time
        v (float): velocity

    Returns:
        np.ndarray: displacement
    """
    return v*t


def disp_binary_v(t: np.ndarray, v: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Displacement under constant velocity magnitude with binary direction 

    Args:
        t (np.ndarray): time. Has to be sorted.
        v (float): velocity
        direction (np.ndarray): direction (+ or -)

    Returns:
        np.ndarray: displacement
    """
    delta_disp = np.diff(np.concatenate((np.array([0]), t)))*v*direction
    return np.cumsum(delta_disp)


def disp_linear_v(t: np.ndarray, k: float, v0: float) -> np.ndarray:
    """Displacement under constant acceleration

    Args:
        t (np.ndarray): time
        k (float): acceleration
        v0 (float): initial velocity

    Returns:
        np.ndarray: displacement
    """
    return 0.5*k*t**2 + v0*t


####################################################################################################
# lower resolution
####################################################################################################
def low_res_partition(coords: np.ndarray, low_res: float) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Partition spatial coordinates into low resolution

    Args:
        coords (np.ndarray): 2-D array of shape (n, 2) containing (x, y) coordinates
        low_res (float): low resolution

    Returns:
        Tuple[np.ndarray, List[np.ndarray]]: discretized (x, y) coordinates and indices of partitioned cells
    """
    grid = (coords // low_res).astype(int)
    min_x, max_x = grid[:, 0].min(), grid[:, 0].max()
    min_y, max_y = grid[:, 1].min(), grid[:, 1].max()
    coords_low_res = []
    groups = []
    for i in range(min_x, max_x+1):
        for j in range(min_y, max_y+1):
            idx = np.where((grid[:, 0] == i) & (grid[:, 1] == j))[0]
            if len(idx) > 0:
                coords_low_res.append([i * low_res, j * low_res])
                groups.append(idx)
    return np.array(coords_low_res), groups


####################################################################################################
# Evaluation
####################################################################################################
def vel_accuracy(adata, vkey):
    """  Compute the cosine similarity between the true cell velocity and the predicted cell velocity """
    true_v = adata.obsm['true_spatial_velocity']
    v = adata.obsm[vkey]
    norm_1 = np.linalg.norm(true_v, axis=1)
    norm_2 = np.linalg.norm(v, axis=1)
    zero_mask = (norm_1 * norm_2 == 0).astype(int)
    return np.mean((true_v * v).sum(1)/(zero_mask * 1 + (1-zero_mask) * norm_1 * norm_2))


def pseudo_celltype(adata, n_type) -> List[Tuple[str, str]]:
    t = adata.obs['true_time'].to_numpy()
    tmax, tmin = t.max(), t.min()
    delta_t = (tmax - tmin) / n_type
    cell_labels = np.array(['0']*adata.n_obs)
    for i in range(n_type - 1):
        cell_labels[(t >= tmin+delta_t*i) & (t < tmin+delta_t*(i+1))] = str(i)
    cell_labels[t >= tmin+delta_t*(n_type-1)] = f'{n_type-1}'
    adata.obs['clusters'] = cell_labels
    return [(f'{i}', f'{i+1}') for i in range(n_type-1)]
