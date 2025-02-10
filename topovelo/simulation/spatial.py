from typing import Dict, List, Literal, Tuple
import numpy as np
import scipy as sp
from .utils import (
    draw_init_pos,
    draw_poisson_event,
    linear_v,
    disp_linear_v,
    const_v,
    disp_const_v,
    disp_binary_v,
    rotate
)


def spatial_dynamics_1d(n_cell: int,
                        n_init: int,
                        angle: float,
                        lamb: float = 10,
                        width: float = 1.0,
                        height: float = 1.0,
                        t_max: float = 20,
                        v0: float = 0.01,
                        velocity_model: Literal['const', 'linear'] = 'const',
                        geom_param: float = 0.8,
                        beta_param: Dict = {'a': 5.0, 'b': 5.0},
                        seed: int = 42,
                        **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate 1D spatial dynamics of cells

    Args:
        n_cell (int): number of cells
        n_init (int): number of initial cells
        angle (float): rotation angle
        lamb (float, optional): exponential rate parameter controling time interval. Defaults to 10.
        width (float, optional): width of the simulated tissue. Defaults to 1.0.
        height (float, optional): height of the simulated tissue. Defaults to 1.0.
        t_max (float, optional): maximum time. Defaults to 20.
        v0 (float, optional): initial velocity. Defaults to 0.01.
        velocity_model (str, optional): type of velocity model. Defaults to 'const'.
        geom_param (float, optional): parameter of geometric distribution controling number of descendants.
            Defaults to 0.8.
        beta_param (Dict, optional): parameter of beta distribution controling the probability of growing up/down
            in the y direction. Defaults to {'a': 5.0, 'b': 5.0}.
        seed (int, optional): random seed. Defaults to 42.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: simulated cell time, coordinates, and velocity
    """
    np.random.seed(seed)
    # Randomly sample initial conditions
    x0, y0 = draw_init_pos(n_init, 'linear', width, seed)
    batch_size = n_cell // n_init
    n_rem = n_cell - batch_size * n_init
    x, y = np.empty((n_cell)), np.empty((n_cell))
    vx, vy = np.empty((n_cell)), np.empty((n_cell))
    t = np.empty((n_cell))
    
    start = 0
    for i in range(n_init):
        n = batch_size + (i < n_rem)
        t_batch = draw_poisson_event(n, lamb, t_max, geom_param, np.random.randint(0, 4294967295))
        t[start: start+n] = t_batch
        if velocity_model == 'linear':  # v0 * t + 1/2 * k * t^2
            k = (height - v0*t_max) / (0.5 * t_max**2)
            vx_batch = linear_v(t_batch, k, v0)
            x[start: start+n] = disp_linear_v(t_batch, k, v0)
        else:  # constant velocity
            vx_batch = const_v(t_max, height)
            x[start: start+n] = x0[i] + disp_const_v(t_batch, vx_batch)
        # Sample vy
        max_vy = const_v(t_max, width*0.2)
        min_vy = const_v(t_max, width*0.02)
        vy_batch = np.random.uniform(min_vy, max_vy, size=n)
        p = sp.stats.beta.rvs(beta_param['a'], beta_param['b'])
        bern_samples = (np.random.uniform(size=(n,)) > p).astype(int)
        b = (bern_samples == 1) + (bern_samples == 0) * (-1)  # direction of growth in y axis

        y[start: start+n] = y0[i] + disp_binary_v(t_batch, vy_batch, b)
        vx[start: start+n] = vx_batch
        vy[start: start+n] = vy_batch * b
        
        start += n
    # Rotate by an angle
    coords = rotate(np.stack([x, y], 1), angle)
    v = rotate(np.array(np.stack([vx, vy], 1)), angle)
    return t, coords, v


def spatial_dynamics_1d_bi(n_cell: int,
                           n_init: int,
                           angle: float,
                           lamb: float = 10,
                           width: float = 1.0,
                           height: Tuple[float, float] = (1.0, 1.0),
                           t_max: float = 20,
                           v0: Tuple[float, float] = (0.01, 0.1),
                           velocity_model: Literal['const', 'linear'] = 'const',
                           geom_param: float = 0.8,
                           beta_param: Dict = {'a': 5.0, 'b': 5.0},
                           seed: int = 42,
                           **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Generate 1D spatial dynamics of cells with two different growth directions.  
    
    Args:
        n_cell (int): number of cells
        n_init (int): number of initial cells
        angle (float): rotation angle
        lamb (float, optional): exponential rate parameter controling time interval. Defaults to 10.
        width (float, optional): width of the simulated tissue. Defaults to 1.0.
        height (float, optional): height of the simulated tissue. Defaults to 1.0.
        t_max (float, optional): maximum time. Defaults to 20.
        v0 (float, optional): initial velocity. Defaults to 0.01.
        velocity_model (str, optional): type of velocity model. Defaults to 'const'.
        geom_param (float, optional): parameter of geometric distribution controling number of descendants.
            Defaults to 0.8.
        beta_param (Dict, optional): parameter of beta distribution controling the probability of growing up/down
            in the y direction. Defaults to {'a': 5.0, 'b': 5.0}.
        seed (int, optional): random seed. Defaults to 42.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: simulated cell time, coordinates, and velocity
    """
    np.random.seed(seed)
    # Randomly sample initial conditions
    seed_1 = np.random.randint(0, 4294967295)
    t_1, coords_1, v_1 = spatial_dynamics_1d(n_cell//2,
                                             n_init//2,
                                             angle,
                                             lamb,
                                             width,
                                             height[0],
                                             t_max,
                                             v0[0],
                                             velocity_model,
                                             geom_param,
                                             beta_param,
                                             seed_1,
                                             **kwargs)
    seed_2 = np.random.randint(0, 4294967295)
    t_2, coords_2, v_2 = spatial_dynamics_1d(n_cell-n_cell//2,
                                             n_init-n_init//2,
                                             angle+np.pi,
                                             lamb,
                                             width,
                                             height[1],
                                             t_max,
                                             v0[1],
                                             velocity_model,
                                             geom_param,
                                             beta_param,
                                             seed_2,
                                             **kwargs)
    return np.concatenate((t_1, t_2)), np.concatenate((coords_1, coords_2)), np.concatenate((v_1, v_2))


def spatial_dynamics_radial(n_cell: int,
                            n_init: int,
                            lamb: float = 10,
                            r_max: float = 1.0,
                            r_init: float = 0.01,
                            t_max: float = 20,
                            vr_0: float = 0.01,
                            velocity_model: Literal['const', 'linear'] = 'const',
                            geom_param: float = 0.8,
                            beta_param: Dict = {'a': 5.0, 'b': 5.0},
                            seed=42,
                            **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate radial spatial dynamics

    Args:
        n_cell (int): number of cells
        n_init (int): number of initial cells
        lamb (float, optional): exponential rate parameter controling time interval. Defaults to 10.
        r_max (float, optional): maximum radius. Defaults to 1.0.
        r_init (float, optional): initial radius in the polar coordinate. Defaults to 0.01.
        t_max (float, optional): maximum time. Defaults to 20.
        vr_0 (float, optional): initial radial velocity. Defaults to 0.01.
        velocity_model (Literal['const', 'str'], optional): type of velocity model. Defaults to 'const'.
        geom_param (float, optional): parameter of geometric distribution controling number of descendants.
            Defaults to 0.8.
        beta_param (Dict, optional): parameter of beta distribution controling the probability of growing up/down
            in the y direction. Defaults to {'a': 5.0, 'b': 5.0}.
        seed (int, optional): random seed. Defaults to 42.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: simulated cell time, coordinates, and velocity
    """
    np.random.seed(seed)
    # Randomly sample initial conditions
    r0, theta0 = draw_init_pos(n_init, 'disk', r_init, seed)
    batch_size = n_cell // n_init
    n_rem = n_cell - batch_size * n_init
    r, theta = np.empty((n_cell)), np.empty((n_cell))
    vr, vtheta = np.empty((n_cell)), np.empty((n_cell))
    t = np.empty((n_cell))
    
    start = 0
    for i in range(n_init):
        n = batch_size + (i < n_rem)
        t_batch = draw_poisson_event(n, lamb, t_max, geom_param, np.random.randint(0, 4294967295))
        t[start: start+n] = t_batch            
        if velocity_model == 'linear':
            k = (r_max - vr_0*t_max) / (0.5 * t_max**2)
            vr_batch = linear_v(t_batch, k, vr_0)
            r[start: start+n] = disp_linear_v(t_batch, k, vr_0)
        else:  # constant velocity
            vr_batch = const_v(t_max, r_max)
            r[start: start+n] = r0[i] + disp_const_v(t_batch, vr_batch)
        # Sample angular velocity
        max_vtheta = const_v(t_max, np.pi/2)
        min_vtheta = const_v(t_max, np.pi/12)
        vtheta_batch = np.random.uniform(min_vtheta, max_vtheta, size=n)
        p = sp.stats.beta.rvs(beta_param['a'], beta_param['b'])
        bern_samples = (np.random.uniform(size=(n,)) > p).astype(int)
        b = (bern_samples == 0) + (bern_samples == 1) * (-1)  # direction of growth in y axis
        theta[start: start+n] = theta0[i] + disp_binary_v(t_batch, vtheta_batch, b)
        vr[start: start+n] = vr_batch
        vtheta[start: start+n] = vtheta_batch * b
        
        start += n
    # Transform back to cartesian coordinates
    coords = np.stack([r*np.cos(theta), r*np.sin(theta)], 1)
    v = np.stack([vr*np.cos(theta)-vtheta*r*np.sin(theta),
                  vr*np.sin(theta)+vtheta*r*np.cos(theta)], 1)
    return t, coords, v


def spatial_dynamics_multiradial(n_cell: int,
                                 n_init: int,
                                 n_part: int,
                                 centers: List[np.ndarray],
                                 lamb: float = 10,
                                 r_max: float = 1.0,
                                 r_init: float = 0.01,
                                 t_max: float = 20,
                                 vr_0: float = 0.01,
                                 velocity_model: Literal['const', 'linear'] = 'const',
                                 geom_param: float = 0.8,
                                 beta_param: Dict = {'a': 5.0, 'b': 5.0},
                                 seed: int = 42,
                                 **kwargs) -> Tuple:
    """Simulate radial spatial dynamics from multiple centers

    Args:
        n_cell (int): number of cells.
        n_init (int): number of initial cells.
        n_part (int): number of partitions (disks).
        centers (List[np.ndarray]): centers of the disks.
        lamb (float, optional): _description_. Defaults to 10.
        r_max (float, optional): _description_. Defaults to 1.0.
        r_init (float, optional): _description_. Defaults to 0.01.
        t_max (float, optional): _description_. Defaults to 20.
        vr_0 (float, optional): _description_. Defaults to 0.01.
        velocity_model (Literal['const', 'linear'], optional): _description_. Defaults to 'const'.
        geom_param (float, optional): parameter of geometric distribution controling number of descendants.
            Defaults to 0.8.
        beta_param (Dict, optional): parameter of beta distribution controling the probability of growing up/down
            in the y direction. Defaults to {'a': 5.0, 'b': 5.0}.
        seed (int, optional): _description_. Defaults to 42.

    Returns:
        Tuple: simulated cell time, coordinates, and velocity
    """
    np.random.seed(seed)
    seed = np.random.randint(0, 4294967295)
    m = n_cell % n_part
    m0 = n_init % n_part
    t, coords, v = [], [], []
    for i in range(n_part):
        seed = np.random.randint(0, 4294967295)
        t_, coords_, v_ = spatial_dynamics_radial(n_cell//n_part+(i < m),
                                                  n_init//n_part+(i < m0),
                                                  lamb,
                                                  r_max,
                                                  r_init,
                                                  t_max,
                                                  vr_0,
                                                  velocity_model,
                                                  geom_param,
                                                  beta_param,
                                                  seed=seed,
                                                  **kwargs)
        coords_ += centers[i]
        t.append(t_)
        coords.append(coords_)
        v.append(v_)
    return t, coords, v
