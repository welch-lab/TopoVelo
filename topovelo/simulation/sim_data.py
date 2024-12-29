from typing import Dict, List, Literal, Optional, Tuple, Union
import anndata
import numpy as np
import scipy as sp
from .spatial import (
    spatial_dynamics_1d,
    spatial_dynamics_1d_bi,
    spatial_dynamics_radial,
    spatial_dynamics_multiradial
)
from .rna import simulate_rna
from .utils import low_res_partition


def simulate(n_cell: int,
             n_gene: int,
             n_init_cell: int,
             n_repress_gene: int,
             sparsity: Tuple[float, float],
             angle: float,
             d: float = 1.0,
             t_max: float = 20,
             growth_type: Literal['1d', '1d (bidirection)', 'radial', 'multi-radial'] = '1d',
             velocity_model: Literal['const', 'linear'] = 'const',
             v0: Union[float, Tuple[float, float]] = 0.01,
             geom_param: float = 0.5,
             beta_param: Dict = {'a': 5.0, 'b': 5.0},
             mu_param: np.ndarray = np.array([1.0, 0.0, 0.0]),
             cov_param: np.ndarray = np.array([[1.0, 0.05, 0.05],
                                               [0.05, 1.0, 0.05],
                                               [0.05, 0.05, 1.0]]),
             noise_level: float = 0.5,
             noise_type: Literal['Gaussian', 'Spatial Gaussian'] = 'Gaussian',
             n_neighbors: int = 16,
             ton_cutoff: List[float] = [0.0, 0.1],
             toff_cutoff: List[float] = [0.3, 1.0],
             rho_spatial_pattern: Literal['Gaussian', 'Polar Gaussian', 'Gaussian (Bidirection)'] = 'Gaussian',
             n_part: int = 4,
             centers: Optional[List[np.ndarray]] = None,
             seed: int = 42) -> anndata.AnnData:
    """Simulate spatial RNA data

    Args:
        n_cell (int): number of cells.
        n_gene (int): number of genes.
        n_init_cell (int): number of root cells.
        n_repress_gene (int): number of repressive genes.
        sparsity (Tuple[float, float]): sparsity of unspliced and spliced matrices.
        angle (float): angle of the growth direction.
        d (float, optional): width in linear growth or radius of a disk in radial growth. Defaults to 1.0.
        t_max (float, optional): maximum time. Defaults to 20.
        growth_type (Literal['1d', '1d (bidirection)', 'radial', 'multi-radial'], optional):
            type of spatial dynamics. Defaults to '1d'.
        velocity_model (Literal['const', 'linear'], optional):
            cell velocity model. Defaults to 'const'.
        v0 (Union[float, Tuple[float, float]], optional): initial velocity. Defaults to 0.01.
        geom_param (float, optional): parameter of geometric distribution controling number of descendants.
            Defaults to 0.8.
        beta_param (Dict, optional): parameter of beta distribution controling the probability of growing up/down
            in the y direction. Defaults to {'a': 5.0, 'b': 5.0}.
        mu_param (np.ndarray): mean of the log normal distribution for transcription, splicing and degradation rates.
        cov_param (np.ndarray): covariance matrix of the log normal distribution for transcription,
            splicing and degradation rates.
        noise_level (float, optional): Gaussian noise level. Defaults to 0.1.
        noise_type (Literal['Gaussian', 'Spatial Gaussian'], optional): . Defaults to 'Gaussian'.
        n_neighbors (int, optional): number of neighbors used for KNN smoothing of Gaussian noise.
            Effective only if `noise_type='Spatial Gaussian'`. Defaults to 16.
        ton_cutoff (List[float], optional): lower and upper quantile limit of switch-on time
            in the RNA velocity model. Defaults to [0.0, 0.2].
        toff_cutoff (List[float], optional): lower and upper quantile limit of switch-off time
            in the RNA velocity model. Defaults to [0.3, 1.0].
        rho_spatial_pattern (Literal['Gaussian', 'Polar Gaussian', 'Gaussian (Bidirection)'], optional):
            Type of spatial pattern for spatially variable transcription rates. Defaults to 'Gaussian'.
        centers (Optional[List[np.ndarray]], optional):
            Coordinates of disk centers. Effective only if `rho_spatial_pattern=Polar Gaussian`. Defaults to None.
        seed (int, optional): random seed. Defaults to 42.

    Raises:
        NotImplementedError: growth_type, velocity model, noise_type or rho_spatial_pattern is not supported

    Returns:
        AnnData
    """
    # Determine spatial dynamics
    if growth_type == '1d':
        assert isinstance(v0, float), 'v0 must be a float for 1d growth'
        t, coords, v = spatial_dynamics_1d(
            n_cell,
            n_init_cell,
            angle, 
            width=d,
            height=d,
            t_max=t_max,
            v0=v0,
            velocity_model=velocity_model,
            geom_param=geom_param,
            beta_param=beta_param,
            seed=seed
        )
    elif growth_type == '1d (bidirection)':
        assert isinstance(v0, tuple), 'v0 must be a tuple for 1d (bidirection) growth'
        t, coords, v = spatial_dynamics_1d_bi(
            n_cell,
            n_init_cell,
            angle, 
            width=d,
            height=(d, d),
            t_max=t_max,
            v0=v0,
            velocity_model=velocity_model,
            geom_param=geom_param,
            beta_param=beta_param,
            seed=seed
        )
    elif growth_type == 'radial':
        assert isinstance(v0, float), 'v0 must be a float for radial growth'
        t, coords, v = spatial_dynamics_radial(
            n_cell,
            n_init_cell,
            r_max=d,
            r_init=d*0.01,
            t_max=20,
            vr_0=v0,
            velocity_model=velocity_model,
            geom_param=geom_param,
            beta_param=beta_param,
            seed=seed
        )
    elif growth_type == 'multi-radial':
        assert centers is not None, 'centers must be provided for multi-radial growth'
        assert isinstance(v0, float), 'v0 must be a float for multi-radial growth'
        t, coords, v = spatial_dynamics_multiradial(
            n_cell,
            n_init_cell,
            n_part,
            centers,
            r_max=d,
            r_init=d*0.01,
            t_max=t_max,
            vr_0=v0,
            velocity_model=velocity_model,
            geom_param=geom_param,
            beta_param=beta_param,
            seed=seed
        )
    else:
        raise NotImplementedError(f'growth_type {growth_type} is not supported')

    # Splicing dynamics
    u, s, rho, rates, ton, toff, is_repressive = simulate_rna(
        t,
        coords,
        n_gene,
        n_repress_gene,
        mu_param,
        cov_param,
        t_max,
        sparsity,
        angle,
        noise_level=noise_level,
        noise_type=noise_type,
        n_neighbors=n_neighbors,
        rho_spatial_pattern=rho_spatial_pattern,
        ton_cutoff=ton_cutoff,
        toff_cutoff=toff_cutoff,
        centers=centers
    )
    # don't forget to concatenate the arrays since multi-radial generates each disk separately
    if growth_type == 'multi-radial':
        t = np.concatenate(t)
        coords = np.concatenate(coords)
        v = np.concatenate(v)    
    
    obs = {
        'true_time': t
    }
    
    var = {
        'true_alpha': rates[:, 0],
        'true_beta': rates[:, 1],
        'true_gamma': rates[:, 2],
        'true_ton': ton,
        'true_toff': toff,
    }
    if is_repressive is not None:
        var['repressive_genes'] = is_repressive
    
    layers = {
        'unspliced': sp.sparse.csr_matrix(u),
        'spliced': sp.sparse.csr_matrix(s),
        'true_rho': rho
    }
    
    adata = anndata.AnnData(s, obs, var, layers=layers)
    adata.obsm['true_spatial_velocity'] = v
    adata.obsm['X_spatial'] = coords

    return adata


def to_low_res(adata: anndata.AnnData, low_res: float) -> anndata.AnnData:
    """Aggregate spatial RNA data into low resolution

    Args:
        adata (anndata.AnnData): spatial RNA data.
        low_res (float): low resolution.
    
    Returns:
        anndata.AnnData: spatial RNA data in low resolution.
    """
    # retreive data and parameters
    coords = adata.obsm['X_spatial']
    v = adata.obsm['true_spatial_velocity']
    u = adata.layers['unspliced']
    s = adata.layers['spliced']
    if isinstance(u, sp.sparse.csr_matrix):
        u = u.toarray()
    if isinstance(s, sp.sparse.csr_matrix):
        s = s.toarray()
    t = adata.obs['true_time'].to_numpy()
    rho = adata.layers['true_rho']
    
    # Perform aggregation
    coords, group_ind = low_res_partition(coords, low_res)
    v = np.array(list(map(lambda ind: v[ind].mean(axis=0), group_ind)))
    u = np.array(list(map(lambda ind: u[ind].sum(axis=0), group_ind)))
    s = np.array(list(map(lambda ind: s[ind].sum(axis=0), group_ind)))
    t = np.array(list(map(lambda ind: t[ind].mean(), group_ind)))
    rho = np.array(list(map(lambda ind: rho[ind].mean(axis=0), group_ind)))
    
    obs = {
        'true_time': t
    }
    
    var = adata.var
    
    layers = {
        'unspliced': sp.sparse.csr_matrix(u),
        'spliced': sp.sparse.csr_matrix(s),
        'true_mean_rho': rho
    }
    
    adata = anndata.AnnData(s, obs, var, layers=layers)
    adata.obsm['true_spatial_velocity'] = v
    adata.obsm['X_spatial'] = coords
    
    return adata