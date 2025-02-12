from typing import List, Literal, Optional, Tuple
import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal
from sklearn.neighbors import NearestNeighbors
from .utils import cart2polar, polar2cart, rotate, rotate_cov
from ..model.model_util import pred_su_numpy


def rho_spatial_gaussian(coords: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Calculates spatially variable transcription rate based on a multivariate normal distribution.

    Args:
        coords (np.ndarray): spatial coordinates.
        mu (np.ndarray): mean of the multivariate normal distribution.
        cov (np.ndarray): covariance matrix of the multivariate normal distribution.

    Returns:
        np.ndarray: transcription rate.
    """
    mn = multivariate_normal(mu, cov)
    return mn.pdf(coords) / mn.pdf(mu)


def rho_spatial_bimodal(coords: np.ndarray,
                        p: float,
                        mu_1: np.ndarray,
                        mu_2: np.ndarray,
                        cov_1: np.ndarray,
                        cov_2: np.ndarray) -> np.ndarray:
    """Calculates spatially variable transcription rate based on a bimodal distribution.

    Args:
        coords (np.ndarray): spatial coordinates.
        p (float): probability of being assigned to the first mixture component.
        mu_1 (np.ndarray): mean of the first mixture component.
        mu_2 (np.ndarray): mean of the second mixture component.
        cov_1 (np.ndarray): covariance matrix of the first mixture component.
        cov_2 (np.ndarray): covariance matrix of the second mixture component.

    Returns:
        np.ndarray: transcription rate.
    """
    mn1 = multivariate_normal(mu_1, cov_1)
    mn2 = multivariate_normal(mu_2, cov_2)
    pmax = max(p*mn1.pdf(mu_1)+(1-p)*mn2.pdf(mu_1), p*mn1.pdf(mu_2)+(1-p)*mn2.pdf(mu_2))
    return (p*mn1.pdf(coords)+(1-p)*mn2.pdf(coords))/pmax


def sample_rates(mu: np.ndarray, cov: np.ndarray, n_sample: int, seed: int = 42) -> np.ndarray:
    """Samples transcription, splicing and degradation rates based on a log normal distribution.

    Args:
        mu (np.ndarray): transcription, splicing and degradation rates. Should have a shape of (3,).
        cov (np.ndarray): covariance matrix. Should have a shape of (3, 3).
        n_sample (int): number of samples.
        seed (int, optional): random seed. Defaults to 42.

    Returns:
        np.ndarray: sampled rates.
    """
    np.random.seed(seed)
    return np.exp(np.random.multivariate_normal(mu, cov, size=n_sample))


def simulate_rna(t: np.ndarray,
                 coords: np.ndarray,
                 n_gene: int,
                 n_repress_gene: int,
                 mu_param: np.ndarray,
                 cov_param: np.ndarray,
                 t_max: float,
                 sparsity: Tuple[float, float],
                 angle: float,
                 noise_level: float = 0.1,
                 noise_type: Literal['Gaussian', 'Spatial Gaussian'] = 'Gaussian',
                 n_neighbors: int = 16,
                 ton_cutoff: List[float] = [0.0, 0.2],
                 toff_cutoff: List[float] = [0.3, 1.0],
                 rho_spatial_pattern: Literal['Gaussian', 'Polar Gaussian', 'Gaussian (Bidirection)'] = 'Gaussian',
                 centers: Optional[List[np.ndarray]] = None,
                 seed=42) -> Tuple:
    """Simulate RNA expression data using RNA velocity with spatially variable transcription rates.

    Args:
        t (np.ndarray): time points.
        coords (np.ndarray): spatial coordinates.
        n_gene (int): number of genes.
        n_repress_gene (int): number of repressive genes.
        mu_param (np.ndarray): mean of the log normal distribution for transcription, splicing and degradation rates.
        cov_param (np.ndarray): covariance matrix of the log normal distribution for transcription,
            splicing and degradation rates.
        t_max (float): maximum time.
        sparsity (Tuple[float, float]): sparsity of unspliced and spliced mRNA data.
        angle (float): rotation angle in radian.
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

    Returns:
        Tuple: simulation results
    """
    np.random.seed(seed)
    # sanity check
    if coords[:, 0].min() == coords[:, 0].max() or coords[:, 1].min() == coords[:, 1].max():
        raise ValueError("The spatial coordinates are not valid.")

    rates = sample_rates(mu_param, cov_param, n_gene, seed)
    # Sample switch-on and switch-off time
    ton = np.random.uniform(t_max*ton_cutoff[0], t_max*ton_cutoff[1], size=(n_gene))
    toff = np.random.uniform(t_max*toff_cutoff[0], t_max*toff_cutoff[1], size=(n_gene))        
    
    # spatially correlated rho
    if rho_spatial_pattern == 'Polar Gaussian':
        if centers is None:
            coords_polar = cart2polar(coords)
            peak_pos_polar = np.stack([np.random.uniform(0, coords_polar[:, 0].max(), size=(n_gene)),
                                       np.random.uniform(coords_polar[:, 1].min(), coords_polar[:, 1].max(), size=(n_gene))], 1)
            sigma_r = np.std(coords_polar[:, 0]) * np.exp(np.random.normal(1, 0.5, size=(n_gene)))
            sigma_theta = np.exp(np.random.normal(1.0, 1.0, size=(n_gene)))
            delta_theta = np.stack([np.arctan2(np.sin(coords_polar[:, 1] - peak_pos_polar[i][1]),
                                               np.cos(coords_polar[:, 1] - peak_pos_polar[i][1])) for i in range(n_gene)], 1)
            rho = np.stack([rho_spatial_gaussian(np.stack([coords_polar[:, 0], 1-np.cos(delta_theta[:, i])], 1),
                                                 np.array([peak_pos_polar[i][0], 0]),
                                                 np.diag([sigma_r[i]**2, sigma_theta[i]**2]))\
                            for i in range(n_gene)], 1)
        else:  # multiple disks
            n_part = len(centers)
            angle_to_next = 2*np.pi/n_part
            # Polar coordinats relative to the center of the first disk
            coords_polar = cart2polar(coords[0] - centers[0])
            peak_pos_polar = np.stack([np.random.uniform(0, coords_polar[:, 0].max(), size=(n_gene)),
                                       np.random.uniform(coords_polar[:, 1].min(), coords_polar[:, 1].max(), size=(n_gene))], 1)
            
            sigma_r = np.std(coords_polar[:, 0]) * np.exp(np.random.normal(1, 0.5, size=(n_gene)))
            sigma_theta = np.exp(np.random.normal(1.0, 1.0, size=(n_gene)))

            # absolute x-y coordinates
            peak_pos = polar2cart(peak_pos_polar) + centers[0]
            rho = []
            
            for i in range(n_part):  # iterate through all disks
                # relative to the center of the i-th disk
                peak_pos_polar_ = cart2polar(rotate(peak_pos, i*angle_to_next) - centers[i])
                coords_polar_ = cart2polar(coords[i] - centers[i])
                # cell by gene matrix
                delta_theta = np.stack([np.arctan2(np.sin(coords_polar_[:, 1] - peak_pos_polar_[i][1]),
                                                   np.cos(coords_polar_[:, 1] - peak_pos_polar_[i][1])) for i in range(n_gene)], 1)
                rho_ = np.stack([rho_spatial_gaussian(np.stack([coords_polar_[:, 0], 1-np.cos(delta_theta[:, i])], 1),
                                                      np.array([peak_pos_polar_[i][0], 0]),
                                                      np.diag([sigma_r[i]**2, sigma_theta[i]**2]))\
                                 for i in range(n_gene)], 1)
                rho.append(rho_)
            t = np.concatenate(t)
            coords = np.concatenate(coords, 0)
            rho = np.concatenate(rho, 0)
    elif rho_spatial_pattern == 'Gaussian (Bidirection)':
        peak_pos = rotate(np.stack([np.random.uniform(coords[:, 0].min(), coords[:, 0].max(), size=(n_gene)),
                                    np.random.uniform(coords[:, 1].min(), coords[:, 1].max(), size=(n_gene))], 1),
                          angle)
        sigma_x = np.std(coords[:, 0]) * np.exp(np.random.normal(1, 0.5, size=(n_gene)))
        sigma_y = np.std(coords[:, 1]) * np.exp(np.random.normal(1, 0.5, size=(n_gene)))
        rho = np.stack([rho_spatial_bimodal(coords,
                                            0.5,
                                            peak_pos[i],
                                            -peak_pos[i],
                                            rotate_cov(np.diag([sigma_x[i]**2, sigma_y[i]**2]), angle),
                                            rotate_cov(np.diag([sigma_x[i]**2, sigma_y[i]**2]), angle))\
                        for i in range(n_gene)], 1)
    else:  # Gaussian
        peak_pos = rotate(np.stack([np.random.uniform(coords[:, 0].min(), coords[:, 0].max(), size=(n_gene)),
                                    np.random.uniform(coords[:, 1].min(), coords[:, 1].max(), size=(n_gene))], 1),
                          angle)
        sigma_x = np.std(coords[:, 0]) * np.exp(np.random.normal(1, 0.5, size=(n_gene)))
        sigma_y = np.std(coords[:, 1]) * np.exp(np.random.normal(1, 0.5, size=(n_gene)))
        rho = np.stack([rho_spatial_gaussian(coords,
                                             peak_pos[i],
                                             rotate_cov(np.diag([sigma_x[i]**2, sigma_y[i]**2]), angle))\
                        for i in range(n_gene)], 1)

    n_cell = len(t)
    t = t.reshape(-1, 1)
    
    # Randomly select a number of repressive genes
    if n_repress_gene > 0:
        perm = np.random.permutation(n_gene)
        repress_idx, ind_idx = perm[:n_repress_gene], perm[n_repress_gene:]
        is_repressive = np.zeros((n_gene)).astype(bool)
        is_repressive[repress_idx] = 1
    
        u_ind_1, s_ind_1 = pred_su_numpy(
            np.clip(t - ton[ind_idx], 0, None),
            0,
            0,
            rates[ind_idx, 0] * rho[:, ind_idx],
            rates[ind_idx, 1],
            rates[ind_idx, 2]
        )
        u0_, s0_ = pred_su_numpy(
            toff[ind_idx] - ton[ind_idx],
            0,
            0,
            rates[ind_idx, 0] * rho[:, ind_idx],
            rates[ind_idx, 1],
            rates[ind_idx, 2]
        )
        u_ind_2, s_ind_2 = pred_su_numpy(
            t - toff[ind_idx],
            u0_,
            s0_,
            0,
            rates[ind_idx, 1],
            rates[ind_idx, 2])
        u, s = np.empty((n_cell, n_gene)), np.empty((n_cell, n_gene))
        u_repress, s_repress = pred_su_numpy(
            t,
            rates[repress_idx, 0] * rho[:, repress_idx] / rates[repress_idx, 1],
            rates[repress_idx, 0] * rho[:, repress_idx] / rates[repress_idx, 2],
            0,
            rates[repress_idx, 1],
            rates[repress_idx, 2]
        )
        u[:, repress_idx] = u_repress
        s[:, repress_idx] = s_repress
        mask = (t < toff[ind_idx])
        u[:, ind_idx] = u_ind_1 * mask + u_ind_2 * (1 - mask)
        s[:, ind_idx] = s_ind_1 * mask + s_ind_2 * (1 - mask)
    else:
        u_ind_1, s_ind_1 = pred_su_numpy(
            np.clip(t - ton, 0, None) ,
            0,
            0,
            rates[:, 0] * rho,
            rates[:, 1],
            rates[:, 2]
        )
        u0_, s0_ = pred_su_numpy(
            toff - ton,
            0,
            0,
            rates[:, 0] * rho,
            rates[:, 1],
            rates[:, 2]
        )
        u_ind_2, s_ind_2 = pred_su_numpy(
            t - toff,
            u0_,
            s0_,
            0,
            rates[:, 1],
            rates[:, 2]
        )
        mask = (t < toff)
        u = u_ind_1 * mask + u_ind_2 * (1 - mask)
        s = s_ind_1 * mask + s_ind_2 * (1 - mask)
        is_repressive = None
    # Add noise
    if noise_type == "Spatial Gaussian":
        eps_u = np.random.normal(0.0, noise_level, size=u.shape)
        eps_s = np.random.normal(0.0, noise_level, size=u.shape)
        knn = NearestNeighbors(n_neighbors=n_neighbors)
        dist, ind = knn.fit(coords).kneighbors()
        sigma = np.median(dist)
        coeff = np.exp(-(dist/sigma)**2)  # n_cell x k
        coeff = coeff / coeff.sum(1).reshape(-1, 1)
        noise_u = np.stack([np.sum(coeff[i].reshape(-1, 1)*eps_u[ind[i]], 0) for i in range(n_cell)])
        noise_s = np.stack([np.sum(coeff[i].reshape(-1, 1)*eps_s[ind[i]], 0) for i in range(n_cell)])
    elif noise_type == "Gaussian":
        noise_u = np.random.normal(0.0, np.clip(noise_level*u, 0.01, None), size=u.shape)
        noise_s = np.random.normal(0.0, np.clip(noise_level*s, 0.01, None), size=u.shape)
    else:
        raise NotImplementedError(f"noise_type {noise_type} is not implemented.")
    mask_u = sp.stats.bernoulli.rvs(1-sparsity[0], size=u.shape)
    mask_s = sp.stats.bernoulli.rvs(1-sparsity[1], size=s.shape)
    u = np.clip(u+noise_u, 0, None)*mask_u
    s = np.clip(s+noise_s, 0, None)*mask_s
    return u, s, rho, rates, ton, toff, is_repressive