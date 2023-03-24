import numpy as np
from scipy.stats import spearmanr, poisson, norm
from scipy.special import loggamma
from sklearn.metrics.pairwise import pairwise_distances
from ..model.model_util import pred_su_numpy, ode_numpy, ode_br_numpy, scv_pred, scv_pred_single
import hnswlib
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn.functional as F


def get_mse(U, S, Uhat, Shat):
    return np.nanmean((U-Uhat)**2+(S-Shat)**2)


def get_mae(U, S, Uhat, Shat):
    return np.nanmean(np.abs(U-Uhat)+np.abs(S-Shat))


def time_corr(t1, t2):
    return spearmanr(t1, t2)


def poisson_log_likelihood(mu, obs):
    return -mu+obs*np.log(mu)-loggamma(obs)


def cell_state(adata, method, key, gene_indices=None, **kwargs):
    cell_state = None
    if gene_indices is None:
        gene_indices = np.array(np.range(adata.n_vars))
    if method == 'scVelo':
        t = adata.layers[f"{key}_t"][:, gene_indices]
        toff = adata.var[f"{key}_t_"].to_numpy()[gene_indices]
        cell_state = (t > toff)
    elif method == 'Vanilla VAE':
        t = adata.obs[f"{key}_time"].to_numpy()
        toff = adata.var[f"{key}_toff"].to_numpy()
        ton = adata.var[f"{key}_ton"].to_numpy()
        toff = toff[gene_indices]
        ton = ton[gene_indices]
        cell_state = (t.reshape(-1, 1) > toff) + (t.reshape(-1, 1) < ton)*2
    elif method in ['VeloVAE', 'FullVB', 'Discrete VeloVAE', 'Discrete FullVB']:
        rho = adata.layers[f"{key}_rho"][:, gene_indices]
        t = adata.obs[f"{key}_time"].to_numpy()
        mask_induction = rho > 0.01
        mask_repression = np.empty(mask_induction.shape, dtype=bool)
        mask_off = np.empty(mask_induction.shape, dtype=bool)
        for i in range(len(gene_indices)):
            if np.any(mask_induction[:, i]):
                ton = np.quantile(t[mask_induction[:, i]], 1e-3)
                mask_repression[:, i] = (rho[:, i] <= 0.1) & (t >= ton)
                mask_off[:, i] = (rho[:, i] <= 0.1) & (t < ton)
            else:
                mask_repression[:, i] = True
                mask_off[:, i] = False
        cell_state = mask_repression * 1 + mask_off * 2
    elif method == 'VeloVI':
        t = adata.layers["fit_t"][:, gene_indices]
        toff = adata.var[f"{key}_t_"].to_numpy()[gene_indices]
        cell_state = (t > toff)
    else:
        cell_state = np.array([3 for i in range(adata.n_obs)])
    return cell_state


# scVelo


def get_err_scv(adata, key='fit'):
    Uhat, Shat = scv_pred(adata, key)
    mse = get_mse(adata.layers['Mu'], adata.layers['Ms'], Uhat, Shat)
    mae = get_mae(adata.layers['Mu'], adata.layers['Ms'], Uhat, Shat)
    logp = np.sum(np.log(adata.var[f"{key}_likelihood"]))
    try:
        run_time = adata.uns[f'{key}_run_time']
    except KeyError:
        run_time = np.nan
    return mse, None, mae, None, logp, None, run_time


def get_pred_scv_demo(adata, key='fit', genes=None, N=100):
    if genes is None:
        genes = adata.var_names
    alpha = adata.var[f"{key}_alpha"].to_numpy()
    beta = adata.var[f"{key}_beta"].to_numpy()
    gamma = adata.var[f"{key}_gamma"].to_numpy()
    toff = adata.var[f"{key}_t_"].to_numpy()
    T = adata.layers[f"{key}_t"]
    scaling = adata.var[f"{key}_scaling"].to_numpy()
    Uhat, Shat = np.zeros((2*N, len(genes))), np.zeros((2*N, len(genes)))
    T_demo = np.zeros((2*N, len(genes)))
    for i, gene in enumerate(genes):
        idx = np.where(adata.var_names == gene)[0][0]
        t_1 = np.linspace(0, toff[idx], N)
        t_2 = np.linspace(toff[idx], max(T[:, idx].max(), toff[i]+T[:, idx].max()*0.01), N)
        t_demo = np.concatenate((t_1, t_2))
        T_demo[:, i] = t_demo
        uhat, shat = scv_pred_single(t_demo,
                                     alpha[idx],
                                     beta[idx],
                                     gamma[idx],
                                     toff[idx],
                                     scaling=scaling[idx],
                                     uinit=0,
                                     sinit=0)
        Uhat[:, i] = uhat
        Shat[:, i] = shat
    return T_demo, Uhat, Shat


# Vanilla VAE


def get_pred_vanilla(adata, key):
    # Vanilla VAE
    alpha = adata.var[f"{key}_alpha"].to_numpy()
    beta = adata.var[f"{key}_beta"].to_numpy()
    gamma = adata.var[f"{key}_gamma"].to_numpy()
    toff = adata.var[f"{key}_toff"].to_numpy()
    ton = adata.var[f"{key}_ton"].to_numpy()
    t = adata.obs[f"{key}_time"].to_numpy()
    scaling = adata.var[f"{key}_scaling"].to_numpy()

    if (f"{key}_uhat" not in adata.layers) or (f"{key}_shat" not in adata.layers):
        Uhat, Shat = ode_numpy(t.reshape(-1, 1), alpha, beta, gamma, ton, toff, scaling)
    else:
        Uhat, Shat = adata.layers[f"{key}_uhat"], adata.layers[f"{key}_shat"]

    return Uhat, Shat


def get_err_vanilla(adata, key, gene_mask=None):
    U, S = adata.layers["Mu"], adata.layers["Ms"]
    Uhat, Shat = get_pred_vanilla(adata, key)
    train_idx, test_idx = adata.uns[f"{key}_train_idx"], adata.uns[f"{key}_test_idx"]

    if gene_mask is None:
        gene_mask = np.ones((adata.n_vars)).astype(bool)

    sigma_u = adata.var[f"{key}_sigma_u"].to_numpy()[gene_mask]
    sigma_s = adata.var[f"{key}_sigma_s"].to_numpy()[gene_mask]
    dist_u_train = np.abs(U[train_idx][:, gene_mask]-Uhat[train_idx][:, gene_mask])
    dist_s_train = np.abs(S[train_idx][:, gene_mask]-Shat[train_idx][:, gene_mask])
    dist_u_test = np.abs(U[test_idx][:, gene_mask]-Uhat[test_idx][:, gene_mask])
    dist_s_test = np.abs(S[test_idx][:, gene_mask]-Shat[test_idx][:, gene_mask])

    logp_train = -dist_u_train**2/(2*sigma_u**2) \
        - dist_s_train**2/(2*sigma_s**2) \
        - np.log(sigma_u) - np.log(sigma_s) - np.log(2*np.pi)
    logp_test = -dist_u_test**2/(2*sigma_u**2) \
        - dist_s_test**2/(2*sigma_s**2) \
        - np.log(sigma_u) - np.log(sigma_s) - np.log(2*np.pi)
    mse_train = np.nanmean(dist_u_train**2+dist_s_train**2)
    mse_test = np.nanmean(dist_u_test**2+dist_s_test**2)
    mae_train = np.nanmean(dist_u_train+dist_s_train)
    mae_test = np.nanmean(dist_u_test+dist_s_test)

    logp_train = np.nanmean(np.sum(logp_train, 1))
    logp_test = np.nanmean(np.sum(logp_test, 1))

    try:
        run_time = adata.uns[f'{key}_run_time']
    except KeyError:
        run_time = np.nan
    return mse_train, mse_test, mae_train, mae_test, logp_train, logp_test, run_time


def get_pred_vanilla_demo(adata, key, genes=None, N=100):
    alpha = adata.var[f"{key}_alpha"].to_numpy()
    beta = adata.var[f"{key}_beta"].to_numpy()
    gamma = adata.var[f"{key}_gamma"].to_numpy()
    toff = adata.var[f"{key}_toff"].to_numpy()
    ton = adata.var[f"{key}_ton"].to_numpy()
    scaling = adata.var[f"{key}_scaling"].to_numpy()
    t = adata.obs[f"{key}_time"].to_numpy()

    t_demo = np.linspace(0, t.max(), N)
    if genes is None:
        Uhat_demo, Shat_demo = ode_numpy(t_demo.reshape(-1, 1), alpha, beta, gamma, ton, toff, scaling)
    else:
        gene_indices = np.array([np.where(adata.var_names == x)[0][0] for x in genes])
        Uhat_demo, Shat_demo = ode_numpy(t_demo.reshape(-1, 1),
                                         alpha[gene_indices],
                                         beta[gene_indices],
                                         gamma[gene_indices],
                                         ton[gene_indices],
                                         toff[gene_indices],
                                         scaling[gene_indices])

    return t_demo, Uhat_demo, Shat_demo


# Cycle VAE


def get_pred_cycle(adata, key):
    alpha = adata.var[f"{key}_alpha"].to_numpy()
    beta = adata.var[f"{key}_beta"].to_numpy()
    gamma = adata.var[f"{key}_gamma"].to_numpy()
    scaling = adata.var[f"{key}_scaling"].to_numpy()

    if (f"{key}_uhat" not in adata.layers) or (f"{key}_shat" not in adata.layers):
        t = adata.obs[f"{key}_time"].to_numpy()
        ton = adata.var[f"{key}_ton"].to_numpy()
        toff = adata.var[f"{key}_toff"].to_numpy()
        Uhat, Shat = ode_numpy(t.reshape(-1, 1), alpha, beta, gamma, ton, toff, scaling)
    else:
        Uhat, Shat = adata.layers[f"{key}_uhat"], adata.layers[f"{key}_shat"]

    return Uhat, Shat


def get_err_cycle(adata, key, gene_mask=None):
    U, S = adata.layers["Mu"], adata.layers["Ms"]
    Uhat, Shat = get_pred_cycle(adata, key)
    train_idx, test_idx = adata.uns[f"{key}_train_idx"], adata.uns[f"{key}_test_idx"]

    if gene_mask is None:
        gene_mask = np.ones((adata.n_vars)).astype(bool)

    sigma_u = adata.var[f"{key}_sigma_u"].to_numpy()[gene_mask]
    sigma_s = adata.var[f"{key}_sigma_s"].to_numpy()[gene_mask]
    dist_u_train = np.abs(U[train_idx][:, gene_mask]-Uhat[train_idx][:, gene_mask])
    dist_s_train = np.abs(S[train_idx][:, gene_mask]-Shat[train_idx][:, gene_mask])
    dist_u_test = np.abs(U[test_idx][:, gene_mask]-Uhat[test_idx][:, gene_mask])
    dist_s_test = np.abs(S[test_idx][:, gene_mask]-Shat[test_idx][:, gene_mask])

    logp_train = -dist_u_train**2/(2*sigma_u**2) \
        - dist_s_train**2/(2*sigma_s**2) \
        - np.log(sigma_u) - np.log(sigma_s) - np.log(2*np.pi)
    logp_test = -dist_u_test**2/(2*sigma_u**2) \
        - dist_s_test**2/(2*sigma_s**2) \
        - np.log(sigma_u) - np.log(sigma_s) - np.log(2*np.pi)
    mse_train = np.nanmean(dist_u_train**2+dist_s_train**2)
    mse_test = np.nanmean(dist_u_test**2+dist_s_test**2)
    mae_train = np.nanmean(dist_u_train+dist_s_train)
    mae_test = np.nanmean(dist_u_test+dist_s_test)

    logp_train = np.nanmean(np.sum(logp_train, 1))
    logp_test = np.nanmean(np.sum(logp_test, 1))

    try:
        run_time = adata.uns[f'{key}_run_time']
    except KeyError:
        run_time = np.nan
    return mse_train, mse_test, mae_train, mae_test, logp_train, logp_test, run_time


def get_pred_cycle_demo(adata, key, genes=None, N=100):
    alpha = adata.var[f"{key}_alpha"].to_numpy()
    beta = adata.var[f"{key}_beta"].to_numpy()
    gamma = adata.var[f"{key}_gamma"].to_numpy()
    scaling = adata.var[f"{key}_scaling"].to_numpy()
    t = adata.obs[f"{key}_time"].to_numpy()
    ton = adata.var[f"{key}_ton"].to_numpy()
    toff = adata.var[f"{key}_toff"].to_numpy()

    t_demo = np.linspace(t.min(), t.max(), N)

    if genes is None:
        Uhat_demo, Shat_demo = ode_numpy(t_demo.reshape(-1, 1), alpha, beta, gamma, ton, toff, scaling)
    else:
        gene_indices = np.array([np.where(adata.var_names == x)[0][0] for x in genes])
        Uhat_demo, Shat_demo = ode_numpy(t_demo.reshape(-1, 1),
                                         alpha[gene_indices],
                                         beta[gene_indices],
                                         gamma[gene_indices],
                                         ton[gene_indices],
                                         toff[gene_indices],
                                         scaling[gene_indices])

    return t_demo, Uhat_demo, Shat_demo


# VeloVAE
def get_pred_velovae(adata, key, scv_key=None, full_vb=False, discrete=False):
    if (f"{key}_uhat" not in adata.layers) or (f"{key}_shat" not in adata.layers):
        rho = adata.layers[f"{key}_rho"]
        alpha = (adata.var[f"{key}_alpha"].to_numpy() if not full_vb else
                 np.exp(adata.var[f"{key}_logmu_alpha"].to_numpy()))
        beta = (adata.var[f"{key}_beta"].to_numpy() if not full_vb else
                np.exp(adata.var[f"{key}_logmu_beta"].to_numpy()))
        gamma = (adata.var[f"{key}_gamma"].to_numpy() if not full_vb else
                 np.exp(adata.var[f"{key}_logmu_gamma"].to_numpy()))
        t = adata.obs[f"{key}_time"].to_numpy()
        scaling = adata.var[f"{key}_scaling"].to_numpy()

        u0, s0 = adata.layers[f"{key}_u0"], adata.layers[f"{key}_s0"]
        t0 = adata.obs[f"{key}_t0"].to_numpy()

        Uhat, Shat = pred_su_numpy((t-t0).reshape(-1, 1), u0, s0, rho*alpha, beta, gamma)
        Uhat = Uhat*scaling
    else:
        Uhat, Shat = adata.layers[f"{key}_uhat"], adata.layers[f"{key}_shat"]

    return Uhat, Shat


def get_err_velovae(adata, key, gene_mask=None, full_vb=False, discrete=False, n_sample=30, seed=2022):
    Uhat, Shat = get_pred_velovae(adata, key, full_vb, discrete)
    train_idx, test_idx = adata.uns[f"{key}_train_idx"], adata.uns[f"{key}_test_idx"]
    if gene_mask is None:
        gene_mask = np.ones((adata.n_vars)).astype(bool)

    if discrete:
        U, S = adata.layers["unspliced"].A, adata.layers["spliced"].A
        lu, ls = adata.obs["library_scale_u"].to_numpy(), adata.obs["library_scale_s"].to_numpy()
        Uhat = Uhat*(lu.reshape(-1, 1))
        Shat = Shat*(ls.reshape(-1, 1))
        Uhat = np.clip(Uhat, a_min=1e-2, a_max=None)
        Shat = np.clip(Shat, a_min=1e-2, a_max=None)

        logp_train = np.log(poisson.pmf(U[train_idx], Uhat[train_idx])+1e-10) \
            + np.log(poisson.pmf(S[train_idx], Shat[train_idx])+1e-10)
        logp_test = np.log(poisson.pmf(U[test_idx], Uhat[test_idx])+1e-10) \
            + np.log(poisson.pmf(S[test_idx], Shat[test_idx])+1e-10)

        # Sample multiple times
        mse_train, mae_train, mse_test, mae_test = 0, 0, 0, 0
        np.random.seed(seed)

        for i in range(n_sample):
            U_sample = poisson.rvs(Uhat)
            S_sample = poisson.rvs(Shat)
            dist_u_train = np.abs(U[train_idx][:, gene_mask]-U_sample[train_idx][:, gene_mask])
            dist_s_train = np.abs(S[train_idx][:, gene_mask]-S_sample[train_idx][:, gene_mask])
            dist_u_test = np.abs(U[test_idx][:, gene_mask]-U_sample[test_idx][:, gene_mask])
            dist_s_test = np.abs(S[test_idx][:, gene_mask]-S_sample[test_idx][:, gene_mask])
            mse_train += np.nanmean(dist_u_train**2+dist_s_train**2)
            mse_test += np.nanmean(dist_u_test**2+dist_s_test**2)
            mae_train += np.nanmean(dist_u_train+dist_s_train)
            mae_test += np.nanmean(dist_u_test+dist_s_test)
        mse_train /= n_sample
        mse_test /= n_sample
        mae_train /= n_sample
        mae_test /= n_sample
    else:
        U, S = adata.layers["Mu"], adata.layers["Ms"]
        sigma_u = adata.var[f"{key}_sigma_u"].to_numpy()[gene_mask]
        sigma_s = adata.var[f"{key}_sigma_s"].to_numpy()[gene_mask]

        dist_u_train = np.abs(U[train_idx][:, gene_mask]-Uhat[train_idx][:, gene_mask])
        dist_s_train = np.abs(S[train_idx][:, gene_mask]-Shat[train_idx][:, gene_mask])
        dist_u_test = np.abs(U[test_idx][:, gene_mask]-Uhat[test_idx][:, gene_mask])
        dist_s_test = np.abs(S[test_idx][:, gene_mask]-Shat[test_idx][:, gene_mask])

        logp_train = -dist_u_train**2/(2*sigma_u**2) \
            - dist_s_train**2/(2*sigma_s**2) \
            - np.log(sigma_u) - np.log(sigma_s) - np.log(2*np.pi)
        logp_test = -dist_u_test**2/(2*sigma_u**2) \
            - dist_s_test**2/(2*sigma_s**2) \
            - np.log(sigma_u) - np.log(sigma_s) - np.log(2*np.pi)

        mse_train = np.nanmean(dist_u_train**2+dist_s_train**2)
        mse_test = np.nanmean(dist_u_test**2+dist_s_test**2)
        mae_train = np.nanmean(dist_u_train+dist_s_train)
        mae_test = np.nanmean(dist_u_test+dist_s_test)

    logp_train = np.nanmean(np.sum(logp_train, 1))
    logp_test = np.nanmean(np.sum(logp_test, 1))

    try:
        run_time = adata.uns[f'{key}_run_time']
    except KeyError:
        run_time = np.nan

    return mse_train, mse_test, mae_train, mae_test, logp_train, logp_test, run_time


def get_pred_velovae_demo(adata, key, genes=None, full_vb=False, discrete=False):
    if (f"{key}_uhat" not in adata.layers) or (f"{key}_shat" not in adata.layers):
        alpha = (adata.var[f"{key}_alpha"].to_numpy() if not full_vb else
                 np.exp(adata.var[f"{key}_logmu_alpha"].to_numpy()))
        beta = (adata.var[f"{key}_beta"].to_numpy() if not full_vb else
                np.exp(adata.var[f"{key}_logmu_beta"].to_numpy()))
        gamma = (adata.var[f"{key}_gamma"].to_numpy() if not full_vb else
                 np.exp(adata.var[f"{key}_logmu_gamma"].to_numpy()))
        t = adata.obs[f"{key}_time"].to_numpy()
        scaling = adata.var[f"{key}_scaling"].to_numpy()
        u0, s0 = adata.layers[f"{key}_u0"], adata.layers[f"{key}_s0"]
        t0 = adata.obs[f"{key}_t0"].to_numpy()
        if genes is None:
            rho = adata.layers[f"{key}_rho"]
            Uhat, Shat = pred_su_numpy((t-t0).reshape(-1, 1), u0, s0, alpha, beta, gamma)
            Uhat = Uhat*scaling
        else:
            gene_indices = np.array([np.where(adata.var_names == x)[0][0] for x in genes])
            rho = adata.layers[f"{key}_rho"][:, gene_indices]
            Uhat, Shat = pred_su_numpy((t-t0).reshape(-1, 1),
                                       u0[:, gene_indices],
                                       s0[:, gene_indices],
                                       rho*alpha[gene_indices],
                                       beta[gene_indices],
                                       gamma[gene_indices])
            Uhat = Uhat*scaling[gene_indices]
    else:
        if genes is None:
            Uhat, Shat = adata.layers[f"{key}_uhat"], adata.layers[f"{key}_shat"]
        else:
            gene_indices = np.array([np.where(adata.var_names == x)[0][0] for x in genes])
            Uhat, Shat = adata.layers[f"{key}_uhat"][:, gene_indices], adata.layers[f"{key}_shat"][:, gene_indices]
    if discrete:
        lu = adata.obs["library_scale_u"].to_numpy().reshape(-1, 1)
        ls = adata.obs["library_scale_s"].to_numpy().reshape(-1, 1)
        Uhat = Uhat * lu
        Shat = Shat * ls
    return Uhat, Shat


# Branching ODE
def get_pred_brode(adata, key):
    if (f"{key}_uhat" not in adata.layers) or (f"{key}_shat" not in adata.layers):
        alpha = adata.varm[f"{key}_alpha"]
        beta = adata.varm[f"{key}_beta"]
        gamma = adata.varm[f"{key}_gamma"]
        t_trans = adata.uns[f"{key}_t_trans"]
        scaling = adata.var[f"{key}_scaling"].to_numpy()
        par = np.argmax(adata.uns[f"{key}_w"], 1)

        t = adata.obs[f"{key}_time"].to_numpy()
        y = adata.obs[f"{key}_label"]

        Uhat, Shat = ode_br_numpy(t.reshape(-1, 1),
                                  y,
                                  par,
                                  alpha=alpha,
                                  beta=beta,
                                  gamma=gamma,
                                  t_trans=t_trans,
                                  scaling=scaling)
        Uhat = Uhat*scaling
    else:
        Uhat, Shat = adata.layers[f"{key}_uhat"], adata.layers[f"{key}_shat"]

    return Uhat, Shat


def get_err_brode(adata, key, gene_mask=None):
    U, S = adata.layers["Mu"], adata.layers["Ms"]
    Uhat, Shat = get_pred_brode(adata, key)
    train_idx, test_idx = adata.uns[f"{key}_train_idx"], adata.uns[f"{key}_test_idx"]

    if gene_mask is None:
        gene_mask = np.ones((adata.n_vars)).astype(bool)
    sigma_u = adata.var[f"{key}_sigma_u"].to_numpy()[gene_mask]
    sigma_s = adata.var[f"{key}_sigma_s"].to_numpy()[gene_mask]
    dist_u_train = np.abs(U[train_idx][:, gene_mask]-Uhat[train_idx][:, gene_mask])
    dist_s_train = np.abs(S[train_idx][:, gene_mask]-Shat[train_idx][:, gene_mask])
    dist_u_test = np.abs(U[test_idx][:, gene_mask]-Uhat[test_idx][:, gene_mask])
    dist_s_test = np.abs(S[test_idx][:, gene_mask]-Shat[test_idx][:, gene_mask])

    logp_train = - dist_u_train**2/(2*sigma_u**2)\
        - dist_s_train**2/(2*sigma_s**2)\
        - np.log(sigma_u) - np.log(sigma_s) - np.log(2*np.pi)
    logp_test = - dist_u_test**2/(2*sigma_u**2)\
        - dist_s_test**2/(2*sigma_s**2)\
        - np.log(sigma_u) - np.log(sigma_s) - np.log(2*np.pi)
    mse_train = np.nanmean(dist_u_train**2+dist_s_train**2)
    mae_train = np.nanmean(dist_u_train+dist_s_train)
    mse_test = np.nanmean(dist_u_test**2+dist_s_test**2)
    mae_test = np.nanmean(dist_u_test+dist_s_test)

    logp_train = np.nanmean(np.sum(logp_train, 1))
    logp_test = np.nanmean(np.sum(logp_test, 1))

    try:
        run_time = adata.uns[f'{key}_run_time']
    except KeyError:
        run_time = np.nan

    return mse_train, mse_test, mae_train, mae_test, logp_train, logp_test, run_time


def get_pred_brode_demo(adata, key, genes=None, N=100):
    t_trans = adata.uns[f"{key}_t_trans"]
    t = adata.obs[f"{key}_time"].to_numpy()
    y = adata.obs[f"{key}_label"].to_numpy()  # integer
    par = np.argmax(adata.uns[f"{key}_w"], 1)
    n_type = len(par)
    t_demo = np.zeros((N*n_type))
    y_demo = np.zeros((N*n_type)).astype(int)
    for i in range(n_type):
        y_demo[i*N:(i+1)*N] = i
        t_demo[i*N:(i+1)*N] = np.linspace(t_trans[i], np.quantile(t[y == i], 0.95), N)
    if genes is None:
        alpha = adata.varm[f"{key}_alpha"].T
        beta = adata.varm[f"{key}_beta"].T
        gamma = adata.varm[f"{key}_gamma"].T
        u0, s0 = adata.varm[f"{key}_u0"].T, adata.varm[f"{key}_s0"].T
        scaling = adata.var[f"{key}_scaling"].to_numpy()
    else:
        gene_indices = np.array([np.where(adata.var_names == x)[0][0] for x in genes])
        alpha = adata.varm[f"{key}_alpha"][gene_indices].T
        beta = adata.varm[f"{key}_beta"][gene_indices].T
        gamma = adata.varm[f"{key}_gamma"][gene_indices].T
        u0, s0 = adata.varm[f"{key}_u0"][gene_indices].T, adata.varm[f"{key}_s0"][gene_indices].T
        scaling = adata.var[f"{key}_scaling"][gene_indices].to_numpy()

    Uhat_demo, Shat_demo = ode_br_numpy(t_demo.reshape(-1, 1),
                                        y_demo,
                                        par,
                                        alpha=alpha,
                                        beta=beta,
                                        gamma=gamma,
                                        t_trans=t_trans,
                                        u0=u0,
                                        s0=s0,
                                        scaling=scaling)

    return t_demo, y_demo, Uhat_demo, Shat_demo


# UniTVelo


def get_pred_utv(adata, B=5000):
    t = adata.layers['fit_t']
    o = adata.var['fit_offset'].values
    a0 = adata.var['fit_a'].values
    t0 = adata.var['fit_t'].values
    h0 = adata.var['fit_h'].values
    i = adata.var['fit_intercept'].values
    gamma = adata.var['fit_gamma'].values
    beta = adata.var['fit_beta'].values

    Nb = adata.n_obs//B
    uhat, shat = np.empty(adata.shape), np.empty(adata.shape)
    for i in range(Nb):
        shat[i*B:(i+1)*B] = h0*np.exp(-a0*(t[i*B:(i+1)*B] - t0)**2) + o
        uhat[i*B:(i+1)*B] = (adata.layers['velocity'][i*B:(i+1)*B]+gamma*shat[i*B:(i+1)*B])/beta + i
    if Nb*B < adata.n_obs:
        shat[Nb*B:] = h0*np.exp(-a0*(t[Nb*B:] - t0)**2) + o
        uhat[Nb*B:] = (adata.layers['velocity'][Nb*B:]+gamma*shat[Nb*B:])/beta + i

    return uhat, shat


def get_err_utv(adata, key, gene_mask=None, B=5000):
    Uhat, Shat = get_pred_utv(adata, B)
    U, S = adata.layers['Mu'], adata.layers['Ms']

    if gene_mask is None:
        gene_mask = np.where(~np.isnan(adata.layers['fit_t'][0]))[0]

    dist_u = np.abs(U[:, gene_mask] - Uhat[:, gene_mask])
    dist_s = np.abs(S[:, gene_mask] - Shat[:, gene_mask])

    # MLE of variance
    var_s = np.var((S[:, gene_mask]-Shat[:, gene_mask]), 0)
    var_u = np.var((U[:, gene_mask]-Uhat[:, gene_mask]), 0)

    mse = np.nanmean(dist_u**2 + dist_s**2)
    mae = np.nanmean(dist_u+dist_s)
    logp = -dist_u**2/(2*var_u)-dist_s**2/(2*var_s) - 0.5*np.log(var_u) - 0.5*np.log(var_s) - np.log(2*np.pi)

    try:
        run_time = adata.uns[f'{key}_run_time']
    except KeyError:
        run_time = np.nan

    return mse, None, mae, None, logp.sum(1).mean(0), None, run_time


def get_pred_utv_demo(adata, genes=None, N=100):
    t = adata.layers['fit_t']

    if genes is None:
        gene_indices = np.array(np.range(adata.n_vars))
    else:
        gene_indices = np.array([np.where(adata.var_names == x)[0][0] for x in genes])

    t_demo = np.linspace(t[:, gene_indices].min(0), t[:, gene_indices].max(0), N)
    o = adata.var['fit_offset'].values[gene_indices]
    a0 = adata.var['fit_a'].values[gene_indices]
    t0 = adata.var['fit_t'].values[gene_indices]
    h0 = adata.var['fit_h'].values[gene_indices]
    i = adata.var['fit_intercept'].values[gene_indices]
    gamma = adata.var['fit_gamma'].values[gene_indices]
    beta = adata.var['fit_beta'].values[gene_indices]

    shat = h0*np.exp(-a0*(t_demo - t0)**2) + o
    vs = shat * (-2*a0*(t_demo - t0))
    uhat = (vs + gamma*shat)/beta + i

    return t_demo, uhat, shat


##########################################################################
# Evaluation utility functions from DeepVelo
# Reference:
# Gao, M., Qiao, C. & Huang, Y. UniTVelo: temporally unified RNA velocity
# reinforces single-cell trajectory inference. Nat Commun 13, 6586 (2022).
# https://doi.org/10.1038/s41467-022-34188-7
##########################################################################

def get_neighbor_idx(adata,
                     topC=30,
                     topG=20):
    Sx_sz = adata.layers["Ms"]
    N_cell, N_gene = Sx_sz.shape

    n_pcas = 30
    pca_ = PCA(
        n_components=n_pcas,
        svd_solver="randomized",
    )
    Sx_sz_pca = pca_.fit_transform(Sx_sz)
    if N_cell < 3000:
        ori_dist = pairwise_distances(Sx_sz_pca, Sx_sz_pca)
        nn_t_idx = np.argsort(ori_dist, axis=1)[:, 1:topC]
    else:
        p = hnswlib.Index(space="l2", dim=n_pcas)
        p.init_index(max_elements=N_cell, ef_construction=200, M=30)
        p.add_items(Sx_sz_pca)
        p.set_ef(max(topC, topG) + 10)
        nn_t_idx = p.knn_query(Sx_sz_pca, k=topC)[0][:, 1:].astype(int)

    return nn_t_idx


def _loss_dv(
    output: torch.Tensor,
    current_state: torch.Tensor,
    idx: torch.LongTensor,
    candidate_states: torch.Tensor,
    n_spliced: int = None,
    l: int = 2,
    *args,
    **kwargs,
):
    batch_size, genes = current_state.shape
    if n_spliced is not None:
        genes = n_spliced
    num_neighbors = idx.shape[1]
    loss = []
    for i in range(num_neighbors):
        ith_neighbors = idx[:, i]
        candidate_state = candidate_states[ith_neighbors, :]  # (batch_size, genes)
        delta = (candidate_state - current_state).detach()  # (batch_size, genes)
        cos_sim = F.cosine_similarity(
            delta[:, :genes], output[:, :genes], dim=1
        )  # (batch_size,)

        # t+1 direction
        candidates = cos_sim.detach() > 0
        if l == 1:
            squared_difference = torch.mean(torch.abs(output - delta), dim=1)
        else:
            squared_difference = torch.mean(torch.pow(output - delta, 2), dim=1)
        squared_difference = squared_difference[candidates]
        loss.append(torch.sum(squared_difference) / len(candidates))

        # t-1 direction, (-output) - delta
        candidates = cos_sim.detach() < 0
        if l == 1:
            squared_difference = torch.mean(torch.pow(output + delta, 1), dim=1)
        else:
            squared_difference = torch.mean(torch.pow(output + delta, 2), dim=1)
        squared_difference = squared_difference[candidates]
        loss.append(torch.sum(squared_difference) / len(candidates))
        # TODO: check why the memory usage is high for this one
    loss = torch.stack(loss).mean()
    return loss.detach().cpu().item()


def get_err_dv(adata, key, gene_mask):
    if gene_mask is None:
        gene_mask = np.array(range(adata.n_vars))
    nn_t_idx = get_neighbor_idx(adata)
    mse_u = _loss_dv(torch.tensor(adata.layers['velocity_unspliced'][:, gene_mask]),
                     torch.tensor(adata.layers['Mu'][:, gene_mask]),
                     torch.tensor(nn_t_idx, dtype=torch.long),
                     torch.tensor(adata.layers['Mu'][:, gene_mask]))
    mse_s = _loss_dv(torch.tensor(adata.layers['velocity'][:, gene_mask]),
                     torch.tensor(adata.layers['Ms'][:, gene_mask]),
                     torch.tensor(nn_t_idx, dtype=torch.long),
                     torch.tensor(adata.layers['Ms'][:, gene_mask]))
    mae_u = _loss_dv(torch.tensor(adata.layers['velocity_unspliced'][:, gene_mask]),
                     torch.tensor(adata.layers['Mu'][:, gene_mask]),
                     torch.tensor(nn_t_idx, dtype=torch.long),
                     torch.tensor(adata.layers['Mu'][:, gene_mask]),
                     l=1)
    mae_s = _loss_dv(torch.tensor(adata.layers['velocity'][:, gene_mask]),
                     torch.tensor(adata.layers['Ms'][:, gene_mask]),
                     torch.tensor(nn_t_idx, dtype=torch.long),
                     torch.tensor(adata.layers['Ms'][:, gene_mask]),
                     l=1)
    try:
        run_time = adata.uns[f'{key}_run_time']
    except KeyError:
        run_time = np.nan
    return mse_u+mse_s, None, mae_u+mae_s, None, None, None, run_time


##########################################################################
# Evaluation utility functions from PyroVelocity
# Reference:
# Qin, Q., Bingham, E., La Manno, G., Langenau, D. M., & Pinello, L. (2022).
# Pyro-Velocity: Probabilistic RNA Velocity inference from single-cell data.
# bioRxiv.
##########################################################################
def get_err_pv(adata, key, gene_mask, discrete=True):
    train_idx, test_idx = adata.uns[f"{key}_train_idx"], adata.uns[f"{key}_test_idx"]
    if discrete:
        U, S = adata.layers['unspliced'].A, adata.layers['spliced'].A
        Mu_u, Mu_s = adata.layers[f'{key}_ut'], adata.layers[f'{key}_st']
        Uhat, Shat = adata.layers[f'{key}_u'], adata.layers[f'{key}_s']
    else:
        U, S = adata.layers['Mu'], adata.layers['Ms']
        Uhat, Shat = adata.layers[f'{key}_u'], adata.layers[f'{key}_s']
        # MLE of variance
        var_s_train, var_s_test = np.var((S[train_idx]-Shat[train_idx]), 0), np.var((S[test_idx]-Shat[test_idx]), 0)
        var_u_train, var_u_test = np.var((U[train_idx]-Uhat[train_idx]), 0), np.var((U[test_idx]-Uhat[test_idx]), 0)

    if gene_mask is None:
        gene_mask = np.ones((adata.n_vars)).astype(bool)

    dist_u_train = np.abs(U[train_idx][:, gene_mask]-Uhat[train_idx][:, gene_mask])
    dist_s_train = np.abs(S[train_idx][:, gene_mask]-Shat[train_idx][:, gene_mask])
    dist_u_test = np.abs(U[test_idx][:, gene_mask]-Uhat[test_idx][:, gene_mask])
    dist_s_test = np.abs(S[test_idx][:, gene_mask]-Shat[test_idx][:, gene_mask])

    if discrete:
        logp_train = np.log(poisson.pmf(U[train_idx], Mu_u[train_idx])+1e-10).sum(1).mean()\
            + np.log(poisson.pmf(S[train_idx], Mu_s[train_idx])+1e-10).sum(1).mean()
        logp_test = np.log(poisson.pmf(U[test_idx], Mu_u[test_idx])+1e-10).sum(1).mean()\
            + np.log(poisson.pmf(S[test_idx], Mu_s[test_idx])+1e-10).sum(1).mean()
    else:
        logp_train = -dist_u_train**2/(2*var_u_train)-dist_s_train**2/(2*var_s_train)\
            - 0.5*np.log(var_u_train) - 0.5*np.log(var_s_train) - np.log(2*np.pi)
        logp_test = -dist_u_test**2/(2*var_u_test)-dist_s_test**2/(2*var_s_test)\
            - 0.5*np.log(var_u_test) - 0.5*np.log(var_s_test) - np.log(2*np.pi)

    mse_train = np.nanmean(dist_u_train**2+dist_s_train**2)
    mse_test = np.nanmean(dist_u_test**2+dist_s_test**2)
    mae_train = np.nanmean(dist_u_train+dist_s_train)
    mae_test = np.nanmean(dist_u_test+dist_s_test)

    try:
        run_time = adata.uns[f'{key}_run_time']
    except KeyError:
        run_time = np.nan

    return mse_train, mse_test, mae_train, mae_test, logp_train, logp_test, run_time


##########################################################################
# Evaluation utility functions from VeloVI
# Reference:
# Gayoso, Adam, et al. "Deep generative modeling of transcriptional dynamics
# for RNA velocity analysis in single cells." bioRxiv (2022).
##########################################################################
def get_err_velovi(adata, key, gene_mask):
    if gene_mask is None:
        gene_mask = np.ones((adata.n_vars)).astype(bool)
    U, S = adata.layers['Mu'][:, gene_mask], adata.layers['Ms'][:, gene_mask]
    Uhat, Shat = adata.layers[f'{key}_uhat'][:, gene_mask], adata.layers[f'{key}_shat'][:, gene_mask]

    try:
        run_time = adata.uns[f'{key}_run_time']
    except KeyError:
        run_time = np.nan

    try:
        train_idx = adata.uns[f'{key}_train_idx']
        test_idx = adata.uns[f'{key}_test_idx']
        dist_u_train = np.abs(U[train_idx]-Uhat[train_idx])
        dist_s_train = np.abs(S[train_idx]-Shat[train_idx])
        dist_u_test = np.abs(U[test_idx]-Uhat[test_idx])
        dist_s_test = np.abs(S[test_idx]-Shat[test_idx])

        mse_train = np.mean(dist_u_train**2+dist_s_train**2)
        mse_test = np.mean(dist_u_test**2+dist_s_test**2)
        mae_train = np.mean(dist_u_train+dist_s_train)
        mae_test = np.mean(dist_u_test+dist_s_test)
        logp_train = np.mean(adata.obs[f'{key}_likelihood'].to_numpy()[train_idx])
        logp_test = np.mean(adata.obs[f'{key}_likelihood'].to_numpy()[test_idx])
    except KeyError:
        dist_u = np.abs(U-Uhat)
        dist_s = np.abs(S-Shat)
        mse = np.mean(dist_u**2+dist_s**2)
        mae = np.mean(dist_u+dist_s)
        logp = np.mean(adata.obs[f'{key}_likelihood'].to_numpy())

        return mse, None, mae, None, logp, None, run_time

    return mse_train, mse_test, mae_train, mae_test, logp_train, logp_test, run_time


##########################################################################
# Evaluation Metrics from UniTVelo
# Reference:
# Gao, M., Qiao, C. & Huang, Y. UniTVelo: temporally unified RNA velocity
# reinforces single-cell trajectory inference. Nat Commun 13, 6586 (2022).
# https://doi.org/10.1038/s41467-022-34188-7
##########################################################################


def summary_scores(all_scores):
    # Summarize group scores.
    # Args:
    #    all_scores (dict{str,list}):
    #        {group name: score list of individual cells}.
    #
    # Returns:
    #    dict{str,float}:
    #        Group-wise aggregation scores.
    #    float:
    #        score aggregated on all samples

    sep_scores = {k: np.mean(s) for k, s in all_scores.items() if s}
    overal_agg = np.mean([s for k, s in sep_scores.items() if s])
    return sep_scores, overal_agg


def keep_type(adata, nodes, target, k_cluster):
    # Select cells of targeted type
    # Args:
    #    adata (Anndata):
    #        Anndata object.
    #    nodes (list):
    #        Indexes for cells
    #    target (str):
    #        Cluster name.
    #    k_cluster (str):
    #        Cluster key in adata.obs dataframe
    # Returns:
    #    list:
    #        Selected cells.

    return nodes[adata.obs[k_cluster][nodes].values == target]


def cross_boundary_correctness(
    adata,
    k_cluster,
    k_velocity,
    cluster_edges,
    return_raw=False,
    x_emb="X_umap",
    gene_mask=None
):
    # Cross-Boundary Direction Correctness Score (A->B)
    # Args:
    #    adata (Anndata):
    #        Anndata object.
    #    k_cluster (str):
    #        key to the cluster column in adata.obs DataFrame.
    #    k_velocity (str):
    #        key to the velocity matrix in adata.obsm.
    #    cluster_edges (list of tuples("A", "B")):
    #        pairs of clusters has transition direction A->B
    #    return_raw (bool):
    #        return aggregated or raw scores.
    #    x_emb (str):
    #        key to x embedding for visualization.
    #   gene_mask (numpy array):
    #       Boolean array to filter out non-velocity genes
    # Returns:
    #    dict:
    #        all_scores indexed by cluster_edges or mean scores indexed by cluster_edges
    #    float:
    #        averaged score over all cells.

    scores = {}
    all_scores = {}
    x_emb_name = x_emb
    if x_emb in adata.obsm:
        x_emb = adata.obsm[x_emb]
        if x_emb_name == "X_umap":
            v_emb = adata.obsm['{}_umap'.format(k_velocity)]
        else:
            v_emb = adata.obsm[[key for key in adata.obsm if key.startswith(k_velocity)][0]]
    else:
        x_emb = adata.layers[x_emb]
        v_emb = adata.layers[k_velocity]
        if gene_mask is None:
            gene_mask = ~np.isnan(v_emb[0])
        x_emb = x_emb[:, gene_mask]
        v_emb = v_emb[:, gene_mask]

    for u, v in cluster_edges:
        sel = adata.obs[k_cluster] == u
        nbs = adata.uns['neighbors']['indices'][sel]  # [n * 30]

        boundary_nodes = map(lambda nodes: keep_type(adata, nodes, v, k_cluster), nbs)
        x_points = x_emb[sel]
        x_velocities = v_emb[sel]

        type_score = []
        for x_pos, x_vel, nodes in zip(x_points, x_velocities, boundary_nodes):
            if len(nodes) == 0:
                continue
            position_dif = x_emb[nodes] - x_pos
            dir_scores = cosine_similarity(position_dif, x_vel.reshape(1, -1)).flatten()
            type_score.append(np.nanmean(dir_scores))
        if len(type_score) == 0:
            print(f'Warning: cell type transition pair ({u},{v}) does not exist in the KNN graph. Ignored.')
        else:
            scores[(u, v)] = np.nanmean(type_score)
            all_scores[(u, v)] = type_score

    if return_raw:
        return all_scores

    return scores, np.mean([sc for sc in scores.values()])


def _combine_scores(scores, cell_types):
    scores_combined = {}
    for i in range(len(cell_types)):
        u = cell_types[i]
        for j in range(i):
            v = cell_types[j]
            if not (u, v) in scores:
                continue
            if scores[(u, v)] > scores[(v, u)]:
                scores_combined[(u, v)] = scores[(u, v)]
            else:
                scores_combined[(v, u)] = scores[(v, u)]
    return scores_combined


def calibrated_cross_boundary_correctness(
    adata,
    k_cluster,
    k_velocity,
    k_time,
    cluster_edges=None,
    return_raw=False,
    x_emb="X_umap",
    gene_mask=None,
    k_std_t=None,
):
    # Calibrated Cross-Boundary Direction Correctness Score (A->B)
    # Args:
    #    adata (Anndata):
    #        Anndata object.
    #    k_cluster (str):
    #        key to the cluster column in adata.obs DataFrame.
    #    k_velocity (str):
    #        key to the velocity matrix in adata.obsm.
    #    k_time (str):
    #       key to the cell time in adata.obs
    #    cluster_edges (list of tuples("A", "B")):
    #        pairs of clusters has transition direction A->B
    #    return_raw (bool):
    #        return aggregated or raw scores.
    #    x_emb (str):
    #        key to x embedding for visualization.
    #   gene_mask (numpy array):
    #       Boolean array to filter out non-velocity genes
    # Returns:
    #    dict:
    #        all_scores indexed by cluster_edges or mean scores indexed by cluster_edges
    #    float:
    #        averaged score over all cells.
    #    dict:
    #        time score := proportion of cells with correct time order in a cell type transition
    #    float:
    #        averaged time score

    scores = {}
    all_scores = {}
    p_fw = {}

    eval_all = cluster_edges is None
    cell_types = np.unique(adata.obs[k_cluster].to_numpy())
    if eval_all:
        cluster_edges = [(u, v) for v in cell_types for u in cell_types]
    x_emb_name = x_emb
    if x_emb in adata.obsm:
        x_emb = adata.obsm[x_emb]
        if x_emb_name == "X_umap":
            v_emb = adata.obsm['{}_umap'.format(k_velocity)]
        else:
            v_emb = adata.obsm[[key for key in adata.obsm if key.startswith(k_velocity)][0]]
    else:
        x_emb = adata.layers[x_emb]
        v_emb = adata.layers[k_velocity]
        if gene_mask is None:
            gene_mask = ~np.isnan(v_emb[0])
        x_emb = x_emb[:, gene_mask]
        v_emb = v_emb[:, gene_mask]
    t = adata.obs[k_time].to_numpy()
    std_t = None if k_std_t is None else adata.obs[k_std_t].to_numpy()

    for u, v in cluster_edges:
        n1, n2 = 0, 0
        sel = adata.obs[k_cluster] == u
        nbs = adata.uns['neighbors']['indices'][sel]  # [n * 30]
        boundary_nodes = map(lambda nodes: keep_type(adata, nodes, v, k_cluster), nbs)

        x_points = x_emb[sel]
        x_velocities = v_emb[sel]

        type_score = []
        sel_indices = np.array(range(adata.n_obs))[sel]
        for x_pos, x_vel, nodes, idx in zip(x_points, x_velocities, boundary_nodes, sel_indices):
            if len(nodes) == 0:
                continue
            if std_t is None:
                p1 = t[nodes] >= t[idx]
                p2 = 1 - p1
            else:
                p1 = norm.cdf(np.zeros((len(nodes))),
                              loc=t[idx]-t[nodes],
                              scale=np.sqrt(std_t[nodes]**2+std_t[idx]**2))
                p2 = 1 - p1
            n1 += np.sum(p1)
            n2 += np.sum(p2)

            position_dif = x_emb[nodes] - x_pos
            dir_scores = cosine_similarity(position_dif, x_vel.reshape(1, -1)).flatten()
            type_score.extend(dir_scores*p1+(-dir_scores)*p2)

        if len(type_score) == 0:
            print(f'Warning: cell type transition pair ({u},{v}) does not exist in the KNN graph. Ignored.')
            pass
        else:
            scores[(u, v)] = np.nanmean(type_score)
            all_scores[(u, v)] = type_score
            p_fw[(u, v)] = n1 / (n1 + n2)

    if return_raw:
        return all_scores
    scores_combined = _combine_scores(scores, cell_types) if eval_all else scores
    return scores, np.mean([sc for sc in scores_combined.values()]), p_fw, np.mean([p for p in p_fw.values()])


def inner_cluster_coh(adata, k_cluster, k_velocity, return_raw=False):
    # In-cluster Coherence Score.
    #
    # Args:
    #    adata (Anndata):
    #        Anndata object.
    #    k_cluster (str):
    #        key to the cluster column in adata.obs DataFrame.
    #    k_velocity (str):
    #        key to the velocity matrix in adata.obsm.
    #    return_raw (bool):
    #        return aggregated or raw scores.
    #
    # Returns:
    #    dict:
    #        all_scores indexed by cluster_edges mean scores indexed by cluster_edges
    #    float:
    #        averaged score over all cells.
    #
    clusters = np.unique(adata.obs[k_cluster])
    scores = {}
    all_scores = {}

    for cat in clusters:
        sel = adata.obs[k_cluster] == cat
        nbs = adata.uns['neighbors']['indices'][sel]
        same_cat_nodes = map(lambda nodes: keep_type(adata, nodes, cat, k_cluster), nbs)

        velocities = adata.layers[k_velocity]
        nan_mask = ~np.isnan(velocities[0])
        velocities = velocities[:, nan_mask]

        cat_vels = velocities[sel]
        cat_score = [cosine_similarity(cat_vels[[ith]], velocities[nodes]).mean()
                     for ith, nodes in enumerate(same_cat_nodes)
                     if len(nodes) > 0]
        all_scores[cat] = cat_score
        scores[cat] = np.mean(cat_score)

    if return_raw:
        return all_scores

    return scores, np.mean([sc for sc in scores.values()])


##########################################################################
# End of Reference
##########################################################################

# New performance metric based on the idea of CBDir


def branch_fit_score(adata, k_cluster, k_velocity, branches, x_emb="X_umap"):
    cell_labels = adata.obs[k_cluster].to_numpy()
    x_emb_name = x_emb
    if x_emb in adata.obsm:
        x_emb = adata.obsm[x_emb]
        if x_emb_name == "X_umap":
            v_emb = adata.obsm['{}_umap'.format(k_velocity)]
        else:
            v_emb = adata.obsm[[key for key in adata.obsm if key.startswith(k_velocity)][0]]
    else:
        x_emb = adata.layers[x_emb]
        v_emb = adata.layers[k_velocity]

    all_boundary_nodes = {}
    for u in branches:
        sel = adata.obs[k_cluster] == u
        nbs = adata.uns['neighbors']['indices'][sel]  # [n * 30]
        for v in branches[u]:
            # cells of type v that are neighbors of type u
            boundary_nodes = map(lambda nodes: keep_type(adata, nodes, v, k_cluster), nbs)
            indices = np.concatenate([x for x in boundary_nodes]).astype(int)
            all_boundary_nodes[v] = indices

    for u in branches:
        sel = adata.obs[k_cluster] == u
        nbs = adata.uns['neighbors']['indices'][sel]  # [n * 30]
        branch_score = []
        for v in branches[u]:
            boundary_nodes = map(lambda nodes: keep_type(adata, nodes, v, k_cluster), nbs)
            x_points = x_emb[sel]
            x_velocities = v_emb[sel]

            type_score = []
            weights = []
            for x_pos, x_vel, nodes in zip(x_points, x_velocities, boundary_nodes):
                if len(nodes) == 0:
                    continue
                # check if the cell has the most transition to type v
                weights.append(np.sum(cell_labels[nodes] == v))

                # cosine similarity with desired neighbor
                position_dif = x_emb[nodes] - x_pos
                dir_scores = cosine_similarity(position_dif, x_vel.reshape(1, -1)).flatten()
                theta_1 = np.nanmean(np.arccos(dir_scores))/np.pi

                # cosine similarity with other branches
                dir_scores = np.array([])
                for w in branches[u]:
                    if w == v:
                        continue
                    position_dif = x_emb[all_boundary_nodes[w]] - x_pos
                    dir_scores = np.concatenate((dir_scores,
                                                 cosine_similarity(position_dif, x_vel.reshape(1, -1)).flatten()))
                theta_2 = np.nanmean(np.arccos(dir_scores))/np.pi
                type_score.append(theta_2-theta_1)
            branch_score.append(np.sum(np.array(type_score)*np.array(weights))/np.sum(weights))
    return branch_score, np.mean(branch_score)


# Cell Cycle Assignment based on scanpy
# Reference: https://github.com/scverse/scanpy_usage/blob/master/180209_cell_cycle/cell_cycle.ipynb
S_GENES_HUMAN = ['MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', 'MCM4', 'RRM1', 'UNG', 'GINS2', 'MCM6',
                 'CDCA7', 'DTL', 'PRIM1', 'UHRF1', 'MLF1IP', 'HELLS', 'RFC2', 'RPA2', 'NASP', 'RAD51AP1',
                 'GMNN', 'WDR76', 'SLBP', 'CCNE2', 'UBR7', 'POLD3', 'MSH2', 'ATAD2', 'RAD51', 'RRM2',
                 'CDC45', 'CDC6', 'EXO1', 'TIPIN', 'DSCC1', 'BLM', 'CASP8AP2', 'USP1', 'CLSPN', 'POLA1',
                 'CHAF1B', 'BRIP1', 'E2F8']
G2M_GENES_HUMAN = ['HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 'TPX2', 'TOP2A', 'NDC80', 'CKS2', 'NUF2',
                   'CKS1B', 'MKI67', 'TMPO', 'CENPF', 'TACC3', 'FAM64A', 'SMC4', 'CCNB2', 'CKAP2L', 'CKAP2',
                   'AURKB', 'BUB1', 'KIF11', 'ANP32E', 'TUBB4B', 'GTSE1', 'KIF20B', 'HJURP', 'CDCA3', 'HN1',
                   'CDC20', 'TTK', 'CDC25C', 'KIF2C', 'RANGAP1', 'NCAPD2', 'DLGAP5', 'CDCA2', 'CDCA8', 'ECT2',
                   'KIF23', 'HMMR', 'AURKA', 'PSRC1', 'ANLN', 'LBR', 'CKAP5', 'CENPE', 'CTCF', 'NEK2', 'G2E3',
                   'GAS2L3', 'CBX5', 'CENPA']

S_GENES_MOUSE = ['Mcm5', 'Pcna', 'Tyms', 'Fen1', 'Mcm2', 'Mcm4', 'Rrm1', 'Ung', 'Gins2', 'Mcm6',
                 'Cdca7', 'Dtl', 'Prim1', 'Uhrf1', 'Mlf1ip', 'Hells', 'Rfc2', 'Rpa2', 'Nasp', 'Rad51ap1',
                 'Gmnn', 'Wdr76', 'Slbp', 'Ccne2', 'Ubr7', 'Pold3', 'Msh2', 'Atad2', 'Rad51', 'Rrm2',
                 'Cdc45', 'Cdc6', 'Exo1', 'Tipin', 'Dscc1', 'Blm', 'Casp8ap2', 'Usp1', 'Clspn', 'Pola1',
                 'Chaf1b', 'Brip1', 'E2f8']

G2M_GENES_MOUSE = ['Hmgb2', 'Cdk1', 'Nusap1', 'Ube2c', 'Birc5', 'Tpx2', 'Top2a', 'Ndc80',
                   'Cks2', 'Nuf2', 'Cks1b', 'Mki67', 'Tmpo', 'Cenpf', 'Tacc3', 'Fam64a',
                   'Smc4', 'Ccnb2', 'Ckap2l', 'Ckap2', 'Aurkb', 'Bub1', 'Kif11', 'Anp32e',
                   'Tubb4b', 'Gtse1', 'Kif20b', 'Hjurp', 'Cdca3', 'Hn1', 'Cdc20', 'Ttk',
                   'Cdc25c', 'Kif2c', 'Rangap1', 'Ncapd2', 'Dlgap5', 'Cdca2', 'Cdca8',
                   'Ect2', 'Kif23', 'Hmmr', 'Aurka', 'Psrc1', 'Anln', 'Lbr', 'Ckap5',
                   'Cenpe', 'Ctcf', 'Nek2', 'G2e3', 'Gas2l3', 'Cbx5', 'Cenpa']


def assign_phase(adata, model='human', embed='umap', save=None):
    import scanpy as sc
    if model == 'human':
        s_genes, g2m_genes = S_GENES_HUMAN, G2M_GENES_HUMAN
    elif model == 'mouse':
        s_genes, g2m_genes = S_GENES_MOUSE, G2M_GENES_MOUSE
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
    cell_cycle_genes = s_genes + g2m_genes
    cell_cycle_genes = np.array([x in cell_cycle_genes for x in adata.var_names])
    adata_cc_genes = adata[:, cell_cycle_genes]
    sc.tl.pca(adata_cc_genes)
    sc.pl.scatter(adata_cc_genes, basis=embed, color='phase', save=save)
