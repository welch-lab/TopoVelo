"""Evaluation Module
Performs performance evaluation for various RNA velocity models and generates figures.
"""
import logging
import numpy as np
import pandas as pd
from os import makedirs
from scipy.stats import spearmanr
from multiprocessing import cpu_count
from scvelo.tl import velocity_graph, velocity_pseudotime
from sklearn.neighbors import NearestNeighbors
from .evaluation_util import *
from .velocity_util import *
from .plot_config import PlotConfig
from ..scvelo_preprocessing.neighbors import neighbors
from ..plotting import set_dpi, plot_cluster, plot_phase_grid, plot_sig_grid, plot_time_grid
logger = logging.getLogger(__name__)


def get_n_cpu(n_cell):
    # used for scVelo parallel jobs
    return int(min(cpu_count(), max(1, n_cell/2000)))


def infers_time(method, adata, tkey):
    return method not in ['STT'] and tkey in adata.obs


def get_velocity_metric_placeholder(cluster_edges):
    """Returns placeholder for velocity metrics.

    Args:
        cluster_edges (list[tuple[str]]): List of cell type transitions.

    Returns:
        tuple: Placeholder for velocity metrics.
    """
    # Convert tuples to a single string
    cluster_edges_ = []
    for pair in cluster_edges:
        cluster_edges_.append(f'{pair[0]} -> {pair[1]}')
    cbdir_embed = dict.fromkeys(cluster_edges_)
    cbdir = dict.fromkeys(cluster_edges_)
    tscore = dict.fromkeys(cluster_edges_)
    iccoh = dict.fromkeys(cluster_edges_)
    nan_arr = np.ones((5)) * np.nan
    return (iccoh, np.nan,
            cbdir_embed, np.nan,
            cbdir, np.nan,
            nan_arr,
            nan_arr,
            nan_arr,
            nan_arr,
            tscore, np.nan,
            np.nan,
            np.nan)


def _spatial_velocity_graph(
    adata,
    spatial_key,
    vkey,
    embed,
    gene_mask,
    n_velocity_neighbors,
    n_jobs
):
    try:
        from scvelo.tl import velocity_graph, velocity_embedding
        n_jobs = get_n_cpu(adata.n_obs) if n_jobs is None else n_jobs
        gene_subset = adata.var_names if gene_mask is None else adata.var_names[gene_mask]

        print('Recalculate velocity graph for evaluating (K)CBDir...')
        neighbors(adata, n_neighbors=n_velocity_neighbors, use_rep=spatial_key)
        velocity_graph(adata, vkey=vkey, gene_subset=gene_subset, n_jobs=n_jobs)
        velocity_embedding(adata, vkey=vkey, basis=embed)

        # Build a spatial graph to evaluate CBDir
        connectivities, distances = None, None
        nbs_info = None
        if 'connectivities' in adata.obsp or 'neighbors' in adata.uns:
            connectivities = adata.obsp['connectivities']
            distances = adata.obsp['distances']
            del adata.obsp['connectivities'], adata.obsp['distances']
            if neighbors in adata.uns:
                nbs_info = adata.uns['neighbors']
                del adata.uns['neighbors']
        adata.uns['neighbors'] = {
            'connectivities_key': 'connectivities',
            'distances_key': 'distances',
            'params': {
                'method': 'spatial',
                'metric': 'euclidean',
                'n_neighbors': n_velocity_neighbors,
            }
        }
        print(f'Build a spatial KNN graph with k={n_velocity_neighbors} for evaluating (K)CBDir...')
        knn = NearestNeighbors(n_neighbors=n_velocity_neighbors, n_jobs=n_jobs).fit(adata.obsm[spatial_key])
        adata.uns['neighbors']['indices'] = knn.kneighbors(adata.obsm[spatial_key], return_distance=False)
        

        
    except ImportError:
        logger.warning("Please install scVelo to compute velocity embedding.\n"
                        "Skipping metrics 'Cross-Boundary Direction Correctness'.")
    assert 'indices' in adata.uns['neighbors']
    return connectivities, distances, nbs_info


def get_velocity_metric(adata,
                        key,
                        vkey,
                        tkey,
                        cluster_key,
                        cluster_edges,
                        spatial_key='X_spatial',
                        n_velocity_neighbors=50,
                        spatial_graph_key=None,
                        gene_mask=None,
                        embed='spatial',
                        n_jobs=None):
    """
    Computes Cross-Boundary Direction Correctness (CBDC) and other performance metrics for RNA or cell velocity analysis.
    This function internally calls `scvelo.tl.velocity_graph`.

    Args:
        adata (anndata.AnnData):
            The AnnData object containing the single-cell data.
        key (str):
            Identifier for cell time, formatted as `{key}_time`.
        vkey (str):
            Key for accessing velocity data in `adata.obsm`.
        tkey (str):
            Key for accessing latent time information in `adata.obs`.
        cluster_key (str):
            Key for cell type annotations within the data.
        cluster_edges (list[tuple[str]]):
            List specifying the ground truth cell type transitions.
            Each transition is a tuple (A, B), where A is a progenitor cell type and B is a descendant type.
        spatial_key (str, optional):
            Key for spatial coordinates. Defaults to 'X_spatial'.
        n_velocity_neighbors (int, optional):
            Number of neighbors used for computing velocity graph and evaluate velocity metrics. 
            Defaults to 50.
        spatial_graph_key (str, optional):
            Key for the spatial graph, if applicable. Defaults to None.
        gene_mask (np.ndarray, optional):
            Boolean array used to filter velocity genes. Defaults to None.
        embed (str, optional):
            Specifies the low-dimensional embedding method. Defaults to 'umap'.
        n_jobs (int, optional):
            Number of parallel jobs to run. Defaults to None (uses single thread).

    Returns:
        tuple: A tuple containing several performance metrics:

            - cbdir_embed (dict): CBDC in embedding space.
            - mean_cbdir_embed (float): Mean CBDC in embedding space.
            - cbdir (dict): CBDC in gene space.
            - mean_cbdir (float): Mean CBDC in gene space.
            - k_cbdir_embed (dict): K-step CBDC in embedding space.
            - mean_k_cbdir_embed (float): Mean K-step CBDC in embedding space.
            - k_cbdir (dict): K-step CBDC in gene space.
            - mean_k_cbdir (float): Mean K-step CBDC in gene space.
            - acc_embed (dict): Accuracy derived from the Mann-Whitney U test in embedding space.
            - mean_acc_embed (float): Mean accuracy from the Mann-Whitney U test in embedding space.
            - acc (dict): Accuracy derived from the Mann-Whitney U test in gene space.
            - mean_acc (float): Mean accuracy from the Mann-Whitney U test in gene space.
            - umtest_embed (dict): Mann-Whitney U test statistics in embedding space.
            - mean_umtest_embed (float): Mean U test statistics in embedding space.
            - umtest (dict): Mann-Whitney U test statistics in gene space.
            - mean_umtest (float): Mean U test statistics in gene space.
            - tscore (dict): Time score assessing temporal ordering.
            - mean_tscore (float): Mean time score.
            - mean_consistency_score (float): Mean score for velocity consistency.
            - mean_sp_vel_consistency (float): Mean score for spatial velocity consistency.
    """
    connectivities, distances, nbs_info = _spatial_velocity_graph(
        adata,
        spatial_key,
        vkey,
        embed,
        gene_mask,
        n_velocity_neighbors,
        n_jobs
    )

    mean_consistency_score = velocity_consistency(adata, vkey, gene_mask)
    mean_sp_vel_consistency = np.nan
    if spatial_graph_key is not None:
        mean_sp_vel_consistency = spatial_velocity_consistency(adata, vkey, spatial_graph_key, gene_mask)

    if len(cluster_edges) > 0:
        cbdir_embed, mean_cbdir_embed = cross_boundary_correctness(adata,
                                                                   cluster_key,
                                                                   vkey,
                                                                   cluster_edges,
                                                                   spatial_graph_key=None,
                                                                   x_emb=f"X_{embed}",
                                                                   gene_mask=gene_mask)

        cbdir, mean_cbdir = cross_boundary_correctness(adata,
                                                       cluster_key,
                                                       vkey,
                                                       cluster_edges,
                                                       spatial_graph_key=None,
                                                       x_emb="Ms",
                                                       gene_mask=gene_mask)

        k_cbdir_embed, mean_k_cbdir_embed = gen_cross_boundary_correctness(adata,
                                                                           cluster_key,
                                                                           vkey,
                                                                           cluster_edges,
                                                                           tkey,
                                                                           spatial_graph_key=None,
                                                                           dir_test=False,
                                                                           x_emb=f"X_{embed}",
                                                                           gene_mask=gene_mask)

        k_cbdir, mean_k_cbdir = gen_cross_boundary_correctness(adata,
                                                               cluster_key,
                                                               vkey,
                                                               cluster_edges,
                                                               tkey,
                                                               spatial_graph_key=None,
                                                               dir_test=False,
                                                               x_emb="Ms",
                                                               gene_mask=gene_mask)

        (acc_embed, mean_acc_embed,
         umtest_embed, mean_umtest_embed) = gen_cross_boundary_correctness_test(adata,
                                                                                cluster_key,
                                                                                vkey,
                                                                                cluster_edges,
                                                                                tkey,
                                                                                spatial_graph_key=None,
                                                                                x_emb=f"X_{embed}",
                                                                                gene_mask=gene_mask)

        (acc, mean_acc,
         umtest, mean_umtest) = gen_cross_boundary_correctness_test(adata,
                                                                    cluster_key,
                                                                    vkey,
                                                                    cluster_edges,
                                                                    tkey,
                                                                    spatial_graph_key=None,
                                                                    x_emb="Ms",
                                                                    gene_mask=gene_mask)

        if f'{key}_time' not in adata.obs:
            tscore, mean_tscore = time_score(adata, 'latent_time', cluster_key, cluster_edges)
        else:
            try:
                tscore, mean_tscore = time_score(adata, f'{key}_time', cluster_key, cluster_edges)
            except KeyError:
                tscore, mean_tscore = np.nan, np.nan
    else:
        mean_cbdir_embed = np.nan
        mean_cbdir = np.nan
        mean_k_cbdir_embed = np.ones((5))*np.nan
        mean_k_cbdir = np.ones((5))*np.nan
        mean_acc_embed = np.ones((5))*np.nan
        mean_acc = np.ones((5))*np.nan
        mean_umtest_embed = np.ones((5))*np.nan
        mean_umtest = np.ones((5))*np.nan
        mean_tscore = np.nan
        mean_consistency_score = np.nan
        mean_sp_vel_consistency = np.nan
        cbdir_embed = dict.fromkeys([])
        cbdir = dict.fromkeys([])
        k_cbdir_embed = dict.fromkeys([])
        k_cbdir = dict.fromkeys([])
        acc_embed = dict.fromkeys([])
        acc = dict.fromkeys([])
        umtest_embed = dict.fromkeys([])
        umtest = dict.fromkeys([])
        tscore = dict.fromkeys([])
    
    # Recover the original KNN graph
    if connectivities is not None:
        adata.obsp['connectivities'] = connectivities
    if distances is not None:
        adata.obsp['distances'] = distances
    if nbs_info is not None:
        adata.uns['neighbors'] = nbs_info

    return (cbdir_embed, mean_cbdir_embed,
            cbdir, mean_cbdir,
            k_cbdir_embed, mean_k_cbdir_embed,
            k_cbdir, mean_k_cbdir,
            acc_embed, mean_acc_embed,
            acc, mean_acc,
            umtest_embed, mean_umtest_embed,
            umtest, mean_umtest,
            tscore, mean_tscore,
            mean_consistency_score,
            mean_sp_vel_consistency)


def gather_stats(**kwargs):
    """Helper function, used for gathering scalar performance metrics."""
    stats = {
        'MSE Train': np.nan,
        'MSE Test': np.nan,
        'MAE Train': np.nan,
        'MAE Test': np.nan,
        'LL Train': np.nan,
        'LL Test': np.nan,
        'Training Time': np.nan,
        'CBDir': np.nan,
        'CBDir (Gene Space)': np.nan,
        'Time Score': np.nan,
        'Velocity Consistency': np.nan,
        'Spatial Velocity Consistency': np.nan,
        'Spatial Time Consistency': np.nan
    }  # contains the performance metrics

    if 'mse_train' in kwargs:
        stats['MSE Train'] = kwargs['mse_train']
    if 'mse_test' in kwargs:
        stats['MSE Test'] = kwargs['mse_test']
    if 'mae_train' in kwargs:
        stats['MAE Train'] = kwargs['mae_train']
    if 'mae_test' in kwargs:
        stats['MAE Test'] = kwargs['mae_test']
    if 'logp_train' in kwargs:
        stats['LL Train'] = kwargs['logp_train']
    if 'logp_test' in kwargs:
        stats['LL Test'] = kwargs['logp_test']
    if 'corr' in kwargs:
        stats['Time Correlation'] = kwargs['corr']
    if 'mean_cbdir_embed' in kwargs:
        stats['CBDir'] = kwargs['mean_cbdir_embed']
    if 'mean_cbdir' in kwargs:
        stats['CBDir (Gene Space)'] = kwargs['mean_cbdir']
    if 'mean_tscore' in kwargs:
        stats['Time Score'] = kwargs['mean_tscore']
    if 'mean_vel_consistency' in kwargs:
        stats['Velocity Consistency'] = kwargs['mean_vel_consistency']
    if 'mean_sp_vel_consistency' in kwargs:
        stats['Spatial Velocity Consistency'] = kwargs['mean_sp_vel_consistency']
    if 'mean_sp_time_consistency' in kwargs:
        stats['Spatial Time Consistency'] = kwargs['mean_sp_time_consistency']
    return stats


def gather_type_stats(**kwargs):
    # Gathers pairwise velocity metrics
    type_dfs = []
    metrics = []
    index_map = {
        'cbdir': 'CBDir (Gene Space)',
        'cbdir_embed': 'CBDir',
        'tscore': 'Time Score'
    }
    for key in kwargs:
        try:
            metrics.append(index_map[key])
            type_dfs.append(pd.DataFrame.from_dict(kwargs[key], orient='index'))
        except KeyError:
            logger.warning(f"Warning: {key} not found in index map, ignored.")
            continue
    stats_type = pd.concat(type_dfs, axis=1).T
    stats_type.index = pd.Index(metrics)
    return stats_type


def gather_multistats(**kwargs):
    """Helper function, used for gathering multi-step performance metrics.
    """
    metrics = {
        'kcbdir': 'K-CBDir (Gene Space)',
        'kcbdir_embed': 'K-CBDir',
        'acc': 'Mann-Whitney Test (Gene Space)',
        'acc_embed': 'Mann-Whitney Test',
        'mwtest': 'Mann-Whitney Test Stats (Gene Space)',
        'mwtest_embed': 'Mann-Whitney Test Stats'
    }
    multi_stats = pd.DataFrame()
    for key in kwargs:
        for i, val in enumerate(kwargs[key]):
            multi_stats.loc[metrics[key], f'{i+1}-step'] = val
    return multi_stats


def gather_type_multistats(**kwargs):
    """Helper function, used for gathering multi-step performance metrics all each cell type transition pairs.
    """
    metrics = {
        'kcbdir': 'K-CBDir (Gene Space)',
        'kcbdir_embed': 'K-CBDir',
        'acc': 'Mann-Whitney Test (Gene Space)',
        'acc_embed': 'Mann-Whitney Test',
        'mwtest': 'Mann-Whitney Test Stats (Gene Space)',
        'mwtest_embed': 'Mann-Whitney Test Stats'
    }
    names = list(kwargs.keys())
    multi_stats = pd.DataFrame(index=pd.Index([metrics[key] for key in names]),
                               columns=pd.MultiIndex.from_product([[], []], names=['Transition', 'Step']))
    for key in kwargs:
        for transition in kwargs[key]:
            for i, val in enumerate(kwargs[key][transition]):
                multi_stats.loc[metrics[key], (transition, f'{i+1}-step')] = val
    return multi_stats


def get_metric(adata,
               method,
               key,
               vkey,
               tkey,
               spatial_graph_key,
               spatial_key,
               n_velocity_neighbors=50,
               cluster_key="clusters",
               gene_key='velocity_genes',
               cluster_edges=None,
               embed='spatial',
               n_jobs=None):
    """
    Compute performance metrics for a specified RNA velocity method.

    Args:
        adata (anndata.AnnData):
            The AnnData object containing the single-cell data.
        method (str):
            The name of the model being evaluated. The velovae package supports evaluation for various RNA velocity methods.
        key (str):
            The key in `.var` or `.varm` used to extract ODE parameters learned by the model.
        vkey (str):
            The key in `.layers` used to extract RNA velocity values.
        tkey (str):
            The key in `.obs` used to extract latent time information.
        spatial_graph_key (str):
            The key in `.obsp` to extract the spatial graph.
        cluster_key (str, optional):
            The key in `.obs` for extracting cell type annotations. Defaults to 'clusters'.
        gene_key (str, optional):
            The key used for filtering genes. Defaults to 'velocity_genes'.
        cluster_edges (list[tuple[str]], optional):
            A list of tuples representing ground truth cell type transitions. Each tuple is of the form (A, B), where A is a progenitor cell type and B is a descendant type. Defaults to None.
        embed (str, optional):
            The name of the low-dimensional embedding used, such as 'umap'. Defaults to 'umap'.
        n_jobs (int, optional):
            The number of parallel jobs to use in scVelo's velocity graph computation. If not specified, it is determined automatically based on the dataset size. Defaults to None.

    Returns:
        pandas.DataFrame:
            A DataFrame containing performance metrics. Rows correspond to metric names and columns correspond to method names.
    """
    if gene_key is not None and gene_key in adata.var:
        gene_mask = adata.var[gene_key].to_numpy()
    else:
        gene_mask = None
    
    # Calculate velocity pseudotime if the method does not infer latent cell time
    if tkey not in adata.obs:
        velocity_graph(
            adata,
            vkey=vkey,
            xkey='Ms',
            gene_subset=None if gene_mask is None else adata.var_names[gene_mask],
            n_jobs=n_jobs
        )
        velocity_pseudotime(
            adata,
            vkey=vkey
        )
        adata.obs[tkey] = adata.obs['velocity_pseudotime'].to_numpy()

    if method == 'scVelo':
        (mse_train, mse_test,
         mae_train, mae_test,
         logp_train, logp_test) = eval_scv(adata)
    elif method == 'Vanilla VAE':
        (mse_train, mse_test,
         mae_train, mae_test,
         logp_train, logp_test) = eval_vanilla(adata, key, gene_mask)
    elif 'VeloVAE' in method or 'TopoVelo' in method:
        (mse_train, mse_test,
         mae_train, mae_test,
         logp_train, logp_test) = eval_vae(adata, key, gene_mask, 'Rate Prior' in method)
    elif method == 'BrODE':
        (mse_train, mse_test,
         mae_train, mae_test,
         logp_train, logp_test) = eval_brode(adata, key, gene_mask)
    elif method == 'Discrete VeloVAE' or method == 'Discrete VeloVAE (Rate Prior)':
        (mse_train, mse_test,
         mae_train, mae_test,
         logp_train, logp_test) = eval_vae(adata, key, gene_mask, 'VeloVAE (Rate Prior)' in method, True)
    elif method == 'UniTVelo':
        (mse_train, mse_test,
         mae_train, mae_test,
         logp_train, logp_test) = eval_utv(adata, key, gene_mask)
    elif method == 'DeepVelo':
        (mse_train, mse_test,
         mae_train, mae_test,
         logp_train, logp_test) = eval_dv(adata, key, gene_mask)
    elif 'PyroVelocity' in method:
        if 'err' in adata.uns:
            mse_train, mse_test = adata.uns['err']['MSE Train'], adata.uns['err']['MSE Test']
            mae_train, mae_test = adata.uns['err']['MAE Train'], adata.uns['err']['MAE Test']
            logp_train, logp_test = adata.uns['err']['LL Train'], adata.uns['err']['LL Test']
        else:
            (mse_train, mse_test,
             mae_train, mae_test,
             logp_train, logp_test) = eval_pv(adata, key, gene_mask, 'Continuous' not in method)
    elif method == 'VeloVI':
        (mse_train, mse_test,
         mae_train, mae_test,
         logp_train, logp_test) = eval_velovi(adata, key, gene_mask)
    else:
        mse_train, mse_test = np.nan, np.nan
        mae_train, mae_test = np.nan, np.nan
        logp_train, logp_test = np.nan, np.nan

    if 'tprior' in adata.obs:
        tprior = adata.obs['tprior'].to_numpy()
        t = (adata.obs["latent_time"].to_numpy()
             if (method in ['scVelo', 'UniTVelo']) else
             adata.obs[f"{key}_time"].to_numpy())
        corr, pval = spearmanr(t, tprior)
    else:
        corr = np.nan

    # Compute velocity metrics using a subset of genes defined by gene_mask
    (cbdir_embed, mean_cbdir_embed,
     cbdir, mean_cbdir,
     k_cbdir_embed, mean_k_cbdir_embed,
     k_cbdir, mean_k_cbdir,
     acc_embed, mean_acc_embed,
     acc, mean_acc,
     mwtest_embed, mean_mwtest_embed,
     mwtest, mean_mwtest,
     tscore, mean_tscore,
     mean_consistency_score,
     mean_sp_vel_consistency) = get_velocity_metric(adata,
                                                    key,
                                                    vkey,
                                                    tkey,
                                                    cluster_key,
                                                    cluster_edges,
                                                    spatial_key,
                                                    n_velocity_neighbors,
                                                    spatial_graph_key,
                                                    gene_mask,
                                                    embed,
                                                    n_jobs)

    mean_sp_time_consistency = spatial_time_consistency(adata, tkey, spatial_graph_key)
    stats = gather_stats(
        mse_train=mse_train,
        mse_test=mse_test,
        mae_train=mae_train,
        mae_test=mae_test,
        logp_train=logp_train,
        logp_test=logp_test,
        corr=corr,
        mean_cbdir=mean_cbdir,
        mean_cbdir_embed=mean_cbdir_embed,
        mean_tscore=mean_tscore,
        mean_vel_consistency=mean_consistency_score,
        mean_sp_vel_consistency=mean_sp_vel_consistency,
        mean_sp_time_consistency=mean_sp_time_consistency
    )
    
    stats_type = gather_type_stats(cbdir=cbdir, cbdir_embed=cbdir_embed, tscore=tscore)
    multi_stats = gather_multistats(
        kcbdir=mean_k_cbdir,
        kcbdir_embed=mean_k_cbdir_embed,
        acc=mean_acc,
        acc_embed=mean_acc_embed,
        mwtest=mean_mwtest,
        mwtest_embed=mean_mwtest_embed
    )
    multi_stats_type = gather_type_multistats(
        kcbdir=k_cbdir,
        kcbdir_embed=k_cbdir_embed,
        acc=acc,
        acc_embed=acc_embed,
        mwtest=mwtest,
        mwtest_embed=mwtest_embed
    )

    return stats, stats_type, multi_stats, multi_stats_type


def _sanity_check(adata, spatial_graph_key, spatial_key, cluster_key, embed):
    if spatial_graph_key not in adata.obsp:
        logger.error(f"Spatial graph with key {spatial_graph_key} not found in .obsp. Please set `spatial_graph_key` properly.")
        return False
    if spatial_key not in adata.obsm:
        logger.error(f"Spatial coordinates with key {spatial_key} not found in .obsm. Please set `spatial_key` properly.")
        return False
    if cluster_key not in adata.obs:
        logger.error(f"Cell type annotations with key {cluster_key} not found in .obs. Please set `cluster_key` properly.")
        return False
    if f'X_{embed}' not in adata.obsm:
        logger.error(f"Embedding with key X_{embed} not found in .obsm. Please set `embed` properly.")
        return False
    return True


def post_analysis(
    adata,
    test_id,
    methods,
    keys,
    spatial_graph_key='spatial_graph',
    spatial_key='X_spatial',
    n_spatial_neighbors=50,
    gene_key=None,
    compute_metrics=False,
    raw_count=False,
    genes=[],
    plot_type=['time', 'cell velocity'],
    cluster_key="clusters",
    cluster_edges=[],
    nplot=500,
    embed="spatial",
    plot_basis="spatial",
    grid_size=(1, 1),
    cluster_plot_config={},
    phase_plot_config={},
    gene_plot_config={},
    time_plot_config={},
    stream_plot_config={},
    dpi=80,
    figure_path=None,
    save_anndata=None,
    **kwargs
):
    """Performs model evaluation and generates figures.

    Args:
        adata (:class:`anndata.AnnData`):
            AnnData object.
        test_id (str):
            Test ID.
        methods (list[str]):
            List of model names.
        keys (list[str]):
            List of keys for extracting model parameters.
        spatial_graph_key (str, optional):
            Key for spatial graph. Defaults to 'spatial_graph'.
        spatial_key (str, optional):
            Key for spatial embedding. Defaults to 'X_spatial'.
        n_spatial_neighbors (int, optional):
            Number of spatial neighbors used for computing velocity graph in velocity stream plots. Defaults to 30.
        gene_key (str, optional):
            Key for filtering the genes. Defaults to 'velocity_genes'.
        compute_metrics (bool, optional):
            Whether to compute performance metrics. Defaults to False.
        raw_count (bool, optional):
            Whether to use raw count for computing metrics. Defaults to False.
        genes (list[str], optional):
            List of gene names. Defaults to [].
        plot_type (list[str], optional):
            List of plot types. Defaults to ['time', 'cell velocity'].
        cluster_key (str, optional):
            Key for cell type annotation. Defaults to "clusters".
        cluster_edges (list[tuple[str]], optional):
            List of ground truth cell type transitions.
            Each transition is of the form (A, B) where A is a progenitor
            cell type and B is a descendant type. Defaults to [].
        nplot (int, optional):
            Number of cells to plot. Defaults to 500.
        embed (str, optional):
            Low-dimensional embedding for evaluating velocity metrics. Defaults to 'spatial'.
        plot_basis (str, optional):
            Low-dimensional embedding for plotting. Defaults to 'spatial'.
        grid_size (tuple[int], optional):
            Grid size for plotting. Defaults to (1, 1).
        cluster_plot_config (dict, optional):
            Configuration for cluster plot. Defaults to {}.
        phase_plot_config (dict, optional):
            Configuration for phase plot. Defaults to {}.
        gene_plot_config (dict, optional):
            Configuration for gene plot. Defaults to {}.
        time_plot_config (dict, optional):
            Configuration for time plot. Defaults to {}.
        stream_plot_config (dict, optional):
            Configuration for stream plot. Defaults to {}.
        dpi (int, optional):
            DPI for plotting. Defaults to 80.
        figure_path (str, optional):
            Path for saving figures. Defaults to None.
        save_anndata (str, optional):
            Filename for saving the AnnData object. Defaults to None.

    Returns:
        tuple: Contains:
            - :class:`pandas.DataFrame`: Performance metrics.
            - :class:`pandas.DataFrame`: Performance metrics for each cell type transition.
            - :class:`pandas.DataFrame`: Performance metrics for each time step.
            - :class:`pandas.DataFrame`: Performance metrics for each cell type transition and each time step.
    """
    # sanity check
    if not _sanity_check(adata, spatial_graph_key, spatial_key, cluster_key, embed):
        return None, None, None, None
    # Specify plotting basis
    sp_basis = spatial_key[2:]
    
    # set the random seed
    random_state = 42 if 'random_state' not in kwargs else kwargs['random_state']
    np.random.seed(random_state)

    # dpi
    set_dpi(dpi)

    # Check figure path
    if figure_path is not None:
        makedirs(figure_path, exist_ok=True)

    # Retrieve data
    if raw_count:
        U, S = adata.layers["unspliced"].A, adata.layers["spliced"].A
    else:
        U, S = adata.layers["Mu"], adata.layers["Ms"]

    # Retrieve cell type labels and encode them as integers
    cell_labels_raw = adata.obs[cluster_key].to_numpy()
    cell_types_raw = np.unique(cell_labels_raw)
    label_dic = {}
    for i, x in enumerate(cell_types_raw):
        label_dic[x] = i
    cell_labels = np.array([label_dic[x] for x in cell_labels_raw])

    # Get gene indices
    if len(genes) > 0:
        gene_indices = []
        gene_rm = []
        for gene in genes:
            idx = np.where(adata.var_names == gene)[0]
            if len(idx) > 0:
                gene_indices.append(idx[0])
            else:
                logger.warning(f"Warning: gene name {gene} not found in AnnData. Removed.")
                gene_rm.append(gene)
        for gene in gene_rm:
            genes.remove(gene)

        if len(gene_indices) == 0:
            logger.warning("No gene names found. Randomly select genes...")
            gene_indices = np.random.choice(adata.n_vars, grid_size[0]*grid_size[1], replace=False).astype(int)
            genes = adata.var_names[gene_indices].to_numpy()
    elif 'gene' in plot_type or 'all' in plot_type:
        logger.warning("No gene names are provided. Randomly select genes...")
        gene_indices = np.random.choice(adata.n_vars, grid_size[0]*grid_size[1], replace=False).astype(int)
        genes = adata.var_names[gene_indices].to_numpy()

    stats = {}
    stats_type_list, multi_stats_list, multi_stats_type_list = [], [], []
    methods_display = []  # allows comparing multiple instances of the same model type

    # Stores the prediction for each method for plotting
    Uhat, Shat, V = {}, {}, {}
    That, Yhat = {}, {}
    vkeys, tkeys = [], []
    for i, method in enumerate(methods):
        vkey = 'velocity' if method in ['scVelo', 'UniTVelo', 'DeepVelo'] else f'{keys[i]}_velocity'
        vkeys.append(vkey)
        tkey = 'latent_time' if method == 'scVelo' else f'{keys[i]}_time'
        tkeys.append(tkey)

    # Compute metrics and generate plots for each method
    for i, method in enumerate(methods):
        # Compute metrics
        if compute_metrics:
            print(f'*** Computing performance metrics {i+1}/{len(methods)} ***')
            try:
                (stats_i,
                 stats_type_i,
                 multi_stats_i,
                 multi_stats_type_i) = get_metric(
                    adata,
                    method,
                    keys[i],
                    vkeys[i],
                    tkeys[i],
                    spatial_graph_key,
                    spatial_key,
                    n_spatial_neighbors,
                    cluster_key,
                    gene_key,
                    cluster_edges,
                    embed,
                    n_jobs=(kwargs['n_jobs']
                            if 'n_jobs' in kwargs
                            else None)
                )
                print('Finished. \n')
                stats_type_list.append(stats_type_i)
                multi_stats_list.append(multi_stats_i)
                multi_stats_type_list.append(multi_stats_type_i)
                # avoid duplicate methods with different keys
                method_ = f"{method} ({keys[i]})" if method in stats else method
                methods_display.append(method_)
                stats[method_] = stats_i
            except KeyError:
                logger.warning(f"Error: model ({method})  with key={keys[i]} not found in AnnData. Skipping...")
                continue
        # Compute prediction for the purpose of plotting (a fixed number of plots)
        if 'phase' in plot_type or 'gene' in plot_type or 'all' in plot_type:
            # avoid duplicate methods with different keys
            method_ = f"{method} ({keys[i]})" if method in V else method

            if method == 'scVelo':
                t_i, Uhat_i, Shat_i = get_pred_scv_demo(adata, keys[i], genes, nplot)
                Yhat[method_] = np.concatenate((np.zeros((nplot)), np.ones((nplot))))
                V[method_] = adata.layers["velocity"][:, gene_indices]
            elif method == 'Vanilla VAE':
                t_i, Uhat_i, Shat_i = get_pred_vanilla_demo(adata, keys[i], genes, nplot)
                Yhat[method_] = None
                V[method_] = adata.layers[f"{keys[i]}_velocity"][:, gene_indices]
            elif 'VeloVAE' in method or 'TopoVelo' in method:
                Uhat_i, Shat_i = get_pred_velovae_demo(adata, keys[i], genes, 'Rate Prior' in method, 'Discrete' in method)
                V[method_] = adata.layers[f"{keys[i]}_velocity"][:, gene_indices]
                t_i = adata.obs[f'{keys[i]}_time'].to_numpy()
                Yhat[method_] = cell_labels
            elif method == 'BrODE':
                t_i, y_i, Uhat_i, Shat_i = get_pred_brode_demo(adata, keys[i], genes, N=100)
                V[method_] = adata.layers[f"{keys[i]}_velocity"][:, gene_indices]
                Yhat[method_] = y_i
            elif method == "UniTVelo":
                t_i, Uhat_i, Shat_i = get_pred_utv_demo(adata, genes, nplot)
                V[method_] = adata.layers["velocity"][:, gene_indices]
                Yhat[method_] = None
            elif method == "DeepVelo":
                t_i = adata.obs[f'{keys[i]}_time'].to_numpy()
                V[method_] = adata.layers["velocity"][:, gene_indices]
                Uhat_i = adata.layers["Mu"][:, gene_indices]
                Shat_i = adata.layers["Ms"][:, gene_indices]
                Yhat[method_] = None
            elif method in ["PyroVelocity", "Continuous PyroVelocity"]:
                t_i = adata.obs[f'{keys[i]}_time'].to_numpy()
                Uhat_i = adata.layers[f'{keys[i]}_u'][:, gene_indices]
                Shat_i = adata.layers[f'{keys[i]}_s'][:, gene_indices]
                V[method_] = adata.layers[f"{keys[i]}_velocity"][:, gene_indices]
                Yhat[method_] = cell_labels
            elif method == "VeloVI":
                t_i = adata.layers['fit_t'][:, gene_indices]
                Uhat_i = adata.layers[f'{keys[i]}_uhat'][:, gene_indices]
                Shat_i = adata.layers[f'{keys[i]}_shat'][:, gene_indices]
                V[method_] = adata.layers[f"{keys[i]}_velocity"][:, gene_indices]
                Yhat[method_] = cell_labels
            elif method == "cellDancer":
                t_i = adata.obs[f'{keys[i]}_time'].to_numpy()
                Uhat_i = adata.layers["Mu"][:, gene_indices]
                Shat_i = adata.layers["Ms"][:, gene_indices]
                V[method_] = adata.layers[f"{keys[i]}_velocity"][:, gene_indices]
                Yhat[method_] = cell_labels

            That[method_] = t_i
            Uhat[method_] = Uhat_i
            Shat[method_] = Shat_i

    if compute_metrics:
        print("---     Integrating Peformance Metrics     ---")
        print(f"Dataset Size: {adata.n_obs} cells, {adata.n_vars} genes")
        stats_df = pd.DataFrame(stats)
        stats_type_df = pd.concat(
            stats_type_list, axis=1, keys=methods_display, names=['Model']
        )
        multi_stats_df = pd.concat(
            multi_stats_list, axis=1, keys=methods_display, names=['Model']
        )
        multi_stats_type_df = pd.concat(
            multi_stats_type_list, axis=1, keys=methods_display, names=['Model']
        )
        pd.set_option("display.precision", 3)

    if plot_type:
        print("---   Plotting  Results   ---")

    # Generate plots
    if 'cluster' in plot_type or "all" in plot_type:
        plot_config = PlotConfig('cluster')
        plot_config.set_multiple(cluster_plot_config)
        if figure_path is not None:
            plot_config.set('save', f"{figure_path}/{test_id}-{plot_basis}.png")
        else:
            plot_config.set('save', None)
        plot_cluster(
            adata.obsm[f"X_{plot_basis}"], adata.obs[cluster_key].to_numpy(), *plot_config.get_all()
        )

    if "time" in plot_type or "all" in plot_type:
        T = {}
        repeated = {}
        capture_time = adata.obs["tprior"].to_numpy() if "tprior" in adata.obs else None
        for i, method in enumerate(methods):
            # avoid duplicate methods with different keys
            method_ = f"{method} ({keys[i]})" if method in T else method
            if keys[i] in repeated:
                continue
            repeated[keys[i]] = True
            if method == 'scVelo':
                T[method_] = adata.obs["latent_time"].to_numpy()
            else:
                T[method_] = adata.obs[f"{keys[i]}_time"].to_numpy()
        k = len(methods)+(capture_time is not None)
        if k > 5:
            n_col = max(int(np.sqrt(k*2)), 1)
            n_row = k // n_col
            n_row += (n_row*n_col < k)
        else:
            n_row = 1
            n_col = k
        plot_config = PlotConfig('time')
        plot_config.set_multiple(time_plot_config)
        plot_config.set('path', figure_path)
        plot_config.set('figname', f'{test_id}_time')
        plot_time_grid(
            T,
            adata.obsm[f"X_{plot_basis}"],
            capture_time,
            None,
            *plot_config.get_all()
        )

    if "phase" in plot_type or "all" in plot_type:
        Labels_phase = {}
        Legends_phase = {}
        Labels_phase_demo = {}
        repeated = {}
        for i, method in enumerate(methods):
            # avoid duplicate methods with different keys
            method_ = f"{method} ({keys[i]})" if method in Labels_phase else method
            if keys[i] in repeated:
                continue
            repeated[keys[i]] = True
            Labels_phase[method_] = cell_state(adata, method, keys[i], gene_indices)
            Legends_phase[method_] = ['Induction', 'Repression', 'Off', 'Unknown']
            Labels_phase_demo[method] = None
        plot_config = PlotConfig('phase')
        plot_config.set_multiple(phase_plot_config)
        plot_config.set('path', figure_path)
        plot_config.set('figname', f'{test_id}_phase')
        plot_phase_grid(
            grid_size[0],
            grid_size[1],
            genes,
            U[:, gene_indices],
            S[:, gene_indices],
            Labels_phase,
            Legends_phase,
            Uhat,
            Shat,
            Labels_phase_demo,
            *plot_config.get_all()
        )

    if 'gene' in plot_type or 'all' in plot_type:
        T = {}
        Labels_sig = {}
        Legends_sig = {}
        for i, method in enumerate(methods):
            # avoid duplicate methods with different keys
            method_ = f"{method} ({keys[i]})" if method in Labels_sig else method
            Labels_sig[method_] = np.array([label_dic[x] for x in adata.obs[cluster_key].to_numpy()])
            Legends_sig[method_] = cell_types_raw
            if method == 'scVelo':
                T[method_] = adata.layers[f"{keys[i]}_t"][:, gene_indices]
                T['scVelo Global'] = adata.obs['latent_time'].to_numpy()*20
                Labels_sig['scVelo Global'] = Labels_sig[method]
                Legends_sig['scVelo Global'] = cell_types_raw
            elif method == 'UniTVelo':
                T[method_] = adata.layers["fit_t"][:, gene_indices]
            elif method == 'VeloVI':
                T[method_] = adata.layers["fit_t"][:, gene_indices]
            else:
                T[method_] = adata.obs[f"{keys[i]}_time"].to_numpy()
        plot_config = PlotConfig('gene')
        plot_config.set_multiple(gene_plot_config)
        plot_config.set('path', figure_path)
        plot_config.set('figname', f'{test_id}_gene')
        plot_sig_grid(
            grid_size[0],
            grid_size[1],
            genes,
            T,
            U[:, gene_indices],
            S[:, gene_indices],
            Labels_sig,
            Legends_sig,
            That,
            Uhat,
            Shat,
            V,
            Yhat,
            *plot_config.get_all()
        )

    if 'embed velocity' in plot_type or 'all' in plot_type:
        sp_basis = spatial_key[2:]
        for i, (method, key, vkey) in enumerate(zip(methods, keys, vkeys)):
            velocity_stream(
                adata,
                method,
                key,
                vkey,
                cell_types_raw,
                plot_basis,
                cluster_key,
                n_spatial_neighbors,
                dpi,
                save=(None if figure_path is None else
                        f'{figure_path}/vel-{plot_basis}-{test_id}-{keys[i]}.png'),
                stream_plot_config=stream_plot_config
            )

    if 'cell velocity' in plot_type or 'all' in plot_type:
        for i, (method, key, vkey) in enumerate(zip(methods, keys, vkeys)):
            velocity_stream(
                adata,
                method,
                key,
                vkey,
                cell_types_raw,
                sp_basis,
                cluster_key,
                n_spatial_neighbors,
                dpi,
                save=(None if figure_path is None else
                      f'{figure_path}/vel-xy-{test_id}-{keys[i]}.png'),
                stream_plot_config=stream_plot_config
            )

    # Cell velocity from the GNN spatial decoder
    if 'GNN cell velocity' in plot_type or 'all' in plot_type:
        for i, (method, key, vkey) in enumerate(zip(methods, keys, vkeys)):
            velocity_stream_on_pred_xy(
                adata,
                method,
                key,
                vkey,
                cell_types_raw,
                cluster_key,
                dpi,
                save=(None if figure_path is None else
                      f'{figure_path}/vel-{key}-predxy-{test_id}-{keys[i]}.png'),
                stream_plot_config=stream_plot_config
            )

    if save_anndata is not None:
        adata.write_h5ad(save_anndata)

    if compute_metrics:
        if figure_path is not None:
            stats_df.to_csv(f"{figure_path}/metrics_{test_id}.csv", sep='\t')
        return stats_df, stats_type_df, multi_stats_df, multi_stats_type_df

    return None, None, None, None
