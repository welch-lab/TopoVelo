"""Evaluation Module
Performs performance evaluation for various RNA velocity models and generates figures.
"""
import numpy as np
import pandas as pd
from os import makedirs
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from multiprocessing import cpu_count
from .evaluation_util import *
from .plot_config import PlotConfig
from ..scvelo_preprocessing.neighbors import neighbors
from ..plotting import set_dpi, get_colors, plot_cluster, plot_phase_grid, plot_sig_grid, plot_time_grid
from ..plotting import compute_figsize


def get_n_cpu(n_cell):
    # used for scVelo parallel jobs
    return int(min(cpu_count(), max(1, n_cell/2000)))


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


def get_velocity_metric(adata,
                        key,
                        vkey,
                        tkey,
                        cluster_key,
                        cluster_edges,
                        spatial_graph_key=None,
                        gene_mask=None,
                        embed='umap',
                        n_jobs=None):
    """
    Computes Cross-Boundary Direction Correctness and In-Cluster Coherence.
    The function calls scvelo.tl.velocity_graph.

    Args:
        adata (:class:`anndata.AnnData`):
            AnnData object.
        key (str):
            Key for cell time in the form of f'{key}_time'.
        vkey (str):
            Key for velocity in adata.obsm.
        tkey (str):
            Key for latent time in adata.obs.
        cluster_key (str):
            Key for cell type annotations.
        cluster_edges (list[tuple[str]]):
            List of ground truth cell type transitions.
            Each transition is of the form (A, B) where A is a progenitor
            cell type and B is a descendant type.
        spatial_graph_key (str, optional):
            Key for spatial graph.
        gene_mask (:class:`np.ndarray`, optional):
            Boolean array to filter out velocity genes. Defaults to None.
        embed (str, optional):
            Low-dimensional embedding. Defaults to 'umap'.
        n_jobs (_type_, optional):
            Number of parallel jobs. Defaults to None.

    Returns:
        tuple

            - **iccoh** (:class:`dict`): In-Cluster Coherence.
            - **mean_iccoh** (_type_): Mean In-Cluster Coherence.
            - **cbdir_embed** (:class:`dict`): Cross-Boundary Direction Correctness in embedding space.
            - **mean_cbdir_embed** (_type_): Mean Cross-Boundary Direction Correctness in embedding space.
            - **cbdir** (:class:`dict`): Cross-Boundary Direction Correctness in gene space.
            - **mean_cbdir** (_type_): Mean Cross-Boundary Direction Correctness in gene space.
            - **k_cbdir_embed** (:class:`dict`): K-step Cross-Boundary Direction Correctness in embedding space.
            - **mean_k_cbdir_embed** (_type_): Mean K-step Cross-Boundary Direction Correctness in embedding space.
            - **k_cbdir** (:class:`dict`): K-step Cross-Boundary Direction Correctness in gene space.
            - **mean_k_cbdir** (_type_): Mean K-step Cross-Boundary Direction Correctness in gene space.
            - **acc_embed** (:class:`dict`): Mann-Whitney U test in embedding space.
            - **mean_acc_embed** (_type_): Mean Mann-Whitney U test in embedding space.
            - **acc** (:class:`dict`): Mann-Whitney U test in gene space.
            - **mean_acc** (_type_): Mean Mann-Whitney U test in gene space.
            - **umtest_embed** (:class:`dict`): Mann-Whitney U test statistics in embedding space.
            - **mean_umtest_embed** (_type_): Mean Mann-Whitney U test statistics in embedding space.
            - **umtest** (:class:`dict`): Mann-Whitney U test statistics in gene space.
            - **mean_umtest** (_type_): Mean Mann-Whitney U test statistics in gene space.
            - **tscore** (:class:`dict`): Time score.
            - **mean_tscore** (_type_): Mean time score.
            - **mean_consistency_score** (_type_): Mean velocity consistency score.
            - **mean_sp_vel_consistency** (_type_): Mean spatial velocity consistency score.
    """
    mean_consistency_score = velocity_consistency(adata, vkey, gene_mask)
    mean_sp_vel_consistency = np.nan
    if spatial_graph_key is not None:
        mean_sp_vel_consistency = spatial_velocity_consistency(adata, vkey, spatial_graph_key, gene_mask)
    if len(cluster_edges) > 0:
        try:
            from scvelo.tl import velocity_graph, velocity_embedding
            n_jobs = get_n_cpu(adata.n_obs) if n_jobs is None else n_jobs
            gene_subset = adata.var_names if gene_mask is None else adata.var_names[gene_mask]
            velocity_graph(adata, vkey=vkey, gene_subset=gene_subset, n_jobs=n_jobs)
            velocity_embedding(adata, vkey=vkey, basis=embed)
        except ImportError:
            print("Please install scVelo to compute velocity embedding.\n"
                  "Skipping metrics 'Cross-Boundary Direction Correctness' and 'In-Cluster Coherence'.")
        iccoh, mean_iccoh = inner_cluster_coh(adata, cluster_key, vkey, gene_mask)
        cbdir_embed, mean_cbdir_embed = cross_boundary_correctness(adata,
                                                                   cluster_key,
                                                                   vkey,
                                                                   cluster_edges,
                                                                   spatial_graph_key,
                                                                   x_emb=f"X_{embed}")
        cbdir, mean_cbdir = cross_boundary_correctness(adata,
                                                       cluster_key,
                                                       vkey,
                                                       cluster_edges,
                                                       spatial_graph_key,
                                                       x_emb="Ms",
                                                       gene_mask=gene_mask)
        k_cbdir_embed, mean_k_cbdir_embed = gen_cross_boundary_correctness(adata,
                                                                           cluster_key,
                                                                           vkey,
                                                                           cluster_edges,
                                                                           tkey,
                                                                           spatial_graph_key,
                                                                           dir_test=False,
                                                                           x_emb=f"X_{embed}",
                                                                           gene_mask=gene_mask)
        
        k_cbdir, mean_k_cbdir = gen_cross_boundary_correctness(adata,
                                                               cluster_key,
                                                               vkey,
                                                               cluster_edges,
                                                               tkey,
                                                               spatial_graph_key,
                                                               dir_test=False,
                                                               x_emb="Ms",
                                                               gene_mask=gene_mask)
        (acc_embed, mean_acc_embed,
         umtest_embed, mean_umtest_embed) = gen_cross_boundary_correctness_test(adata,
                                                                                cluster_key,
                                                                                vkey,
                                                                                cluster_edges,
                                                                                tkey,
                                                                                spatial_graph_key,
                                                                                x_emb=f"X_{embed}",
                                                                                gene_mask=gene_mask)

        (acc, mean_acc,
         umtest, mean_umtest) = gen_cross_boundary_correctness_test(adata,
                                                                    cluster_key,
                                                                    vkey,
                                                                    cluster_edges,
                                                                    tkey,
                                                                    spatial_graph_key,
                                                                    x_emb="Ms",
                                                                    gene_mask=gene_mask)
        if not f'{key}_time' in adata.obs:
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
        mean_iccoh = np.nan
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
        iccoh = dict.fromkeys([])
    return (iccoh, mean_iccoh,
            cbdir_embed, mean_cbdir_embed,
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
        'In-Cluster Coherence': np.nan,
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
    if 'mean_iccoh' in kwargs:
        stats['In-Cluster Coherence'] = kwargs['mean_iccoh']
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
            print(f"Warning: {key} not found in index map, ignored.")
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
    multi_stats = pd.DataFrame(index=pd.Index(list(kwargs.keys())),
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
               cluster_key="clusters",
               gene_key='velocity_genes',
               cluster_edges=None,
               embed='umap',
               n_jobs=None):
    """
    Get performance metrics given a method.

    Args:
        adata (:class:`anndata.AnnData`):
            AnnData object.
        method (str):
            Model name. The velovae package also provides evaluation for other RNA velocity methods.
        key (str):
            Key in .var or .varm for extracting the ODE parameters learned by the model.
        vkey (str):
            Key in .layers for extracting rna velocity.
        tkey (str):
            Key in .obs for extracting latent time
        spatial_graph_key (str):
            Key in .obsp for extracting the spatial graph
        cluster_key (str, optional):
            Key in .obs for extracting cell type annotation. Defaults to "clusters".
        gene_key (str, optional):
            Key for filtering the genes.. Defaults to 'velocity_genes'.
        cluster_edges (list[tuple[str]], optional):
            List of ground truth cell type transitions.
            Each transition is of the form (A, B) where A is a progenitor
            cell type and B is a descendant type.
            Defaults to None.
        embed (str, optional):
            Low-dimensional embedding name.. Defaults to 'umap'.
        n_jobs (int, optional):
            Number of parallel jobs. Used in scVelo velocity graph computation.
            By default, it is automatically determined based on dataset size.
            Defaults to None.

    Returns:
        stats (:class:`pandas.DataFrame`):
            Stores the performance metrics. Rows are metric names and columns are method names.
    """
    if gene_key is not None and gene_key in adata.var:
        gene_mask = adata.var[gene_key].to_numpy()
    else:
        gene_mask = None

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
    (iccoh, mean_iccoh,
     cbdir_embed, mean_cbdir_embed,
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
                                                    spatial_graph_key,
                                                    gene_mask,
                                                    embed,
                                                    n_jobs)

    mean_sp_time_consistency = spatial_time_consistency(adata, tkey, spatial_graph_key)
    stats = gather_stats(mse_train=mse_train,
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
                         mean_sp_time_consistency=mean_sp_time_consistency)
    
    stats_type = gather_type_stats(cbdir=cbdir, cbdir_embed=cbdir_embed, tscore=tscore)
    multi_stats = gather_multistats(kcbdir=mean_k_cbdir,
                                    kcbdir_embed=mean_k_cbdir_embed,
                                    acc=mean_acc,
                                    acc_embed=mean_acc_embed,
                                    mwtest=mean_mwtest,
                                    mwtest_embed=mean_mwtest_embed)
    multi_stats_type = gather_type_multistats(kcbdir=k_cbdir,
                                              kcbdir_embed=k_cbdir_embed,
                                              acc=acc,
                                              acc_embed=acc_embed,
                                              mwtest=mwtest,
                                              mwtest_embed=mwtest_embed)
    return stats, stats_type, multi_stats, multi_stats_type


def post_analysis(adata,
                  test_id,
                  methods,
                  keys,
                  spatial_graph_key=None,
                  spatial_key=None,
                  n_spatial_neighbors=8,
                  gene_key='velocity_genes',
                  compute_metrics=False,
                  raw_count=False,
                  spatial_velocity_graph=False,
                  genes=[],
                  plot_type=['time', 'gene', 'stream'],
                  cluster_key="clusters",
                  cluster_edges=[],
                  nplot=500,
                  embed="umap",
                  grid_size=(1, 1),
                  phase_plot_config={},
                  gene_plot_config={},
                  time_plot_config={},
                  stream_plot_config={},
                  dpi=80,
                  figure_path=None,
                  save_anndata=None,
                  **kwargs):
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
            Key for spatial graph. Defaults to None.
        spatial_key (str, optional):
            Key for spatial embedding. Defaults to None.
        n_spatial_neighbors (int, optional):
            Number of spatial neighbors. Defaults to 8.
        gene_key (str, optional):
            Key for filtering the genes. Defaults to 'velocity_genes'.
        compute_metrics (bool, optional):
            Whether to compute performance metrics. Defaults to True.
        raw_count (bool, optional):
            Whether to use raw count for computing metrics. Defaults to False.
        spatial_velocity_graph (bool, optional):
            Whether to recompute the spatial graph. Defaults to False.
        genes (list[str], optional):
            List of gene names. Defaults to [].
        plot_type (list[str], optional):
            List of plot types. Defaults to ['time', 'gene', 'stream'].
        cluster_key (str, optional):
            Key for cell type annotation. Defaults to "clusters".
        cluster_edges (list[tuple[str]], optional):
            List of ground truth cell type transitions.
            Each transition is of the form (A, B) where A is a progenitor
            cell type and B is a descendant type.
            Defaults to [].
        nplot (int, optional):
            Number of cells to plot. Defaults to 500.
        embed (str, optional):
            Low-dimensional embedding name. Defaults to 'umap'.
        grid_size (tuple[int], optional):
            Grid size for plotting. Defaults to (1, 1).
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
        dict: Performance metrics.
        dict: Performance metrics for each cell type transition.
        dict: Performance metrics for each time step.
        dict: Performance metrics for each cell type transition and each time step.
    """
    # set the random seed
    random_state = 42 if 'random_state' not in kwargs else kwargs['random_state']
    np.random.seed(random_state)
    # dpi
    set_dpi(dpi)
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
                print(f"Warning: gene name {gene} not found in AnnData. Removed.")
                gene_rm.append(gene)
        for gene in gene_rm:
            genes.remove(gene)

        if len(gene_indices) == 0:
            print("Warning: No gene names found. Randomly select genes...")
            gene_indices = np.random.choice(adata.n_vars, grid_size[0]*grid_size[1], replace=False).astype(int)
            genes = adata.var_names[gene_indices].to_numpy()
    else:
        print("Warning: No gene names are provided. Randomly select genes...")
        gene_indices = np.random.choice(adata.n_vars, grid_size[0]*grid_size[1], replace=False).astype(int)
        genes = adata.var_names[gene_indices].to_numpy()
        print(genes)

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

    # recompute the spatial KNN graph
    if spatial_velocity_graph:
        if spatial_key is not None:
            print(f'Computing a spatial graph using KNN on {spatial_key} with k={n_spatial_neighbors}')
            if 'connectivities' in adata.obsp or 'neighbors' in adata.uns:
                print(f'Warning: overwriting the original KNN graph! (.uns, .obsp)')
            neighbors(adata, n_neighbors=n_spatial_neighbors, use_rep=spatial_key, method='sklearn')
        else:
            raise KeyError

    # Compute metrics and generate plots for each method
    for i, method in enumerate(methods):
        if compute_metrics:
            print(f'*** Computing performance metrics {i+1}/{len(methods)} ***')
            (stats_i, stats_type_i,
             multi_stats_i, multi_stats_type_i) = get_metric(adata,
                                                             method,
                                                             keys[i],
                                                             vkeys[i],
                                                             tkeys[i],
                                                             spatial_graph_key,
                                                             cluster_key,
                                                             gene_key,
                                                             cluster_edges,
                                                             embed,
                                                             n_jobs=(kwargs['n_jobs']
                                                                     if 'n_jobs' in kwargs
                                                                     else None))
            print('Finished. \n')
            stats_type_list.append(stats_type_i)
            multi_stats_list.append(multi_stats_i)
            multi_stats_type_list.append(multi_stats_type_i)
            # avoid duplicate methods with different keys
            method_ = f"{method} ({keys[i]})" if method in stats else method
            methods_display.append(method_)
            stats[method_] = stats_i

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
        stats_type_df = pd.concat(stats_type_list,
                                  axis=1,
                                  keys=methods_display,
                                  names=['Model'])
        multi_stats_df = pd.concat(multi_stats_list,
                                   axis=1,
                                   keys=methods_display,
                                   names=['Model'])
        multi_stats_type_df = pd.concat(multi_stats_type_list,
                                        axis=1,
                                        keys=methods_display,
                                        names=['Model'])
        pd.set_option("display.precision", 3)

    print("---   Plotting  Results   ---")

    # Generate plots
    if 'cluster' in plot_type or "all" in plot_type:
        plot_cluster(adata.obsm[f"X_{embed}"],
                     adata.obs[cluster_key].to_numpy(),
                     save=(None if figure_path is None else
                           f"{figure_path}/{test_id}_{embed}.png"))

    if "time" in plot_type or "all" in plot_type:
        X_embed = adata.obsm[f"X_{embed}"]
        T = {}
        capture_time = adata.obs["tprior"].to_numpy() if "tprior" in adata.obs else None
        for i, method in enumerate(methods):
            # avoid duplicate methods with different keys
            method_ = f"{method} ({keys[i]})" if method in T else method
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
        plot_time_grid(T,
                       X_embed,
                       capture_time,
                       None,
                       *plot_config.get_all())

    if "phase" in plot_type or "all" in plot_type:
        Labels_phase = {}
        Legends_phase = {}
        Labels_phase_demo = {}
        for i, method in enumerate(methods):
            # avoid duplicate methods with different keys
            method_ = f"{method} ({keys[i]})" if method in Labels_phase else method
            Labels_phase[method_] = cell_state(adata, method, keys[i], gene_indices)
            Legends_phase[method_] = ['Induction', 'Repression', 'Off', 'Unknown']
            Labels_phase_demo[method] = None
        plot_config = PlotConfig('phase')
        plot_config.set_multiple(phase_plot_config)
        plot_config.set('path', figure_path)
        plot_config.set('figname', f'{test_id}_phase')
        plot_phase_grid(grid_size[0],
                        grid_size[1],
                        genes,
                        U[:, gene_indices],
                        S[:, gene_indices],
                        Labels_phase,
                        Legends_phase,
                        Uhat,
                        Shat,
                        Labels_phase_demo,
                        *plot_config.get_all())

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
        plot_sig_grid(grid_size[0],
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
                      *plot_config.get_all())

    if 'cell velocity' in plot_type or 'all' in plot_type:
        plot_config = PlotConfig('stream')
        plot_config.set_multiple(stream_plot_config)
        if plot_config.get('palette') is None:
            palette = get_colors(len(cell_types_raw))
            plot_config.set('palette', palette)
        try:
            from scvelo.tl import velocity_graph
            from scvelo.pl import velocity_embedding_stream
            for i, vkey in enumerate(vkeys):
                if methods[i] in ['scVelo', 'UniTVelo', 'DeepVelo']:
                    gene_subset = adata.var_names[adata.var['velocity_genes'].to_numpy()]
                else:
                    gene_subset = adata.var_names[~np.isnan(adata.layers[vkey][0])]
                xkey = 'Ms' if 'xkey' not in kwargs else kwargs['xkey']
                velocity_graph(adata, vkey=vkey, xkey=xkey, gene_subset=gene_subset,
                               n_jobs=(kwargs['n_jobs']
                                       if 'n_jobs' in kwargs
                                       else get_n_cpu(adata.n_obs)))
                if 'spatial_graph_params' in adata.uns:
                    radius = adata.uns['spatial_graph_params']['radius']
                    if radius is not None:
                        adata.uns[f'{vkey}_graph'] = adata.uns[f'{vkey}_graph']\
                            .multiply(adata.obsp['distances'] < radius)
                        adata.uns[f'{vkey}_graph_neg'] = adata.uns[f'{vkey}_graph_neg']\
                            .multiply(adata.obsp['distances'] < radius)
                velocity_embedding_stream(adata,
                                          basis=embed,
                                          vkey=vkey,
                                          color=cluster_key,
                                          title="",
                                          figsize=(plot_config.get('width'), plot_config.get('height')),
                                          density=plot_config.get('density'),
                                          palette=plot_config.get('palette'),
                                          size=plot_config.get('markersize'),
                                          alpha=plot_config.get('alpha'),
                                          legend_loc=plot_config.get('legend_loc'),
                                          legend_fontsize=plot_config.get('legend_fontsize'),
                                          linewidth=plot_config.get('linewidth'),
                                          arrow_size=plot_config.get('arrow_size'),
                                          arrow_color=plot_config.get('arrow_color'),
                                          perc=plot_config.get('perc'),
                                          cutoff_perc=plot_config.get('cutoff_perc'),
                                          dpi=dpi,
                                          show=True,
                                          save=(None if figure_path is None else
                                                f'{figure_path}/{test_id}_{keys[i]}.png'))
        except ImportError:
            print('Please install scVelo in order to generate stream plots')
            pass
    
    # Cell velocity from the GNN spatial decoder
    if 'GNN cell velocity' in plot_type or 'all' in plot_type:
        plot_config = PlotConfig('stream')
        plot_config.set_multiple(stream_plot_config)
        if plot_config.get('palette') is None:
            palette = get_colors(len(cell_types_raw))
            plot_config.set('palette', palette)
        try:
            from scvelo.tl import velocity_graph
            from scvelo.pl import velocity_embedding_stream
            for i, vkey in enumerate(vkeys):
                if 'TopoVelo' in methods[i]:
                    # Clip the velocity to remove outliers
                    v = adata.obsm[f"{vkey}_{keys[i]}_xy"]
                    q1, q3 = np.quantile(v, 0.75, 0), np.quantile(v, 0.25, 0)
                    v = np.stack([np.clip(v[:, 0], q3[0]-1.5*(q1[0]-q3[0]), q1[0]+1.5*(q1[0]-q3[0])),
                                  np.clip(v[:, 1], q3[1]-1.5*(q1[0]-q3[0]), q1[1]+1.5*(q1[1]-q3[1]))], 1)
                    v = knn_smooth(v, adata.obsp["connectivities"])
                    adata.obsm[f"{vkey}_dec_{keys[i]}_xy"] = v
                    # Use predicted coordinates
                    adata.uns[f"{keys[i]}_velocity_params"]["embeddings"] = f"{keys[i]}_xy"
                    velocity_embedding_stream(adata,
                                              basis=f"{keys[i]}_xy",
                                              vkey=f"{vkey}_dec",
                                              recompute=False,
                                              color=cluster_key,
                                              title="",
                                              figsize=(plot_config.get('width'), plot_config.get('height')),
                                              density=plot_config.get('density'),
                                              palette=plot_config.get('palette'),
                                              size=plot_config.get('markersize'),
                                              alpha=plot_config.get('alpha'),
                                              legend_loc=plot_config.get('legend_loc'),
                                              legend_fontsize=plot_config.get('legend_fontsize'),
                                              linewidth=plot_config.get('linewidth'),
                                              arrow_size=plot_config.get('arrow_size'),
                                              arrow_color=plot_config.get('arrow_color'),
                                              perc=plot_config.get('perc'),
                                              cutoff_perc=plot_config.get('cutoff_perc'),
                                              dpi=dpi,
                                              show=True,
                                              save=(None if figure_path is None else
                                                    f'{figure_path}/{test_id}_{keys[i]}_cell_velocity.png'))
        except ImportError:
            print('Please install scVelo in order to generate stream plots')
            pass

    if save_anndata is not None:
        adata.write_h5ad(save_anndata)

    if compute_metrics:
        if figure_path is not None:
            stats_df.to_csv(f"{figure_path}/metrics_{test_id}.csv", sep='\t')
        return stats_df, stats_type_df, multi_stats_df, multi_stats_type_df

    return None, None, None, None
