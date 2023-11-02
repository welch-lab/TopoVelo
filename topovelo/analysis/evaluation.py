"""Evaluation Module
Performs performance evaluation for various RNA velocity models and generates figures.
"""
import numpy as np
import pandas as pd
from os import makedirs
from .evaluation_util import *
from topovelo.plotting import set_dpi, get_colors, plot_cluster, plot_phase_grid, plot_sig_grid, plot_time_grid
from multiprocessing import cpu_count
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from ..scvelo_preprocessing.neighbors import neighbors


def get_n_cpu(n_cell):
    # used for scVelo parallel jobs
    return int(min(cpu_count(), max(1, n_cell/2000)))


def get_velocity_metric_placeholder(cluster_edges):
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

            - dict: In-Cluster Coherence per cell type transition
            - float: Mean In-Cluster Coherence
            - dict: CBDir (embedding) per cell type transition
            - float: Mean CBDir (embedding)
            - dict: CBDir per cell type transition
            - float: Mean CBDir
            - :class:`numpy.ndarray` Mean k-step CBDir (embedding)
            - :class:`numpy.ndarray` Mean k-step CBDir
            - :class:`numpy.ndarray` k-step Mann-Whitney U test result (embedding)
            - :class:`numpy.ndarray` k-step Mann-Whitney U test result
            - dict: Time Accuracy Score per cell type transition
            - float: Mean Time Accuracy Score
            - float: Velocity Consistency
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
    # Helper function, used for gathering scalar performance metrics
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
            Stores the performance metrics. Rows are metric names and columns are method names
    """
    if gene_key is not None and gene_key in adata.var:
        gene_mask = adata.var[gene_key].to_numpy()
    else:
        gene_mask = None

    if method == 'scVelo':
        (mse_train, mse_test,
         mae_train, mae_test,
         logp_train, logp_test) = get_err_scv(adata)
    elif method == 'Vanilla VAE':
        (mse_train, mse_test,
         mae_train, mae_test,
         logp_train, logp_test) = get_err_vanilla(adata, key, gene_mask)
    elif method == 'Cycle VAE':
        (mse_train, mse_test,
         mae_train, mae_test,
         logp_train, logp_test) = get_err_cycle(adata, key, gene_mask)
    elif 'VeloVAE' in method or 'TopoVelo' in method:
        (mse_train, mse_test,
         mae_train, mae_test,
         logp_train, logp_test) = get_err_velovae(adata, key, gene_mask, 'Rate Prior' in method)
    elif method == 'BrODE':
        (mse_train, mse_test,
         mae_train, mae_test,
         logp_train, logp_test) = get_err_brode(adata, key, gene_mask)
    elif method == 'Discrete VeloVAE' or method == 'Discrete VeloVAE (Rate Prior)':
        (mse_train, mse_test,
         mae_train, mae_test,
         logp_train, logp_test) = get_err_velovae(adata, key, gene_mask, 'VeloVAE (Rate Prior)' in method, True)
    elif method == 'UniTVelo':
        (mse_train, mse_test,
         mae_train, mae_test,
         logp_train, logp_test) = get_err_utv(adata, key, gene_mask)
    elif method == 'DeepVelo':
        (mse_train, mse_test,
         mae_train, mae_test,
         logp_train, logp_test) = get_err_dv(adata, key, gene_mask)
    elif 'PyroVelocity' in method:
        if 'err' in adata.uns:
            mse_train, mse_test = adata.uns['err']['MSE Train'], adata.uns['err']['MSE Test']
            mae_train, mae_test = adata.uns['err']['MAE Train'], adata.uns['err']['MAE Test']
            logp_train, logp_test = adata.uns['err']['LL Train'], adata.uns['err']['LL Test']
        else:
            (mse_train, mse_test,
             mae_train, mae_test,
             logp_train, logp_test) = get_err_pv(adata, key, gene_mask, 'Continuous' not in method)
    elif method == 'VeloVI':
        (mse_train, mse_test,
         mae_train, mae_test,
         logp_train, logp_test) = get_err_velovi(adata, key, gene_mask)
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
                  compute_metrics=True,
                  raw_count=False,
                  spatial_velocity_graph=False,
                  genes=[],
                  plot_type=['time', 'gene', 'stream'],
                  cluster_key="clusters",
                  cluster_edges=[],
                  nplot=500,
                  frac=0.0,
                  embed="umap",
                  time_colormap='plasma',
                  dot_size=50,
                  grid_size=(1, 1),
                  sparsity_correction=True,
                  stream_figsize=None,
                  dpi=80,
                  figure_path=None,
                  save=None,
                  **kwargs):
    """High-level API for method evaluation and plotting after training.
    This function computes performance metrics and generates plots based on user input.

    Args:
        adata (:class:`anndata.AnnData`):
            AnnData object.
        test_id (str):
            Used for naming the figures.
            For example, it can be set as the name of the dataset.
        methods (list[str]):
            Contains the methods to compare with.
            Now supports {'scVelo', 'UniTVelo', 'DeepVelo', 'cellDancer', 'VeloVI', 'PyroVelocity',
            'VeloVAE', 'VeloVAE (Rate Prior)', 'Discrete VeloVAE', 'Discrete VeloVAE (Rate Prior)', 'BrODE',
            'TopoVelo (GCN)', 'TopoVelo (GAT)'}.
        keys (list[str]):
            Used for extracting ODE parameters from .var or .varm from anndata
            It should be of the same length as methods.
        spatial_graph_key (str):
            Key in .obsp storing the adjacency matrix of a spatial graph.
        spatial_key (str):
            Key in .obsm storing spatial coordinates
        n_spatial_neighbors (int, optional):
            Number of neighbors in the spatial KNN graph,
            effective only if spatial_key is not None
        gene_key (str, optional):
            Key in .var for gene filtering. Usually set to select velocity genes.
            Defaults to 'velocity_genes'.
        compute_metrics (bool, optional):
            Whether to compute the performance metrics for the methods. Defaults to True.
        raw_count (bool, optional):
            Whether to plot raw count numbers for discrete models. Defaults to False.
        spatial_velocity_graph (bool, optional):
            Whether to use the spatial knn graph in velocity graph computation.
            Affects the velocity stream plot.
            Defaults to False.
        genes (list[str], optional):
            Genes to plot. Used when plot_type contains "phase" or "gene".
            If not provided, gene(s) will be randomly sampled for plotting. Defaults to [].
        plot_type (list, optional):
            Type of plots to generate.
            Now supports {'time', 'gene', 'stream', 'phase', 'cluster'}.
            Defaults to ['time', 'gene', 'stream'].
        cluster_key (str, optional):
            Key in .obs containing the cell type annotations. Defaults to "clusters".
        cluster_edges (list[str], optional):
            List of ground-truth cell type ancestor-descendant relations, e.g. (A, B)
            means cell type A is the ancestor of type B. This is used for computing
            velocity metrics. Defaults to [].
        nplot (int, optional):
            Number of data points in the line prediction.
            This is to save memory. For plotting line predictions, we don't need
            as many points as the original dataset contains. Defaults to 500.
        frac (float, optional):
            Parameter for the loess plot.
            A higher value means larger time window and the resulting fitted line will
            be smoother. Disabled if set to 0.
            Defaults to 0.0.
        embed (str, optional):
            2D embedding used for visualization of time and cell type.
            The true key for the embedding is f"X_{embed}" in .obsm.
            Defaults to "umap".
        grid_size (tuple[int], optional):
            Grid size for plotting the genes.
            n_row * n_col >= len(genes). Defaults to (1, 1).
        sparsity_correction (bool, optional):
            Whether to sample cells non-uniformly across time and count values so
            that regions with sparser data point distributions will not be missed
            in gene plots due to sampling. Default to True.
        figure_path (str, optional):
            Path to save the figures.. Defaults to None.
        save (str, optional):
            Path + output file name to save the AnnData object to a .h5ad file.
            Defaults to None.

    kwargs:
        random_state (int):
            Random number seed. Default to 42.
        n_jobs (int):
            Number of CPU cores used for parallel computing in scvelo.tl.velocity_graph.
        save_format (str):
            Figure save_format. Default to 'png'.


    Returns:
        tuple

            - :class:`pandas.DataFrame`: Contains the dataset-wise performance metrics of all methods.
            - :class:`pandas.DataFrame`: Contains the performance metrics of each pair of ancestor and desendant cell types.

        Saves the figures to 'figure_path'.

        Notice that the two output dataframes will be None if 'compute_metrics' is set to False.
    """
    # set the random seed
    random_state = 42 if not 'random_state' in kwargs else kwargs['random_state']
    np.random.seed(random_state)
    # dpi
    set_dpi (dpi)
    if figure_path is not None:
        makedirs(figure_path, exist_ok=True)
    # Retrieve data
    if raw_count:
        U, S = adata.layers["unspliced"].A, adata.layers["spliced"].A
    else:
        U, S = adata.layers["Mu"], adata.layers["Ms"]
    X_embed = adata.obsm[f"X_{embed}"]
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
        #if spatial_graph_key in adata.obsp:
        #    n_spatial_neighbors = adata.uns['neighbors']['indices'].shape[1]
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
            # Integer-encoded cell type
            cell_labels_raw = adata.obs[cluster_key].to_numpy()
            cell_types_raw = np.unique(cell_labels_raw)
            cell_labels = np.zeros((len(cell_labels_raw)))
            for j in range(len(cell_types_raw)):
                cell_labels[cell_labels_raw == cell_types_raw[j]] = j

            if method == 'scVelo':
                t_i, Uhat_i, Shat_i = get_pred_scv_demo(adata, keys[i], genes, nplot)
                Yhat[method_] = np.concatenate((np.zeros((nplot)), np.ones((nplot))))
                V[method_] = adata.layers["velocity"][:, gene_indices]
            elif method == 'Vanilla VAE':
                t_i, Uhat_i, Shat_i = get_pred_vanilla_demo(adata, keys[i], genes, nplot)
                Yhat[method_] = None
                V[method_] = adata.layers[f"{keys[i]}_velocity"][:, gene_indices]
            elif method == 'Cycle VAE':
                t_i, Uhat_i, Shat_i = get_pred_cycle_demo(adata, keys[i], genes, nplot)
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
    save_format = kwargs["save_format"] if "save_format" in kwargs else "png"

    if 'cluster' in plot_type or "all" in plot_type:
        plot_cluster(adata.obsm[f"X_{embed}"],
                     adata.obs[cluster_key].to_numpy(),
                     save=(None if figure_path is None else 
                           f"{figure_path}/{test_id}_umap.{save_format}"))

    # Generate plots
    if "time" in plot_type or "all" in plot_type:
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
        plot_time_grid(T,
                       X_embed,
                       capture_time,
                       None,
                       dot_size=dot_size,
                       down_sample=min(10, max(1, adata.n_obs//5000)),
                       grid_size=(n_row, n_col),
                       color_map=time_colormap,
                       save=(None if figure_path is None else
                             f"{figure_path}/{test_id}_time.{save_format}"))

    if len(genes) == 0:
        return

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
                        path=figure_path,
                        figname=test_id,
                        save_format=save_format)

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
                      frac=frac,
                      down_sample=min(20, max(1, adata.n_obs//5000)),
                      sparsity_correction=sparsity_correction,
                      path=figure_path,
                      figname=test_id,
                      save_format=save_format)

    if 'stream' in plot_type or 'all' in plot_type:
        try:
            from scvelo.tl import velocity_graph
            from scvelo.pl import velocity_embedding_stream
            colors = get_colors(len(cell_types_raw))
            if 'stream_legend_loc' in kwargs:
                stream_legend_loc = kwargs['stream_legend_loc']
            else:
                stream_legend_loc = 'on data' if len(colors) <= 10 else 'right margin'
            legend_fontsize = (kwargs['legend_fontsize'] if 'legend_fontsize' in kwargs else
                               np.clip(15 - np.clip(len(colors)-10, 0, None), 8, None))
            for i, vkey in enumerate(vkeys):
                if methods[i] in ['scVelo', 'UniTVelo', 'DeepVelo']:
                    gene_subset = adata.var_names[adata.var['velocity_genes'].to_numpy()]
                else:
                    gene_subset = adata.var_names[~np.isnan(adata.layers[vkey][0])]
                xkey = 'Ms' if 'xkey' not in kwargs else kwargs['xkey']
                velocity_graph(adata, vkey=vkey, xkey=xkey, gene_subset=gene_subset, n_jobs=get_n_cpu(adata.n_obs))
                velocity_embedding_stream(adata,
                                          basis=embed,
                                          vkey=vkey,
                                          color=cluster_key,
                                          title="",
                                          figsize=stream_figsize,
                                          palette=colors,
                                          size=dot_size,
                                          legend_loc=stream_legend_loc,
                                          legend_fontsize=legend_fontsize,
                                          cutoff_perc=0.0,
                                          dpi=dpi,
                                          show=True,
                                          save=(None if figure_path is None else
                                                f'{figure_path}/{test_id}_{keys[i]}.png'))
                if 'TopoVelo' in methods[i]:
                    # adata.uns[f"{vkey}_params"]["embeddings"].append(f'{keys[i]}_xy')
                    adata.obsm[f"{vkey}_dec_{embed}"] = adata.obsm[f"{keys[i]}_velocity_{keys[i]}_xy"]
                    velocity_embedding_stream(adata,
                                              basis=embed,
                                              vkey=f"{vkey}_dec",
                                              recompute=False,
                                              color=cluster_key,
                                              title="",
                                              figsize=stream_figsize,
                                              palette=colors,
                                              size=dot_size,
                                              legend_loc=stream_legend_loc,
                                              legend_fontsize=legend_fontsize,
                                              cutoff_perc=0.0,
                                              dpi=dpi,
                                              show=True,
                                              save=(None if figure_path is None else
                                                    f'{figure_path}/{test_id}_{keys[i]}_true_velocity.png'))
        except ImportError:
            print('Please install scVelo in order to generate stream plots')
            pass
    if save is not None:
        adata.write_h5ad(save)
    if compute_metrics:
        if figure_path is not None:
            stats_df.to_csv(f"{figure_path}/metrics_{test_id}.csv", sep='\t')
        return stats_df, stats_type_df, multi_stats_df, multi_stats_type_df

    return None, None, None, None
