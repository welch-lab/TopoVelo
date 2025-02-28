import logging
from multiprocessing import cpu_count
from typing  import Any, Dict, Iterable, Optional, Union
from anndata import AnnData
import numpy as np
import scanpy as sc
import scvelo as scv
from scvelo.tl import velocity_graph, velocity, velocity_embedding
from scvelo.pl import velocity_embedding_stream
from ..scvelo_preprocessing.neighbors import neighbors
from .plot_config import PlotConfig
from ..model import rna_velocity_vae
from ..plotting import get_colors, compute_figsize


def get_n_cpu(n_cell):
    # used for scVelo parallel jobs
    return int(min(cpu_count(), max(1, n_cell/2000)))


def _config_stream_plot(
    adata: AnnData,
    cell_types_raw: Iterable[str],
    stream_plot_config: Dict[str, Any] = {}
) -> PlotConfig:
    # Configure plotting
    plot_config = PlotConfig('stream')
    plot_config.set_multiple(stream_plot_config)
    if plot_config.get('palette') is None:
        palette = get_colors(len(cell_types_raw))
        plot_config.set('palette', palette)
    return plot_config


def knn_smooth_1d(vals, graph, perc=[2, 98]):
    v_max_to = np.percentile(vals, perc[1])
    v_min_to = np.percentile(vals, perc[0])
    vals_sm = graph.dot(vals)
    v_max_from = vals_sm.max()
    v_min_from = vals_sm.min()
    
    return (vals_sm - v_min_from)/(v_max_from - v_min_from)*(v_max_to - v_min_to) + v_min_to


def knn_smooth_2d(vals, graph, perc=[2, 98]):
    return np.stack([knn_smooth_1d(vals[:, i], graph, perc) for i in range(vals.shape[1])], axis=1)


def knn_smooth(vals, graph, perc=[2, 98]):
    if vals.ndim == 1:
        return knn_smooth_1d(vals, graph, perc)
    return knn_smooth_2d(vals, graph, perc)


def rna_velocity(adata: AnnData, method: str, key: str, vkey: str):
    if vkey in adata.layers:
        return True
    if 'TopoVelo' in method or 'VeloVAE' in method:
        full_vb = 'Rate Prior' in method
        rna_velocity_vae(
            adata,
            key,
            full_vb=full_vb
        )
    else:
        velocity(
            adata,
            vkey=vkey,
            mode='dynamical'
        )
    return True


def velocity_stream(
    adata: AnnData,
    method: str,
    key: str,
    vkey: str,
    cell_types_raw: Iterable[str],
    basis: str = 'spatial',
    cluster_key: str = 'clusters',
    n_spatial_neighbors: int = 30,
    dpi: int = 100,
    save: Optional[str] = None,
    stream_plot_config: Dict[str, Any] = {},
    **kwargs
):
    """
    Generates a velocity stream plot for RNA velocity analysis using specified methods.

    This function calculates RNA velocity, constructs a velocity graph, and generates
    a stream plot of cell velocities on a specified basis, permitting a detailed visualization
    of cell transitions and dynamics.

    Parameters:
        adata (AnnData): Annotated data matrix with single-cell RNA-seq data.
        method (str): The method to use for RNA velocity calculation.
            Options include 'scVelo', 'UniTVelo', 'DeepVelo', etc.
        key (str): The key in `adata.layers` representing the data to use for velocity calculations.
        vkey (str): The key in `adata.layers` where computed velocities should be stored.
        cell_types_raw (Iterable[str]): List of cell types to be included in the analysis.
        basis (str, optional): The embedding basis ('spatial' by default) to project velocities on.
        cluster_key (str, optional): The key in `adata.obs` indicating the cell clustering. Defaults to 'clusters'.
        dpi (int, optional): Dots per inch for saved figures. Defaults to 100.
        save (Optional[str], optional): The filename to save the plot; if `None`, the plot is not saved. Defaults to None.
        stream_plot_config (Dict[str, Any], optional): Configuration dictionary for stream plot aesthetics.
        **kwargs: Additional parameters for configuration, such as the number of jobs (`n_jobs`)
            for parallel processing and custom `xkey` for computing the velocity graph.

    Returns:
        None: The function generates a plot and optionally saves it to the specified location.
    """
    logger = logging.getLogger(__name__)
    plot_config = _config_stream_plot(adata, cell_types_raw, stream_plot_config)
    figsize = (plot_config.get('width'), plot_config.get('height'))

    # Calculate RNA velocity
    rna_velocity(adata, method, key, vkey)

    # Set velocity genes
    if method in ['scVelo', 'UniTVelo', 'DeepVelo']:
        gene_subset = adata.var_names[adata.var['velocity_genes'].to_numpy()]
    else:
        gene_subset = adata.var_names[~np.isnan(adata.layers[vkey][0])]
    
    
    # recompute the spatial KNN graph
    spatial_velocity_graph = basis.lower() not in ['umap', 'tsne', 'pca']
    if spatial_velocity_graph:
        print(f'Computing a spatial graph using KNN on {basis} with k={n_spatial_neighbors}')
        figsize = compute_figsize(
            adata.obsm[f'X_{basis}'],
            plot_config.get('width'),
            plot_config.get('height'),
            plot_config.get('real_aspect_ratio'),
            plot_config.get('fix')
        )
        if 'connectivities' in adata.obsp or 'neighbors' in adata.uns:
            logger.warning(f'Overwriting the original KNN graph! (.uns, .obsp)')
            connectivities = adata.obsp['connectivities']
            distances = adata.obsp['distances']
            del adata.obsp['connectivities'], adata.obsp['distances']
            nbs_info = None
            if 'neighbors' in adata.uns:
                nbs_info = adata.uns['neighbors'].copy()
                del adata.uns['neighbors']
                assert nbs_info is not None
        neighbors(adata, n_neighbors=n_spatial_neighbors, use_rep=f'X_{basis}')

    # Velocity graph
    xkey = 'Ms' if 'xkey' not in kwargs else kwargs['xkey']
    velocity_graph(
        adata,
        vkey=vkey,
        xkey=xkey,
        gene_subset=gene_subset,
        n_jobs=(kwargs['n_jobs']
                if 'n_jobs' in kwargs
                else get_n_cpu(adata.n_obs))
    )
    
    # Use radius for spatial graph
    if 'spatial_graph_params' in adata.uns and spatial_velocity_graph:
        if 'radius' in adata.uns['spatial_graph_params']:
            radius = adata.uns['spatial_graph_params']['radius']
            if radius is not None:
                adata.uns[f'{vkey}_graph'] = adata.uns[f'{vkey}_graph']\
                    .multiply(adata.obsp['distances'] < radius)
                adata.uns[f'{vkey}_graph_neg'] = adata.uns[f'{vkey}_graph_neg']\
                    .multiply(adata.obsp['distances'] < radius)
    
    # Velocity embedding
    velocity_embedding(
        adata,
        basis=basis,
        vkey=vkey
    )
    
    velocity_embedding_stream(
        adata,
        basis=basis,
        vkey=vkey,
        color=cluster_key,
        title="",
        figsize=figsize,
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
        save=save
    )
    
    # Recover the original KNN graph
    if spatial_velocity_graph:
        adata.obsp['connectivities'] = connectivities
        adata.obsp['distances'] = distances
        if nbs_info is not None:
            print('Recovering the original KNN graph from .uns')
            adata.uns['neighbors'] = nbs_info


def velocity_stream_on_pred_xy(
    adata: AnnData,
    method: str,
    key: str,
    vkey: str,
    cell_types_raw: Iterable[str],
    cluster_key: str = 'clusters',
    dpi: int = 100,
    save: Optional[str] = None,
    stream_plot_config: Dict[str, Any] = {},
    **kwargs
):
    """
    DEPRECATED: Generates a velocity stream plot using predicted X and Y coordinates for specified cell types.

    This function checks for the existence of velocity embeddings in `adata.obsm`, clips velocity to remove outliers,
    smooths it using KNN, and updates embeddings with velocity parameters. It then creates a stream plot with
    optional configurations for visualization.

    Parameters:
        adata (AnnData): Annotated data matrix.
        method (str): The method name for logging purposes.
        key (str): Key used to identify specific parameters in `adata`.
        vkey (str): Key prefix for velocity data in `adata.obsm`.
        cell_types_raw (Iterable[str]): List of cell types to include in the plot.
        cluster_key (str, optional): Key for clustering data in `adata.obs`. Defaults to 'clusters'.
        dpi (int, optional): The resolution of the plot. Defaults to 100.
        save (Optional[str], optional): If provided, the path to save the plot. Defaults to None.
        stream_plot_config (Dict[str, Any], optional): Additional stream plot configuration. Defaults to {}.
        **kwargs: Additional keyword arguments.

    Returns:
        None: The function is used for its side effects of plotting.

    Warnings:
        This function is deprecated and may be removed or replaced in future versions.
    """
    
    logger = logging.getLogger(__name__)
    plot_config = _config_stream_plot(adata, cell_types_raw, stream_plot_config)

    # Look for velocity embedding on predicted X and Y
    if f"{vkey}_{key}_xy" not in adata.obsm:
        logger.warning(f"{method}: Velocity embedding {vkey}_{key}_xy not found in adata.obsm. Skipping stream plot.")
        return

   # Clip the velocity to remove outliers
    v = adata.obsm[f"{vkey}_{key}_xy"]
    q1, q3 = np.quantile(v, 0.75, 0), np.quantile(v, 0.25, 0)
    v = np.stack(
        [
            np.clip(v[:, 0], q3[0]-1.5*(q1[0]-q3[0]), q1[0]+1.5*(q1[0]-q3[0])),
            np.clip(v[:, 1], q3[1]-1.5*(q1[0]-q3[0]), q1[1]+1.5*(q1[1]-q3[1]))
        ],
        1
    )
    v = knn_smooth(v, adata.obsp["connectivities"])
    adata.obsm[f"{vkey}_{key}_xy"] = v
    # Use predicted coordinates
    adata.uns[f"{key}_velocity_params"]["embeddings"] = f"{key}_xy"
    velocity_embedding_stream(
        adata,
        basis=f"{key}_xy",
        vkey=vkey,
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
        save=save
    )