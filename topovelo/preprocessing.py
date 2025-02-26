import scanpy
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors, BallTree
from scanpy.pp import pca
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix, eye
import pandas as pd
from .scvelo_preprocessing import *
from .model.scvelo_util import leastsq_NxN, R_squared


def count_peak_expression(adata, cluster_key="clusters"):
    # Count the number of genes with peak expression in each cell type.
    def encodeType(cell_types_raw):
        # Use integer to encode the cell types.
        # Each cell type has one unique integer label.
        # Map cell types to integers
        label_dic = {}
        label_dic_rev = {}
        for i, type_ in enumerate(cell_types_raw):
            label_dic[type_] = i
            label_dic_rev[i] = type_

        return label_dic, label_dic_rev
    cell_labels = adata.obs[cluster_key]
    cell_types = np.unique(cell_labels)
    label_dic, label_dic_rev = encodeType(cell_types)
    cell_labels = np.array([label_dic[x] for x in cell_labels])
    n_type = len(cell_types)

    X = np.array(adata.layers["spliced"].A+adata.layers["unspliced"].A)
    peak_expression = np.stack([np.quantile(X[cell_labels == j], 0.9, 0) for j in range(n_type)])
    peak_type = np.argmax(peak_expression, 0)
    peak_hist = np.array([np.sum(peak_type == i) for i in range(n_type)])  # gene count
    peak_val_hist = [peak_expression[:, peak_type == i][i] for i in range(n_type)]  # peak expression
    peak_gene = [np.where(peak_type == i)[0] for i in range(n_type)]  # peak gene index list

    out_peak_count = {}
    out_peak_expr = {}
    out_peak_gene = {}
    for i in range(n_type):
        out_peak_count[label_dic_rev[i]] = peak_hist[i]
        out_peak_expr[label_dic_rev[i]] = peak_val_hist[i]
        out_peak_gene[label_dic_rev[i]] = np.array(peak_gene[i])

    return out_peak_count, out_peak_expr, out_peak_gene


def balanced_gene_selection(adata, n_gene, cluster_key):
    # select the same number of genes for each cell type.
    if n_gene > adata.n_vars:
        return
    cell_labels = adata.obs[cluster_key].to_numpy()
    cell_types = np.unique(cell_labels)
    n_type = len(cell_types)
    count, peak_expr, peak_gene = count_peak_expression(adata, cluster_key)
    length_list = [len(peak_gene[x]) for x in cell_types]
    order_length = np.argsort(length_list)
    k = 0
    s = 0
    while s+length_list[order_length[k]]*(n_type-k) < n_gene:
        s = s+length_list[order_length[k]]
        k = k+1

    gene_list = []
    # Cell types with all peak genes picked
    for i in range(k):
        gene_list.extend(peak_gene[cell_types[order_length[i]]])
    n_gene_per_type = (n_gene - s)//(n_type-k)
    for i in range(k, n_type-1):
        gene_idx_order = np.flip(np.argsort(peak_expr[cell_types[order_length[i]]]))
        gene_list.extend(peak_gene[cell_types[order_length[i]]][gene_idx_order[:n_gene_per_type]])
    if k < n_type-1:
        gene_idx_order = np.flip(np.argsort(peak_expr[cell_types[order_length[-1]]]))
        n_res = n_gene - s - n_gene_per_type*(n_type-k-1)
        gene_list.extend(peak_gene[cell_types[order_length[-1]]][gene_idx_order[:n_res]])
    gene_subsel = np.zeros((adata.n_vars), dtype=bool)
    gene_subsel[np.array(gene_list).astype(int)] = True
    adata._inplace_subset_var(gene_subsel)
    return


def filt_gene_sparsity(adata, thred_u=0.99, thred_s=0.99):
    N, G = adata.n_obs, adata.n_vars
    sparsity_u = np.zeros((G))
    sparsity_s = np.zeros((G))
    for i in tqdm(range(G)):
        sparsity_u[i] = np.sum(adata.layers["unspliced"][:, i].A.squeeze() == 0)/N
        sparsity_s[i] = np.sum(adata.layers["spliced"][:, i].A.squeeze() == 0)/N
    gene_subset = (sparsity_u < thred_u) & (sparsity_s < thred_s)
    print(f"Kept {np.sum(gene_subset)} genes after sparsity filtering")
    adata._inplace_subset_var(gene_subset)


def rank_gene_selection(adata, cluster_key, **kwargs):
    if "cell_types" not in kwargs:
        cell_types = np.unique(adata.obs[cluster_key].to_numpy())
    else:
        cell_types = kwargs["cell_types"]
    use_raw = kwargs["use_raw"] if "use_raw" in kwargs else False
    layer = kwargs["layer"] if "layer" in kwargs else None
    scanpy.tl.rank_genes_groups(adata,
                                groupby=cluster_key,
                                use_raw=use_raw,
                                layer=layer,
                                method='wilcoxon',
                                pts=True)
    min_in_group_fraction = kwargs["min_in_group_fraction"] if "min_in_group_fraction" in kwargs else 0.1
    min_fold_change = kwargs["min_fold_change"] if "min_fold_change" in kwargs else 1.5
    max_out_group_fraction = kwargs["max_out_group_fraction"] if "max_out_group_fraction" in kwargs else 0.5
    compare_abs = kwargs["compare_abs"] if "compare_abs" in kwargs else False
    scanpy.tl.filter_rank_genes_groups(adata,
                                       groupby=cluster_key,
                                       use_raw=False,
                                       min_in_group_fraction=min_in_group_fraction,
                                       min_fold_change=min_fold_change,
                                       max_out_group_fraction=max_out_group_fraction,
                                       compare_abs=compare_abs)
    gene_subset = np.zeros((adata.n_vars), dtype=bool)
    # Build a gene index mapping
    gene_dic = {}
    for i, x in enumerate(adata.var_names):
        gene_dic[x] = i
    gene_set = set()
    for ctype in cell_types:
        names = adata.uns['rank_genes_groups_filtered']['names'][ctype].astype(str)
        adata.uns['rank_genes_groups_filtered']['names'][ctype] = names
        gene_set = gene_set.union(set(names))
    for gene in gene_set:
        if gene != 'nan':
            gene_subset[gene_dic[gene]] = True
    print(f"Picked {len(gene_set)-1} genes")
    adata._inplace_subset_var(gene_subset)
    del adata.uns['rank_genes_groups']['pts']
    del adata.uns['rank_genes_groups']['pts_rest']
    del adata.uns['rank_genes_groups_filtered']['pts']
    del adata.uns['rank_genes_groups_filtered']['pts_rest']


def preprocess(adata,
               n_gene=1000,
               cluster_key="clusters",
               spatial_smoothing=False,
               spatial_key=None,
               tkey=None,
               selection_method="scv",
               min_count_per_cell=None,
               min_genes_expressed=None,
               min_shared_counts=10,
               min_shared_cells=10,
               min_counts_s=None,
               min_cells_s=None,
               max_counts_s=None,
               max_cells_s=None,
               min_counts_u=None,
               min_cells_u=None,
               max_counts_u=None,
               max_cells_u=None,
               npc=30,
               n_neighbors=30,
               n_spatial_neighbors=50,
               use_highly_variable=True,
               genes_retain=None,
               perform_clustering=False,
               resolution=1.0,
               compute_umap=False,
               umap_min_dist=0.25,
               keep_raw=True,
               **kwargs):
    """Preprocess the data.

    Args:
        adata (AnnData): Annotated data matrix.
        n_gene (int): Number of genes to keep. If n_gene < 0, all genes are kept.
        cluster_key (str): Key for cell type annotation.
        spatial_smoothing (bool): Whether to perform spatial smoothing.
        spatial_key (str): Key for spatial coordinates.
            Effective only when spatial_smoothing is True.
        tkey (str): Key for capture time.
        selection_method (str): Method for gene selection. "scv" for scVelo, "balanced" for balanced gene selection,
            "wilcoxon" for Wilcoxon rank-sum test.
        min_count_per_cell (int): Minimal number of counts per cell.
        min_genes_expressed (int): Minimal number of genes expressed per cell.
        min_shared_counts (int): Minimal number of shared counts for each gene.
        min_shared_cells (int): Minimal number of shared cells for each gene.
        min_counts_s (int): Minimal number of counts for spliced mRNA.
        min_cells_s (int): Minimal number of cells for spliced mRNA.
        max_counts_s (int): Maximal number of counts for spliced mRNA.
        max_cells_s (int): Maximal number of cells for spliced mRNA.
        min_counts_u (int): Minimal number of counts for unspliced mRNA.
        min_cells_u (int): Minimal number of cells for unspliced mRNA.
        max_counts_u (int): Maximal number of counts for unspliced mRNA.
        max_cells_u (int): Maximal number of cells for unspliced mRNA.
        npc (int): Number of principal components.
        n_neighbors (int): Number of neighbors for KNN averaging.
        n_spatial_neighbors (int): Number of neighbors for spatial KNN averaging.
        use_highly_variable (bool): Whether to use highly variable genes for PCA calculation.
        genes_retain (list): List of genes to keep.
        perform_clustering (bool): Whether to perform clustering.
        resolution (float): Resolution for clustering.
        compute_umap (bool): Whether to compute UMAP coordinates.
        umap_min_dist (float): Minimal distance for UMAP.
        keep_raw (bool): Whether to keep the raw count data.
        **kwargs: Additional arguments for gene selection.
    """
    # Preprocessing
    # 1. Cell, Gene filtering and data normalization
    n_cell = adata.n_obs
    if min_count_per_cell is not None:
        scanpy.pp.filter_cells(adata, min_counts=min_count_per_cell)
    if min_genes_expressed is not None:
        scanpy.pp.filter_cells(adata, min_genes=min_genes_expressed)
    if n_cell - adata.n_obs > 0:
        print(f"Filtered out {n_cell - adata.n_obs} cells with low counts.")

    if keep_raw:
        gene_names_all = np.array(adata.var_names)
        U_raw = adata.layers["unspliced"]
        S_raw = adata.layers["spliced"]

    if n_gene > 0 and n_gene < adata.n_vars:
        flavor = kwargs["flavor"] if "flavor" in kwargs else "seurat"
        if selection_method == "balanced":
            print("Balanced gene selection.")
            filter_genes(adata,
                         min_counts=min_counts_s,
                         min_cells=min_cells_s,
                         max_counts=max_counts_s,
                         max_cells=max_cells_s,
                         min_counts_u=min_counts_u,
                         min_cells_u=min_cells_u,
                         max_counts_u=max_counts_u,
                         max_cells_u=max_cells_u,
                         retain_genes=genes_retain)
            balanced_gene_selection(adata, n_gene, cluster_key)
            normalize_per_cell(adata)
            log1p(adata)
        elif selection_method == "wilcoxon":
            print("Marker gene selection using Wilcoxon test.")
            filter_genes(adata,
                         min_counts=min_counts_s,
                         min_cells=min_cells_s,
                         max_counts=max_counts_s,
                         max_cells=max_cells_s,
                         min_counts_u=min_counts_u,
                         min_cells_u=min_cells_u,
                         max_counts_u=max_counts_u,
                         max_cells_u=max_cells_u,
                         retain_genes=genes_retain)
            normalize_per_cell(adata)
            log1p(adata)
            if adata.n_vars > n_gene:
                filter_genes_dispersion(adata,
                                        n_top_genes=n_gene,
                                        retain_genes=genes_retain,
                                        flavor=flavor)
            rank_gene_selection(adata, cluster_key, **kwargs)
        else:
            filter_and_normalize(adata,
                                 min_shared_counts=min_shared_counts,
                                 min_shared_cells=min_shared_cells,
                                 min_counts=min_counts_s,
                                 min_counts_u=min_counts_u,
                                 n_top_genes=n_gene,
                                 retain_genes=genes_retain,
                                 flavor=flavor)
    elif genes_retain is not None:
        gene_subset = np.zeros(adata.n_vars, dtype=bool)
        for i in range(len(genes_retain)):
            indices = np.where(adata.var_names == genes_retain[i])[0]
            if len(indices) == 0:
                continue
            gene_subset[indices[0]] = True
        adata._inplace_subset_var(gene_subset)
        normalize_per_cell(adata)
        log1p(adata)
        adata.var['highly_variable'] = np.array([True]*adata.n_vars)
    else:
        normalize_per_cell(adata)
        log1p(adata)
        adata.var['highly_variable'] = np.array([True]*adata.n_vars)

    # 2. KNN Averaging
    # remove_duplicate_cells(adata)
    if 'X_pca' not in adata.obsm:
        print("Run PCA.")
        pca(adata, n_comps=npc)
    if spatial_smoothing:
        print('Spatial KNN smoothing.')
        moments(adata,
                n_pcs=npc,
                use_highly_variable=use_highly_variable,
                n_neighbors=n_spatial_neighbors,
                method='sklearn',
                use_rep=spatial_key)
    else:
        moments(adata,
                n_pcs=npc,
                use_highly_variable=use_highly_variable,
                n_neighbors=n_neighbors)

    if keep_raw:
        print("Keep raw unspliced/spliced count data.")
        gene_idx = np.array([np.where(gene_names_all == x)[0][0] for x in adata.var_names])
        adata.layers["unspliced"] = U_raw[:, gene_idx].astype(int)
        adata.layers["spliced"] = S_raw[:, gene_idx].astype(int)

    # 3. Obtain cell clusters
    if perform_clustering:
        scanpy.tl.leiden(adata, key_added='clusters', resolution=resolution)

    # 4. Obtain Capture Time (If available)
    if tkey is not None:
        capture_time = adata.obs[tkey].to_numpy()
        if isinstance(capture_time[0], str):
            tprior = np.array([float(x[1:]) for x in capture_time])
        else:
            tprior = capture_time
        tprior = tprior - tprior.min() + 0.01
        adata.obs["tprior"] = tprior

    # 5. Compute Umap coordinates for visulization
    if compute_umap:
        print("Computing UMAP coordinates.")
        if "X_umap" in adata.obsm:
            print("Warning: Overwriting existing UMAP coordinates.")
        scanpy.tl.umap(adata, min_dist=umap_min_dist)


def get_spatialde(adata, min_counts=1, min_counts_u=1):
    """Use spatialDE to identify spatially variable genes.

    Args:
        adata (class:`AnnData`): Annotated data matrix.
        min_counts (int, optional): Minimum total count for gene filtering. Defaults to 1.
        min_counts_u (int, optional): Minimum total unspliced read count for gene filtering.. Defaults to 1.

    Raises:
        ImportError: Please install NaiveDE and SpatialDE.

    Returns:
        array-like: List of spatially variable genes sorted by q-value.
    """
    try:
        import NaiveDE as nde
        import SpatialDE as spd
    except ImportError:
        raise ImportError("Please install NaiveDE and SpatialDE.")
        return
    filter_genes(adata, min_counts=min_counts, min_counts_u=min_counts_u)
    adata.var_names_make_unique()
    counts = pd.DataFrame(adata.X.todense(), columns=adata.var_names, index=adata.obs_names)
    if adata.obsm['X_spatial'].shape[1] == 2:
        coord = pd.DataFrame(adata.obsm['X_spatial'], columns=['x_coord', 'y_coord'], index=adata.obs_names)
    elif adata.obsm['X_spatial'].shape[1] == 3:
        coord = pd.DataFrame(adata.obsm['X_spatial'], columns=['x_coord', 'y_coord', 'z_coord'], index=adata.obs_names)
    else:
        coord = pd.DataFrame(adata.obsm['X_spatial'][:, :2], columns=['x_coord', 'y_coord'], index=adata.obs_names)
    norm_expr = nde.stabilize(counts.T).T
    adata.obs['total_counts'] = adata.X.sum(1).A1
    coord['total_counts'] = adata.obs['total_counts'].to_numpy()
    resid_expr = nde.regress_out(coord, norm_expr.T, 'np.log(total_counts)').T
    results = spd.run(coord, resid_expr)
    
    results.index = results["g"].to_numpy().astype(str)
    results = results.drop_duplicates("g")
    
    adata.var = pd.concat([adata.var, results.loc[adata.var.index.values, :]], axis=1)
    gene_sort = results.sort_values("qval").index.to_numpy().astype(str)
    return gene_sort


def preprocess_spatialde(adata,
                         n_gene,
                         min_counts=1,
                         min_counts_u=1,
                         n_pcs=30,
                         n_neighbors=30,
                         save=None):
    """Preprocess the data using spatialDE.

    Args:
        adata (AnnData): Annotated data matrix.
        n_gene (int): Number of genes to keep.
        min_counts (int): Minimal number of counts.
        min_counts_u (int): Minimal number of unspliced counts.
        n_pcs (int): Number of principal components.
        n_neighbors (int): Number of neighbors for KNN averaging.
        save (str): File path to save the processed data.
    
    Note:
        This function preprocesses the data using spatialDE, which is a method for spatial transcriptomics.
    """
    gene_sort = get_spatialde(adata, min_counts, min_counts_u)
    
    normalize_per_cell(adata)
    adata._inplace_subset_var(list(gene_sort[:n_gene]))
    log1p(adata)
    moments(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
    
    if save is None:
        return
    adata.write_h5ad(save)


def build_spatial_graph(adata,
                        spatial_key,
                        graph_key='spatial_graph',
                        n_neighbors=16,
                        method='KNN',
                        dist_cutoff=None,
                        radius=None):
    """Build spatial graph.

    Args:
        adata (AnnData): Annotated data matrix.
        spatial_key (str): Key for spatial coordinates.
        graph_key (str): Key for saving the spatial graph.
        n_neighbors (int): Number of neighbors for KNN averaging.
            Defaults to 16.
        method (str): Method for building spatial graph.
            "KNN" for KNN graph, "Delaunay" for Delaunay triangulation.
        dist_cutoff (float): Distance cutoff.
            Remove edges with distance larger than the percentage with a value of dist_cutoff.
            Defaults to None.
        radius (float): Radius of the neighborhood.
            When method is 'KNN' or 'Delaunay', we omit all neighbors with distance larger than radius.
            When method is 'BallTree', we use radius to build the graph.
            Defaults to None.
    """
    if method == 'KNN':
        x_pos = adata.obsm[spatial_key][:, :2]
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(x_pos)
        
        if isinstance(dist_cutoff, float) or isinstance(dist_cutoff, int):
            dist = nn.kneighbors_graph(mode='distance')
            nonzero_dist = dist.data
            thred = np.percentile(nonzero_dist, dist_cutoff)
            if isinstance(radius, float) or isinstance(radius, int):
                thred = min(thred, radius)
            adata.obsp[graph_key] = nn.kneighbors_graph().multiply(dist < thred)
            
        elif dist_cutoff == 'auto':
            dist = nn.kneighbors_graph(mode='distance')
            nonzero_dist = dist.data
            q1, q3 = np.quantile(nonzero_dist, 0.25), np.quantile(nonzero_dist, 0.75)
            thred = q3 + 1.5 * (q3 - q1)
            if isinstance(radius, float) or isinstance(radius, int):
                thred = min(thred, radius)
            adata.obsp[graph_key] = nn.kneighbors_graph().multiply(dist < thred)
        else:
            if isinstance(radius, float) or isinstance(radius, int):
                dist = nn.kneighbors_graph(mode='distance')
                nonzero_dist = dist.data
                thred = radius
                adata.obsp[graph_key] = nn.kneighbors_graph().multiply(dist < radius)
            else:
                adata.obsp[graph_key] = nn.kneighbors_graph()
                thred = np.inf

    elif method == 'Delaunay':
        x_pos = adata.obsm[spatial_key][:, :2]
        tri = Delaunay(x_pos)
        adata.obsp[graph_key] = csr_matrix((np.ones((len(tri.vertex_neighbor_vertices[1]))),
                                            tri.vertex_neighbor_vertices[1],
                                            tri.vertex_neighbor_vertices[0]))
        if isinstance(dist_cutoff, float) or isinstance(dist_cutoff, int):
            idx_1, idx_2 = adata.obsp[graph_key].nonzero()
            dist = np.linalg.norm(x_pos[idx_1] - x_pos[idx_2], axis=1)
            thred = np.percentile(dist, dist_cutoff)
            mask = (dist < thred).astype(int)
            mtx = csr_matrix((mask, (idx_1, idx_2)))
            adata.obsp[graph_key] = adata.obsp[graph_key].multiply(mtx)
        elif dist_cutoff == 'auto':
            idx_1, idx_2 = adata.obsp[graph_key].nonzero()
            dist = np.linalg.norm(x_pos[idx_1] - x_pos[idx_2], axis=1)
            q1, q3 = np.quantile(dist, 0.25), np.quantile(dist, 0.75)
            thred = q3 + 1.5 * (q3 - q1)
            mask = (dist < thred).astype(int)
            mtx = csr_matrix((mask, (idx_1, idx_2)))
            adata.obsp[graph_key] = adata.obsp[graph_key].multiply(mtx)
    elif method == 'BallTree':
        x_pos = adata.obsm[spatial_key]
        tree = BallTree(x_pos)
        assert radius is not None, "Please provide a radius for BallTree."
        nbs = tree.query_radius(x_pos, r=radius)
        row_idx = np.concatenate([np.repeat(i, len(x)) for i, x in enumerate(nbs)])
        col_idx = np.concatenate(nbs)
        adata.obsp[graph_key] = csr_matrix((np.ones(len(row_idx)), (row_idx, col_idx)), shape=(len(x_pos), len(x_pos)))
        print(f'Ball Tree: average number of neighbors: {len(row_idx) / len(nbs):.1f}')
        
    # Record the parameters for building the spatial graph
    adata.uns['spatial_graph_params'] = {}
    adata.uns['spatial_graph_params']['method'] = method
    adata.uns['spatial_graph_params']['n_neighbors'] = n_neighbors
    adata.uns['spatial_graph_params']['dist_cutoff'] = dist_cutoff
    adata.uns['spatial_graph_params']['radius'] = radius


def pick_ref_batch(adata, batch_key, percent=95, min_r2=0.01, eps=1e-3):
    """Pick a reference batch for batch-corrected TopoVelo model.
    We determine the reference batch by using the steady-state model to
    fit u, s for each gene and then pick the batch with the most number
    of velocity genes.
    This step should be performed after the data is preprocessed.

    Args:
        adata (AnnData): Annotated data matrix.
        batch_key (str): Key for batch annotation.
        percent (float): Percentile for fitting steady-state model.
        min_r2 (float): Minimal R2 for fitting steady-state model.
        eps (float): Minimal value for gamma.
    """
    batch_labels = adata.obs[batch_key].to_numpy()
    batch_names = np.unique(batch_labels)
    try:
        n_vel_genes = {}
        for batch in batch_names:
            batch_idx = np.where(batch_labels == batch)[0]
            u, s = adata.layers["Ms"][batch_idx, :], adata.layers["Mu"][batch_idx, :]
            offset, gamma = leastsq_NxN(s, u, False, perc=[100-percent, percent])
            gamma = np.clip(gamma, eps, None)
            residual = u-gamma*s-offset
            r2 = R_squared(residual, total=u-u.mean(0))
            velocity_genes = (r2 > min_r2) & (r2 < 0.95) & (gamma > 0.01) & (np.max(s > 0, 0) > 0) & (np.max(u > 0, 0) > 0)
            n_vel_genes[batch] = np.sum(velocity_genes)
        ref_batch = None
        max_count = -1
        for batch, count in n_vel_genes.items():
            if count > max_count:
                ref_batch = batch
    except KeyError:
        print("Warning: unspliced/spliced count are not KNN smoothed. Please run preprocess first.")
        ref_batch = batch_names[0]
    return ref_batch