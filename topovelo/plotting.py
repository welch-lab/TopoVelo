import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import igraph as ig
import pandas as pd
import pynndescent
from seaborn import kdeplot
from sklearn.neighbors import NearestNeighbors, BallTree
from scipy.stats import norm
from scipy.ndimage import gaussian_filter


#######################################################################################
# Default colors and markers for plotting
#######################################################################################
TAB10 = list(plt.get_cmap("tab10").colors)
TAB20 = list(plt.get_cmap("tab20").colors)
TAB20B = list(plt.get_cmap("tab20b").colors)
TAB20C = list(plt.get_cmap("tab20c").colors)
RAINBOW = [plt.cm.rainbow(i) for i in range(256)]
CATEGORICAL = ["#4f8c9d", "#fa756b", "#20b465", "#ce2bbc", "#51f310", "#660081",
               "#c9dd87", "#3f3369", "#f6bb86", "#0c4152", "#edb4ec", "#0b5313",
               "#b0579a", "#f4d403", "#7d0af6", "#698e4e", "#fb2076", "#65e6f9",
               "#74171f", "#b7c8e2", "#473a0a", "#7363e7", "#9f6c3b", "#1f84ec"]

markers = ["o", "x", "s", "v", "+", "d", "1", "*", "^", "p", "h", "8", "1", "2", "|"]
# change dpi via the function set_dpi()
DPI = 300
MAX_WIDTH = 18 / 2.54
MAX_HEIGHT = 21 / 2.54
WIDTH = 6 / 2.54
HEIGHT = 4 / 2.54
FONTSIZE = 7
LABEL_FONTSIZE = 7
LEGEND_FONTSIZE = 5
TICK_FONTSIZE = 5
COLORBAR_FONTSIZE = 7
COLORBAR_TICK_FONTSIZE = 5
TITLE_FONTSIZE = 7
MARKERSIZE = 3
LINEWIDTH = 1.0
MARKERSCALE = 1.0


def set_dpi(dpi):
    global DPI
    DPI = dpi


def set_figure_size(width, height):
    global WIDTH, HEIGHT
    WIDTH = width
    HEIGHT = height


def set_fontsize(fontsize, legend_fontsize=None, tick_fontsize=None, colorbar_fontsize=None, title_fontsize=None):
    global FONTSIZE, LEGEND_FONTSIZE, TICK_FONTSIZE, COLORBAR_FONTSIZE, TITLE_FONTSIZE
    FONTSIZE = fontsize
    if legend_fontsize is not None:
        LEGEND_FONTSIZE = legend_fontsize
    if tick_fontsize is not None:
        TICK_FONTSIZE = tick_fontsize
    if colorbar_fontsize is not None:
        COLORBAR_FONTSIZE = colorbar_fontsize
    if title_fontsize is not None:
        TITLE_FONTSIZE = title_fontsize


def _set_figsize(x_embed, width=WIDTH, height=HEIGHT, real_aspect_ratio=False, fix='width', margin=0.0):
    figsize = (width, height)
    if real_aspect_ratio:
        aspect_ratio = (x_embed[:, 1].max() - x_embed[:, 1].min()) / (x_embed[:, 0].max() - x_embed[:, 0].min())
        if fix == 'height':
            figsize = (height/aspect_ratio+margin, height)
        else:
            figsize = (width, (width-margin)*aspect_ratio+margin)
    return figsize


def compute_figsize(x_embed, width=WIDTH, height=HEIGHT, real_aspect_ratio=True, fix='width', margin=0.0):
    """Compute the figure size based on the aspect ratio of the data embedding.

    Args:
        x_embed (:class:`numpy.ndarray`):
            2D embedding for visualization, (N,2)
        real_aspect_ratio (bool, optional):
            Whether to set the aspect ratio of the plot to be the same as the data.
            Defaults to False.
        width (float, optional):
            Figure width. Defaults to WIDTH.
        height (float, optional):
            Figure height. Defaults to HEIGHT.
        fix (str, optional):
            Whether to fix the width or height of the figure. Defaults to 'width'.
        margin (float, optional):
            Margin of the figure. Defaults to 0.0.
    """
    return _set_figsize(x_embed, width, height, real_aspect_ratio, fix, margin)


def get_colors(n, color_map=None):
    """Get colors for plotting cell clusters.
    The colors can either be a categorical colormap or a continuous colormap.

    Args:
        n (int):
            Number of cell clusters
        color_map (str, optional):
            User-defined colormap. 
            If not set, the colors will be chosen as
            the colors for tabular data in matplotlib.
            Defaults to None.

    Returns:
        list[tuple]: list of color parameters
    """
    if color_map is None:  # default color
        if n <= 10:
            return TAB10[:n]
        elif n <= 24:
            return CATEGORICAL[:n]
        elif n <= 40:
            TAB40 = TAB20B+TAB20C
            return TAB40[:n]
        else:
            print("Warning: Number of colors exceeds the maximum (40)! Use a continuous colormap (256) instead.")
            return RAINBOW[:n]
    else:
        color_map_obj = list(plt.get_cmap(color_map).colors)
        k = len(color_map_obj)//n
        colors = ([color_map_obj(i) for i in range(0, len(color_map_obj), k)]
                  if k > 0 else
                  [color_map_obj(i) for i in range(len(color_map_obj))])
    return colors


def save_fig(fig,
             save,
             bbox_extra_artists=None,
             disable_display=False):
    """Save a figure

    Args:
        fig (:class:`matplotlib.figure.Figure`):
            Figure object.
        save (str):
            Figure name for saving (including path).
        bbox_extra_artists (tuple, optional):
            Extra artists to be included in the bounding box. Defaults to None.
    """
    global DPI
    if save is not None:
        try:
            idx = save.rfind('.')
            fig.savefig(save,
                        dpi=DPI,
                        bbox_extra_artists=bbox_extra_artists,
                        format=save[idx+1:],
                        bbox_inches='tight',
                        transparent=True)
        except FileNotFoundError:
            print("Saving failed. File path doesn't exist!")

        if disable_display:
            plt.close(fig)


############################################################
# Functions used in debugging.
############################################################
def plot_sig(t,
             u, s,
             upred, spred,
             cell_labels=None,
             title="Gene",
             save=None,
             **kwargs):
    """Generate a 2x2 u/s-t plot for a single gene

    Args:
        t (:class:`numpy.ndarray`):
            Cell time, (N, )
        u (:class:`numpy.ndarray`):
            Unspliced counts of a single gene, (N, )
        s (:class:`numpy.ndarray`):
            Spliced counts of a single gene, (N, )
        upred (:class:`numpy.ndarray`):
            Predicted unspliced counts of a single gene, (N, )
        spred (:class:`numpy.ndarray`):
            Predicted spliced counts of a single gene, (N, )
        cell_labels (:class:`numpy.ndarray`, optional):
            Cell type annotations. Defaults to None.
        title (str, optional):
            Figure title. Defaults to "Gene".
        save (str, optional):
            Figure name for saving (including path). Defaults to None.
    """
    D = kwargs['sparsify'] if 'sparsify' in kwargs else 1
    tscv = kwargs['tscv'] if 'tscv' in kwargs else t
    tdemo = kwargs["tdemo"] if "tdemo" in kwargs else t
    if cell_labels is None:
        fig, ax = plt.subplots(2, 1, figsize=(15, 12), facecolor='white')
        ax[0].plot(t[::D], u[::D], 'b.', label="raw")
        ax[1].plot(t[::D], s[::D], 'b.', label="raw")
        ax[0].plot(tdemo, upred, '.', color='lawngreen', label='Prediction', linewidth=2.0)
        ax[1].plot(tdemo, spred, '.', color='lawngreen', label="Prediction", linewidth=2.0)

        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("U", fontsize=18)

        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("S", fontsize=18)

        handles, labels = ax[1].get_legend_handles_labels()
    else:
        fig, ax = plt.subplots(2, 2, figsize=(24, 12), facecolor='white')
        labels_pred = kwargs['labels_pred'] if 'labels_pred' in kwargs else []
        labels_demo = kwargs['labels_demo'] if 'labels_demo' in kwargs else None
        cell_types = np.unique(cell_labels)
        colors = get_colors(len(cell_types), None)

        # Plot the input data in the true labels
        for i, type_ in enumerate(cell_types):
            mask_type = cell_labels == type_
            ax[0, 0].scatter(tscv[mask_type][::D],
                             u[mask_type][::D],
                             s=8.0,
                             color=colors[i % len(colors)],
                             alpha=0.7,
                             label=type_,
                             edgecolors='none')
            ax[0, 1].scatter(tscv[mask_type][::D],
                             s[mask_type][::D],
                             s=8.0,
                             color=colors[i % len(colors)],
                             alpha=0.7,
                             label=type_, edgecolors='none')
            if len(labels_pred) > 0:
                mask_mytype = labels_pred == type_
                ax[1, 0].scatter(t[mask_mytype][::D],
                                 u[mask_mytype][::D],
                                 s=8.0,
                                 color=colors[i % len(colors)],
                                 alpha=0.7,
                                 label=type_,
                                 edgecolors='none')
                ax[1, 1].scatter(t[mask_mytype][::D],
                                 s[mask_mytype][::D],
                                 s=8.0,
                                 color=colors[i % len(colors)],
                                 alpha=0.7,
                                 label=type_,
                                 edgecolors='none')
            else:
                ax[1, 0].scatter(t[mask_type][::D],
                                 u[mask_type][::D],
                                 s=8.0,
                                 color=colors[i % len(colors)],
                                 alpha=0.7,
                                 label=type_,
                                 edgecolors='none')
                ax[1, 1].scatter(t[mask_type][::D],
                                 s[mask_type][::D],
                                 s=8.0,
                                 color=colors[i % len(colors)],
                                 alpha=0.7,
                                 label=type_,
                                 edgecolors='none')

        if labels_demo is not None:
            for i, type_ in enumerate(cell_types):
                mask_mytype = labels_demo == type_
                order = np.argsort(tdemo[mask_mytype])
                ax[1, 0].plot(tdemo[mask_mytype][order],
                              upred[mask_mytype][order],
                              color=colors[i % len(colors)],
                              linewidth=2.0)
                ax[1, 1].plot(tdemo[mask_mytype][order],
                              spred[mask_mytype][order],
                              color=colors[i % len(colors)],
                              linewidth=2.0)
        else:
            order = np.argsort(tdemo)
            ax[1, 0].plot(tdemo[order], upred[order], 'k.', linewidth=2.0)
            ax[1, 1].plot(tdemo[order], spred[order], 'k.', linewidth=2.0)

        if 't_trans' in kwargs:
            t_trans = kwargs['t_trans']
            for i, type_ in enumerate(cell_types):
                ax[1, 0].plot([t_trans[i], t_trans[i]], [0, u.max()], '-x', color=colors[i % len(colors)])
                ax[1, 1].plot([t_trans[i], t_trans[i]], [0, s.max()], '-x', color=colors[i % len(colors)])
        for j in range(2):
            ax[j, 0].set_xlabel("Time")
            ax[j, 0].set_ylabel("U", fontsize=18)

            ax[j, 1].set_xlabel("Time")
            ax[j, 1].set_ylabel("S", fontsize=18)
            handles, labels = ax[1, 0].get_legend_handles_labels()

        if 'subtitles' in kwargs:
            ax[0, 0].set_title(f"Unspliced, {kwargs['subtitles'][0]}")
            ax[0, 1].set_title(f"Spliced, {kwargs['subtitles'][0]}")
            ax[1, 0].set_title(f"Unspliced, {kwargs['subtitles'][1]}")
            ax[1, 1].set_title(f"Spliced, {kwargs['subtitles'][1]}")
        else:
            ax[0, 0].set_title('Unspliced, True Label')
            ax[0, 1].set_title('Spliced, True Label')
            ax[1, 0].set_title('Unspliced, VAE')
            ax[1, 1].set_title('Spliced, VAE')
    lgd = fig.legend(handles,
                     labels,
                     fontsize=15,
                     markerscale=5,
                     ncol=4,
                     bbox_to_anchor=(0.0, 1.0, 1.0, 0.25),
                     loc='center')
    fig.suptitle(title, fontsize=28)
    plt.tight_layout()

    save_fig(fig, save, (lgd,))
    return


def plot_phase(u, s,
               upred, spred,
               title,
               track_idx=None,
               labels=None,
               types=None,
               save=None):
    """Plot the phase portrait of a gene

    Args:
        u (:class:`numpy array`):
            Unspliced counts of a single gene, (N, )
        s (:class:`numpy array`):
            Spliced counts of a single gene, (N, )
        upred (:class:`numpy array`):
            Predicted unspliced counts of a single gene, (N, )
        spred (:class:`numpy array`):
            Predicted spliced counts of a single gene, (N, )
        title (str):
            Figure title.
        track_idx (:class:`numpy array`, optional):
            Cell indices to plot correspondence between data points and predicted phase portrait.
            Defaults to None.
        labels (_type_, optional):
            Cell state annotation (off, induction or repression). Defaults to None.
        types (:class:`numpy.ndarray`, optional):
            Unique cell types
        save (str, optional):
            Figure name for saving (including path). Defaults to None.
    """
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
    if labels is None or types is None:
        ax.scatter(s, u, c="b", alpha=0.5)
    else:
        colors = get_colors(len(types), None)
        for i, type_ in enumerate(types):
            ax.scatter(s[labels == i], u[labels == i], color=colors[i % len(colors)], alpha=0.3, label=type_)
    ax.plot(spred, upred, 'k.', label="ode")
    # Plot the correspondence
    if track_idx is None:
        perm = np.random.permutation(len(s))
        Nsample = 50
        s_comb = np.stack([s[perm[:Nsample]], spred[perm[:Nsample]]]).ravel('F')
        u_comb = np.stack([u[perm[:Nsample]], upred[perm[:Nsample]]]).ravel('F')
    else:
        s_comb = np.stack([s[track_idx], spred[track_idx]]).ravel('F')
        u_comb = np.stack([u[track_idx], upred[track_idx]]).ravel('F')

    for i in range(0, len(s_comb), 2):
        ax.plot(s_comb[i:i+2], u_comb[i:i+2], 'k-', linewidth=0.8)
    ax.set_xlabel("S", fontsize=18)
    ax.set_ylabel("U", fontsize=18)

    handles, labels = ax.get_legend_handles_labels()
    lgd = fig.legend(handles,
                     labels,
                     fontsize=15,
                     markerscale=5,
                     ncol=4,
                     bbox_to_anchor=(0.0, 1.0, 1.0, 0.25),
                     loc='center')
    fig.suptitle(title)

    save_fig(fig, save, (lgd,))


def plot_cluster_axis(ax,
                      x_embed,
                      cell_labels,
                      palette=None,
                      color_map=None,
                      embed=None,
                      real_aspect_ratio=False,
                      markersize=20):
    """Same as plot_cluster but returns the axes object.
    """
    cell_types = np.unique(cell_labels)
    x = x_embed[:, 0]
    y = x_embed[:, 1]
    x_range = x.max()-x.min()
    y_range = y.max()-y.min()
    if palette is None:
        palette = get_colors(len(cell_types), color_map)

    n_char_max = np.max([len(x) for x in cell_types])
    for i, typei in enumerate(cell_types):
        mask = cell_labels == typei
        xbar, ybar = np.mean(x[mask]), np.mean(y[mask])
        ax.scatter(x[mask], y[mask], s=markersize, color=palette[i % len(palette)], edgecolors='none', label=typei)
    if embed is not None:
        ax.set_xlabel(f'{embed} 1')
        ax.set_ylabel(f'{embed} 2')
    return ax


def plot_cluster(x_embed,
                 cell_labels,
                 embed=None,
                 figsize=None,
                 real_aspect_ratio=True,
                 palette=None,
                 markersize=MARKERSIZE,
                 markerscale=MARKERSCALE,
                 show_legend=True,
                 legend_fontsize=LEGEND_FONTSIZE,
                 ncols=None,
                 bbox_to_anchor=None,
                 save=None,):
    """Plot the predicted cell types from the encoder

    Args:
        x_embed (:class:`numpy.ndarray`):
            2D embedding for visualization, (N,2)
        cell_labels (:class:`numpy.ndarray`):
             Cell type annotation, (N,)
        embed (str, optional):
            Embedding name. Used for labeling axes. Defaults to 'umap'.
        figsize (tuple, optional):
            Figure size. Defaults to None.
        real_aspect_ratio (bool, optional):
            Whether to set the aspect ratio of the plot to be the same as the data.
            Defaults to False.
        palette (list, optional):
            Color palette for cell clusters. Defaults to None.
        markersize (int, optional):
            Marker size. Defaults to 3.
        markerscale (float, optional):
            Marker scale. Defaults to 1.0.
        legend_fontsize (int, optional):
            Legend font size. Defaults to 5.
        ncols (int, optional):
            Number of columns in the legend. Defaults to None.
        bbox_to_anchor (tuple, optional):
            Bounding box for the legend. Defaults to None.
        save (str, optional):
            Figure name for saving (including path). Defaults to None.
    """
    cell_types = np.unique(cell_labels)
    if figsize is None:
        figsize = _set_figsize(x_embed, real_aspect_ratio=real_aspect_ratio)
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    x = x_embed[:, 0]
    y = x_embed[:, 1]
    x_range = x.max()-x.min()
    y_range = y.max()-y.min()

    if palette is None:
        palette = get_colors(len(cell_types))

    n_char_max = np.max([len(x) for x in cell_types])
    for i, typei in enumerate(cell_types):
        mask = cell_labels == typei
        xbar, ybar = np.mean(x[mask]), np.mean(y[mask])
        ax.scatter(x[mask], y[mask],
                   s=markersize,
                   color=palette[i % len(palette)],
                   edgecolors='none')
        n_char = len(typei)
        if bbox_to_anchor is None and show_legend:
            txt = ax.text(xbar - x_range*4e-3*n_char, ybar - y_range*4e-3, typei, fontsize=max(15, 100//n_char_max), color='k')
            txt.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='black'))
    if bbox_to_anchor is not None:
        if len(bbox_to_anchor) == 2 and show_legend:
            lgd = ax.legend(cell_types,
                            fontsize=legend_fontsize,
                            markerscale=markerscale,
                            bbox_to_anchor=bbox_to_anchor)
        elif len(bbox_to_anchor) == 4 and show_legend:
            if ncols is None:
                ncols = max(len(cell_types), 5)
            lgd = ax.legend(cell_types,
                            ncols=ncols,
                            fontsize=legend_fontsize,
                            markerscale=markerscale,
                            bbox_to_anchor=bbox_to_anchor,
                            loc='center')
    if embed is not None:
        ax.set_xlabel(f'{embed} 1')
        ax.set_ylabel(f'{embed} 2')
    ax.set_axis_off()
    if bbox_to_anchor is not None:
        save_fig(fig, save, (lgd,))
    else:
        save_fig(fig, save)


def plot_train_loss(loss, iters, save=None):
    """Plots the training loss values versus iteration numbers.

    Args:
        loss (array like):
            Loss values.
        iters (array like):
            Iteration numbers.
        save (str, optional):
            Figure name for saving (including path). Defaults to None.
    """
    fig, ax = plt.subplots(facecolor='white')
    ax.plot(iters, loss, '.-')
    ax.set_title("Training Loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")

    save_fig(fig, save)


def plot_test_loss(loss, iters, save=None):
    """Plots the validation loss values versus iteration numbers.

    Args:
        loss (array like):
            Loss values.
        iters (array like):
            Iteration numbers.
        save (str, optional):
            Figure name for saving (including path). Defaults to None.
    """
    fig, ax = plt.subplots(facecolor='white')
    ax.plot(iters, loss, '.-')
    ax.set_title("Testing Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    save_fig(fig, save)


def cellwise_vel(adata,
                 key,
                 gidx,
                 plot_indices,
                 dt=0.2,
                 plot_raw=False,
                 u0=None,
                 s0=None,
                 t0=None,
                 save=None):
    """Plots u and s vs. time and velocity arrows for a subset of cells.
    Used for debugging.

    Args:
        adata (:class:`anndata.AnnData`):
            AnnData Object.
        key (str):
            Key for extracting inferred latent variables and parameters.
        gidx (int):
            Index of the gene to plot.
        plot_indices (:class:`numpy.ndarray`):
            Indices of cells for velocity quiver plot.
        dt (float, optional):
            Time interval to compute displacement of u and s. Defaults to 0.2.
        plot_raw (bool, optional):
            Whether to plot raw count data. Defaults to False.
        u0 (:class:`numpy.ndarray`, optional):
            Unspliced initial conditions. Defaults to None.
        s0 (:class:`numpy.ndarray`, optional):
            Spliced initial conditions. Defaults to None.
        t0 (:class:`numpy.ndarray`, optional):
            Time at initial conditions. Defaults to None.
        save (str, optional):
            Figure name for saving (including path). Defaults to None.
    """
    fig, ax = plt.subplots(1, 2, figsize=(18, 6), facecolor='white')

    u = np.array(adata.layers["unspliced"][:, gidx].todense()).squeeze()
    s = np.array(adata.layers["spliced"][:, gidx].todense()).squeeze()
    t = adata.obs[f'{key}_time'].to_numpy()
    if u0 is None:
        u0 = adata.layers[f'{key}_u0'][:, gidx]
    if s0 is None:
        s0 = adata.layers[f'{key}_s0'][:, gidx]
    if t0 is None:
        t0 = adata.obs[f'{key}_t0'].to_numpy()

    uhat = adata.layers[f'{key}_uhat'][:, gidx]
    shat = adata.layers[f'{key}_shat'][:, gidx]
    scaling_u = adata.var[f'{key}_scaling_u'].to_numpy()[gidx]
    scaling_s = adata.var[f'{key}_scaling_s'].to_numpy()[gidx]
    rho = adata.layers[f'{key}_rho'][:, gidx]
    try:
        alpha = adata.var[f'{key}_alpha'].to_numpy()[gidx]
        beta = adata.var[f'{key}_beta'].to_numpy()[gidx]
    except KeyError:
        alpha = np.exp(adata.var[f'{key}_logmu_alpha'].to_numpy()[gidx])
        beta = np.exp(adata.var[f'{key}_logmu_beta'].to_numpy()[gidx])
    vu = rho * alpha - beta * uhat / scaling_u
    v = adata.layers[f'{key}_velocity'][:, gidx]
    ax[0].plot(t, uhat/scaling_u, '.', color='grey', alpha=0.1)
    ax[1].plot(t, shat/scaling_s, '.', color='grey', alpha=0.1)
    if plot_raw:
        ax[0].plot(t[plot_indices], u[plot_indices], 'o', color='b', label="Raw Count")
        ax[1].plot(t[plot_indices], s[plot_indices], 'o', color='b')
    if dt > 0:
        ax[0].quiver(t[plot_indices],
                     uhat[plot_indices]/scaling_u,
                     dt*np.ones((len(plot_indices),)),
                     vu[plot_indices]*dt,
                     angles='xy')
        ax[1].quiver(t[plot_indices],
                     shat[plot_indices]/scaling_s,
                     dt*np.ones((len(plot_indices),)),
                     v[plot_indices]*dt,
                     angles='xy')
    for i, k in enumerate(plot_indices):
        if i == 0:
            ax[0].plot([t0[k], t[k]], [u0[k]/scaling_u, uhat[k]/scaling_u], 'r-o', label='Prediction')
            ax[0].plot([t0[k]], [u0[k]/scaling_u], 'co', label='Initial Condition')
        else:
            ax[0].plot([t0[k], t[k]], [u0[k]/scaling_u, uhat[k]/scaling_u], 'r-o')
            ax[0].plot([t0[k]], [u0[k]/scaling_u], 'co')
        ax[1].plot([t0[k], t[k]], [s0[k]/scaling_s, shat[k]/scaling_s], 'r-o')
        ax[1].plot([t0[k]], [s0[k]/scaling_s], 'co')
        if plot_raw:
            ax[0].plot(t[k]*np.ones((2,)), [min(u[k], uhat[k]/scaling_u), max(u[k], uhat[k]/scaling_u)], 'b--')
            ax[1].plot(t[k]*np.ones((2,)), [min(s[k], shat[k]/scaling_s), max(s[k], shat[k]/scaling_s)], 'b--')

    ax[0].set_ylabel("U", fontsize=16)
    ax[1].set_ylabel("S", fontsize=16)
    fig.suptitle(adata.var_names[gidx], fontsize=30)
    fig.legend(loc=1, fontsize=18)
    plt.tight_layout()

    save_fig(fig, save)


def cellwise_vel_embedding(adata,
                           key,
                           type_name=None,
                           idx=None,
                           embed='umap',
                           markersize=80,
                           real_aspect_ratio=False,
                           save=None):
    """Plots velocity of a subset of cells on an embedding space.
    Used for debugging.

    Args:
        adata (:class:`anndata.AnnData`):
            AnnData object.
        key (str):
            Key for retreiving parameters and data for plotting.
        type_name (str, optional):
            Specific cell type to plot. Defaults to None.
        idx (:class:`numpy.ndarray`, optional):
            Indices of cells for plotting. Defaults to None.
            When set to None, cells will be randomly sampled.
        embed (str, optional):
            Embedding velocity is computed upon. Defaults to 'umap'.
        save (str, optional):
            Figure name for saving (including path). Defaults to None.

    Returns:
        :class:`numpy.ndarray`: Indices of cells for plotting.
    """
    if f'{key}_velocity_graph' not in adata.uns:
        print("Please run 'velocity_graph' and 'velocity_embedding' first!")
        return

    A = np.array(adata.uns[f"{key}_velocity_graph"].todense())
    A_neg = np.array(adata.uns[f"{key}_velocity_graph_neg"].todense())
    v_embed = adata.obsm[f'{key}_velocity_{embed}']
    x_embed = adata.obsm[f'X_{embed}']
    x_embed_1, x_embed_2 = x_embed[:, 0], x_embed[:, 1]

    if idx is None:
        if type_name is None:
            idx = np.random.choice(x_embed.shape[0])
        else:
            cell_labels = adata.obs["clusters"].to_numpy()
            idx = np.random.choice(np.where(cell_labels == type_name)[0])

    neighbors = np.where((A[idx] > 0) | (A_neg[idx] < 0))[0]
    t = adata.obs[f'{key}_time'].to_numpy()

    figsize = _set_figsize(x_embed, real_aspect_ratio=real_aspect_ratio)
    fig, ax = plt.subplots(1, 2, figsize=figsize, facecolor='white')
    ax[0].plot(x_embed_1, x_embed_2, '.', color='grey', alpha=0.25)
    tmask = t[neighbors] > t[idx]
    ax[0].plot(x_embed_1[neighbors[~tmask]], x_embed_2[neighbors[~tmask]], 'c.', label="Earlier Neighbors")
    ax[0].plot(x_embed_1[neighbors[tmask]], x_embed_2[neighbors[tmask]], 'b.', label="Later Neighbors")
    ax[0].plot(x_embed_1[[idx]], x_embed_2[[idx]], 'ro', label="Target Cell")
    ax[0].legend(loc=1)

    corr = A[idx, neighbors]+A_neg[idx, neighbors]
    _plot_heatmap(ax[1], corr, x_embed[neighbors], 'Cosine Similarity', markersize=markersize)
    ax[1].quiver(x_embed_1[[idx]], x_embed_2[[idx]], [v_embed[idx, 0]], [v_embed[idx, 1]], angles='xy')
    ax[1].plot(x_embed_1[[idx]], x_embed_2[[idx]], 'ks', markersize=10, label="Target Cell")

    save_fig(fig, save)
    return idx


from networkx import DiGraph, Graph
from matplotlib.collections import LineCollection
##########################################################
# Reference: scVelo
# https://github.com/theislab/scvelo/blob/main/scvelo/plotting/velocity_graph.py#L163
##########################################################
def _draw_networkx_edges(
    G,
    pos,
    edgelist=None,
    width=1.0,
    edge_color="k",
    style="solid",
    alpha=None,
    arrowstyle="-|>",
    arrowsize=3,
    edge_cmap=None,
    edge_vmin=None,
    edge_vmax=None,
    ax=None,
    arrows=True,
    label=None,
    node_size=300,
    nodelist=None,
    node_shape="o",
    connectionstyle=None,
    min_source_margin=0,
    min_target_margin=0,
):
    """Draw the edges of the graph G. Adjusted from networkx."""
    try:
        from numbers import Number

        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.colors import colorConverter, Colormap, Normalize
        from matplotlib.patches import FancyArrowPatch
    except ImportError:
        raise ImportError("Matplotlib required for draw()")
    except RuntimeError:
        print("Matplotlib unable to open display")
        raise
    if ax is None:
        ax = plt.gca()

    if edgelist is None:
        edgelist = list(G.edges())

    if not edgelist or len(edgelist) == 0:  # no edges!
        print('No edges!')
        return None

    if nodelist is None:
        nodelist = list(G.nodes())

    # FancyArrowPatch handles color=None different from LineCollection
    if edge_color is None:
        edge_color = "k"

    # set edge positions
    edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])

    # Check if edge_color is an array of floats and map to edge_cmap.
    # This is the only case handled differently from matplotlib
    if (
        np.iterable(edge_color)
        and (len(edge_color) == len(edge_pos))
        and np.alltrue([isinstance(c, Number) for c in edge_color])
    ):
        if edge_cmap is not None:
            assert isinstance(edge_cmap, Colormap)
        else:
            edge_cmap = plt.get_cmap()
        if edge_vmin is None:
            edge_vmin = min(edge_color)
        if edge_vmax is None:
            edge_vmax = max(edge_color)
        color_normal = Normalize(vmin=edge_vmin, vmax=edge_vmax)
        edge_color = [edge_cmap(color_normal(e)) for e in edge_color]

    if not G.is_directed() or not arrows:
        edge_collection = LineCollection(
            edge_pos,
            colors=edge_color,
            linewidths=width,
            antialiaseds=(1,),
            linestyle=style,
            alpha=alpha,
        )

        edge_collection.set_cmap(edge_cmap)
        edge_collection.set_clim(edge_vmin, edge_vmax)

        edge_collection.set_zorder(1)  # edges go behind nodes
        edge_collection.set_label(label)
        ax.add_collection(edge_collection)

        return edge_collection

    arrow_collection = None

    if G.is_directed() and arrows:
        # Note: Waiting for someone to implement arrow to intersection with
        # marker.  Meanwhile, this works well for polygons with more than 4
        # sides and circle.

        def to_marker_edge(marker_size, marker):
            if marker in "s^>v<d":  # `large` markers need extra space
                return np.sqrt(2 * marker_size) / 2
            else:
                return np.sqrt(marker_size) / 2

        # Draw arrows with `matplotlib.patches.FancyarrowPatch`
        arrow_collection = []
        mutation_scale = arrowsize  # scale factor of arrow head

        # FancyArrowPatch doesn't handle color strings
        arrow_colors = colorConverter.to_rgba_array(edge_color, alpha)
        for i, (src, dst) in enumerate(edge_pos):
            x1, y1 = src
            x2, y2 = dst
            shrink_source = 0  # space from source to tail
            shrink_target = 0  # space from  head to target
            if np.iterable(node_size):  # many node sizes
                source, target = edgelist[i][:2]
                source_node_size = node_size[nodelist.index(source)]
                target_node_size = node_size[nodelist.index(target)]
                shrink_source = to_marker_edge(source_node_size, node_shape)
                shrink_target = to_marker_edge(target_node_size, node_shape)
            else:
                shrink_source = shrink_target = to_marker_edge(node_size, node_shape)

            if shrink_source < min_source_margin:
                shrink_source = min_source_margin

            if shrink_target < min_target_margin:
                shrink_target = min_target_margin

            if len(arrow_colors) == len(edge_pos):
                arrow_color = arrow_colors[i]
            elif len(arrow_colors) == 1:
                arrow_color = arrow_colors[0]
            else:  # Cycle through colors
                arrow_color = arrow_colors[i % len(arrow_colors)]

            if np.iterable(width):
                if len(width) == len(edge_pos):
                    line_width = width[i]
                else:
                    line_width = width[i % len(width)]
            else:
                line_width = width

            arrow = FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle=arrowstyle,
                shrinkA=shrink_source,
                shrinkB=shrink_target,
                mutation_scale=mutation_scale,
                color=arrow_color,
                linewidth=line_width,
                connectionstyle=connectionstyle,
                linestyle=style,
                zorder=1,
            )  # arrows go behind nodes

            # There seems to be a bug in matplotlib to make collections of
            # FancyArrowPatch instances. Until fixed, the patches are added
            # individually to the axes instance.
            arrow_collection.append(arrow)
            ax.add_patch(arrow)

    # update view
    minx = np.amin(np.ravel(edge_pos[:, :, 0]))
    maxx = np.amax(np.ravel(edge_pos[:, :, 0]))
    miny = np.amin(np.ravel(edge_pos[:, :, 1]))
    maxy = np.amax(np.ravel(edge_pos[:, :, 1]))

    w = maxx - minx
    h = maxy - miny
    padx, pady = 0.05 * w, 0.05 * h
    corners = (minx - padx, miny - pady), (maxx + padx, maxy + pady)
    ax.update_datalim(corners)
    ax.autoscale_view()

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return arrow_collection


def plot_spatial_graph(adata,
                       graph_key="spatial_graph",
                       basis="spatial",
                       palette=None,
                       width=WIDTH,
                       height=HEIGHT,
                       fix='width',
                       node_size=30,
                       edge_width=0.25,
                       arrowsize=3,
                       edge_color='gray',
                       legend_fontsize=LEGEND_FONTSIZE,
                       show_legend=False,
                       real_aspect_ratio=True,
                       components=[0, 1],
                       save=None):
    """Plot the spatial graph on the spatial embedding.
    
    Args:
        adata (:class:`anndata.AnnData`):
            AnnData object.
        graph_key (str, optional):
            Key for the spatial graph. Defaults to "spatial_graph".
        basis (str, optional):
            Basis for spatial embedding. Defaults to "spatial".
        palette (str, optional):
            Color palette for cell types. Defaults to None.
        width (int, optional):
            Figure width. Defaults to WIDTH.
        height (int, optional):
            Figure height. Defaults to HEIGHT.
        fix (str, optional):
            Fix one of the dimensions, either 'width' or 'height'. Defaults to 'width'.
        node_size (int, optional):
            Node size. Defaults to 30.
        edge_width (float, optional):
            Edge width. Defaults to 0.25.
        arrowsize (int, optional):
            Arrow size. Defaults to 3.
        edge_color (str, optional):
            Edge color. Defaults to 'gray'.
        legend_fontsize (int, optional):
            Legend fontsize. Defaults to LEGEND_FONTSIZE.
        show_legend (bool, optional):
            Whether to show legend. Defaults to False.
        components (list[int], optional):
            Components to plot. Used in the case of more than 2 dimensions. Defaults to [0, 1].
        save (str, optional):
            Figure name for saving (including path). Defaults to None.
    """
    T = adata.obsp[graph_key].A

    x_emb = adata.obsm[f"X_{basis}"][:, np.array(components)]
    
    xmin, xmax = x_emb[:, 0].min(), x_emb[:, 0].max()
    ymin, ymax = x_emb[:, 1].min(), x_emb[:, 1].max()
    figsize = compute_figsize(x_emb, width, height, real_aspect_ratio)
    fig, ax = plt.subplots(figsize=figsize)
    
    edge_collection = _draw_networkx_edges(
        Graph(T),
        x_emb,
        node_size=node_size,
        width=edge_width,
        edge_color=edge_color,
        arrowsize=arrowsize
    )
    edge_segs = edge_collection.get_segments()
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    # scv.pl.scatter(adata, basis=basis, title="", legend_loc='right margin', size=node_size, ax=ax)
    ax = plot_cluster_axis(ax,
                           x_emb,
                           adata.obs['clusters'].to_numpy(),
                           palette)

    colors = 'gray'

    line_segments = LineCollection(edge_segs,
                                   linewidths=edge_width,
                                   colors=colors,
                                   linestyle='solid')
    ax.add_collection(line_segments)
    # ax.set_title('Spatial Graph', fontsize=28)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles,
                        labels,
                        fontsize=legend_fontsize,
                        markerscale=2.0,
                        bbox_to_anchor=(1.0, 1.0),
                        loc='upper left')
        plt.tight_layout()
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 2, box.height])
        if save is not None:
            save_fig(fig, save, (lgd,))
            return
    if save is not None:
        save_fig(fig, save)


#########################################################################
# Velocity quiver plot on the phase portrait
# Reference:
# Shengyu Li#, Pengzhi Zhang#, Weiqing Chen, Lingqun Ye, 
# Kristopher W. Brannan, Nhat-Tu Le, Jun-ichi Abe, John P. Cooke, 
# Guangyu Wang. A relay velocity model infers cell-dependent RNA velocity. 
# Nature Biotechnology (2023) https://doi.org/10.1038/s41587-023-01728-5
#########################################################################
def pick_grid_points(x, grid_size=(30, 30), percentile=25):
    def gaussian_kernel(X, mu=0, sigma=1):
        return np.exp(-(X - mu)**2 / (2*sigma**2)) / np.sqrt(2*np.pi*sigma**2)
    grs = []
    for dim_i in range(x.shape[1]):
        m, M = np.min(x[:, dim_i]), np.max(x[:, dim_i])
        m = m - 0.025 * np.abs(M - m)
        M = M + 0.025 * np.abs(M - m)
        gr = np.linspace(m, M, grid_size[dim_i])
        grs.append(gr)
    meshes_tuple = np.meshgrid(*grs)
    gridpoints_coordinates = np.vstack([i.flat for i in meshes_tuple]).T
    gridpoints_coordinates = gridpoints_coordinates + norm.rvs(loc=0, scale=0.15, size=gridpoints_coordinates.shape)
    
    np.random.seed(42)  # set random seed
    
    nn = NearestNeighbors()

    neighbors_1 = min((x.shape[0]-1), 20)
    nn.fit(x)
    dist, ixs = nn.kneighbors(gridpoints_coordinates, neighbors_1)

    ix_choice = ixs[:, 0].flat[:]
    ix_choice = np.unique(ix_choice)

    nn = NearestNeighbors()

    neighbors_2 = min((x.shape[0]-1), 20)
    nn.fit(x)
    dist, ixs = nn.kneighbors(x[ix_choice], neighbors_2)

    density_extimate = gaussian_kernel(dist, mu=0, sigma=0.5).sum(1)
    bool_density = density_extimate > np.percentile(density_extimate, percentile)
    ix_choice = ix_choice[bool_density]
    return ix_choice


def plot_phase_vel(adata,
                   gene,
                   key,
                   dt=0.05,
                   grid_size=(30, 30),
                   percentile=25,
                   markersize=20,
                   width=WIDTH,
                   height=HEIGHT,
                   save=None):
    """Plots RNA velocity stream on a phase portrait.

    Args:
        adata (:class:`anndata.AnnData`):
            AnnData object
        gene (str):
            Gene name.
        key (str):
            Key for retreiving data.
        dt (float, optional):
            Time interval used to compute u and s displacement.
            Defaults to 0.05.
        grid_size (tuple[int], optional):
            Number of rows and columns for grid points
            on which velocity will be computed based on KNN interpolation.
            Defaults to (30, 30).
        percentile (int, optional):
            Hyperparameter for grid point picking. Defaults to 25.
        markersize (int, optional):
            Marker size. Defaults to 20.
        width (int, optional):
            Figure width. Defaults to WIDTH.
        height (int, optional):
            Figure height. Defaults to HEIGHT.
        save (str, optional):
            Figure name for saving (including path). Defaults to None.
    """
    fig, ax = plt.subplots(figsize=(width, height))
    gidx = np.where(adata.var_names == gene)[0][0]
    scaling = adata.var[f'{key}_scaling'].iloc[gidx]
    t = adata.obs[f'{key}_time'].to_numpy()
    vu = adata.layers[f'{key}_velocity_u'][:, gidx]
    vs = adata.layers[f'{key}_velocity'][:, gidx]
    u = adata.layers['Mu'][:, gidx]/scaling
    s = adata.layers['Ms'][:, gidx]
    x = np.stack([s, u]).T
    _plot_heatmap(ax, t, x, 'time', markersize=markersize)
    grid_points = pick_grid_points(x, grid_size, percentile)
    ax.quiver(s[grid_points],
              u[grid_points],
              dt*vs[grid_points],
              dt*vu[grid_points],
              angles='xy',
              scale=None,
              scale_units='inches',
              headwidth=5.0,
              headlength=8.0,
              color='k')
    save_fig(fig, save)


def plot_velocity(x_embed,
                  vx,
                  vy,
                  width=WIDTH,
                  height=HEIGHT,
                  fix='width',
                  real_aspect_ratio=False,
                  save=None):
    """2D quiver plot of velocity

    Args:
        x_embed (:class:`numpy.ndarray`):
            2D coordinates for visualization, (N, 2)
        vx (:class:`numpy.ndarray`):
            Velocity in the x direction.
        vy (:class:`numpy.ndarray`):
            Velocity in the y direction.
        width (int, optional):
            Figure width. Defaults to WIDTH.
        height (int, optional):
            Figure height. Defaults to HEIGHT.
        fix (str, optional):
            Whether to fix the width or height of the figure. Defaults to 'width'.
        real_aspect_ratio (bool, optional):
            Whether to set the aspect ratio of the plot to be the same as the data.
            Defaults to False.
        save (str, optional):
            Figure name for saving (including path). Defaults to None.
    """
    umap1, umap2 = x_embed[:, 0], x_embed[:, 1]
    figsize = _set_figsize(x_embed, width, height, real_aspect_ratio, fix=fix)
    fig, ax = plt.subplots(figsize=figsize)
    v = np.sqrt(vx**2+vy**2)
    vmax, vmin = np.quantile(v, 0.95), np.quantile(v, 0.05)
    v = np.clip(v, vmin, vmax)
    ax.plot(umap1, umap2, '.', alpha=0.5)
    ax.quiver(umap1, umap2, vx, vy, v, angles='xy')

    save_fig(fig, save)

#########################################################################
# Evaluation Plots
#########################################################################


def plot_spatial_extrapolation(xy,
                               xy_ext,
                               cell_labels,
                               colors=None,
                               markersize=MARKERSIZE,
                               legend_fontsize=LEGEND_FONTSIZE,
                               figsize=(WIDTH, HEIGHT),
                               save=None):
    """
    Plots the spatial extrapolation of data points.

    This function takes in the coordinates of original data points (xy), the extrapolated coordinates (xy_ext),
    the labels of the cells (cell_labels), and optional parameters for customizing the plot.
    It creates a scatter plot with the original data points shown in gray and the extrapolated data points
    shown in different colors based on their cell labels.

    Args:
        xy (ndarray): The coordinates of the original data points. Shape (n, 2).
        xy_ext (ndarray): The extrapolated coordinates. Shape (m, 2).
        cell_labels (ndarray): The labels of the cells. Shape (m,).
        colors (list, optional): The colors to use for different cell labels. If not provided, default colors will be used.
        markersize (int, optional): The size of the dots in the scatter plot. Default is MARKERSIZE.
        legend_fontsize (int, optional): The font size of the legend. Default is LEGEND_FONTSIZE.
        figsize (tuple, optional): The size of the figure. Default is (WIDTH, HEIGHT).
        save (str, optional): The file path to save the figure. If not provided, the figure will not be saved.

    Returns:
        None

    Note:
        This function is tentative and not stable. It may undergo changes in future versions.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(xy[:, 0],
               xy[:, 1],
               color='gray',
               s=markersize,
               alpha=0.1,
               edgecolors='none')
    cell_types = np.unique(cell_labels)
    if colors is None:
        colors = get_colors(len(cell_types))
    for i, type_ in enumerate(cell_types):
        ax.scatter(xy_ext[cell_labels == type_, 0],
                   xy_ext[cell_labels == type_, 1],
                   label=type_,
                   s=markersize,
                   color=colors[i],
                   edgecolors=None)
    
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles,
                    labels,
                    fontsize=legend_fontsize,
                    markerscale=2.0,
                    bbox_to_anchor=(0.05, 1.0),
                    loc='upper right')
    plt.tight_layout()
    if save is not None:
        save_fig(fig, save, bbox_extra_artists=(lgd,))


def plot_legend(adata,
                cluster_key='clusters',
                figsize=(6, 1),
                ncol=1,
                markerscale=MARKERSCALE,
                fontsize=LEGEND_FONTSIZE,
                palette=None,
                save='figures/legend.pdf'):
    """Plots figure legend containing all cell types.

    Args:
        adata (:class:`anndata.AnnData`):
            AnnData object.
        cluster_key (str, optional):
            Key for cell type annotations. Defaults to 'clusters'.
        ncol (int, optional):
            Number of columns of the legend. Defaults to 1.
        markerscale (float, optional):
            Marker scale. Defaults to MARKERSCALE.
        fontsize (int, optional):
            Font size. Defaults to LEGEND_FONTSIZE.
        palette (list, optional):
            List of colors for cell types. Defaults to None.
        save (str, optional):
            Figure name for saving (including path). Defaults to 'figures/legend.png'.
    """
    cell_labels = adata.obs[cluster_key].to_numpy()
    cell_labels = np.array([str(x) for x in cell_labels])
    cell_types = np.unique(cell_labels)
    if palette is None:
        palette = get_colors(len(cell_types))
    
    lines = []

    fig, ax = plt.subplots(figsize=figsize)
    for i, x in enumerate(cell_types):
        line = ax.plot([], 'o', color=palette[i], label=x)
        lines.append(line)
    ax.axis("off")
    lgd = ax.legend(markerscale=markerscale,
                    ncol=ncol,
                    fontsize=fontsize,
                    loc='center',
                    frameon=False)
    plt.tight_layout()
    save_fig(fig, save, (lgd,))


def _set_colorbar(ax,
                  vmin,
                  vmax,
                  cmap,
                  colorbar_name,
                  colorbar_ticklabels=None,
                  colorbar_fontsize=7,
                  colorbar_ticks=None,
                  colorbar_pos=[1.04, 0.2, 0.05, 0.6],
                  colorbar_tick_fontsize=5,
                  labelpad=5):
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cax = ax.inset_axes(colorbar_pos)
    cbar = plt.colorbar(sm, ax=ax, cax=cax)
    cbar.ax.get_yaxis().labelpad = labelpad
    cbar.ax.set_ylabel(colorbar_name, rotation=270, fontsize=colorbar_fontsize)

    if colorbar_ticklabels is not None:
        cbar.ax.set_yticklabels(colorbar_ticklabels, fontsize=colorbar_tick_fontsize)
        if colorbar_ticks is None:
            colorbar_ticks = np.linspace(vmin, vmax, len(colorbar_ticklabels))
            cbar.set_ticks(colorbar_ticks, labels=colorbar_ticklabels, fontsize=colorbar_tick_fontsize)
        else:
            cbar.set_ticks(colorbar_ticks, labels=colorbar_ticklabels, fontsize=colorbar_tick_fontsize)
    else:
        if colorbar_ticks is None:
            cbar.set_ticks([vmin, vmax], labels=[vmin, vmax], fontsize=colorbar_tick_fontsize)
        else:
            cbar.set_ticks(colorbar_ticks, labels=colorbar_ticks, fontsize=colorbar_tick_fontsize)
        cbar.ax.tick_params(labelsize=colorbar_tick_fontsize)
    return ax


def _plot_heatmap(ax,
                  vals,
                  x_embed,
                  colorbar_name,
                  colorbar_ticklabels=None,
                  markersize=MARKERSIZE,
                  show_colorbar=True,
                  colorbar_fontsize=COLORBAR_FONTSIZE,
                  colorbar_limits=None,
                  colorbar_ticks=None,
                  colorbar_tick_fontsize=COLORBAR_TICK_FONTSIZE,
                  colorbar_pos=[1.04, 0.2, 0.05, 0.6],
                  cmap='plasma',
                  axis_off=False):
    """General heatmap plotting helper function.
    """
    if isinstance(colorbar_limits, (list, tuple)):
        vmin, vmax = colorbar_limits[0], colorbar_limits[1]
    else:
        vmin = np.quantile(vals, 0.01)
        vmax = np.quantile(vals, 0.99)
        if vmin > 1e-3:
            vmin = round(vmin, 3)
        if vmax > 1e-3:
            vmax = round(vmax, 3)
    if x_embed.shape[1] == 2:
        ax.scatter(x_embed[:, 0],
                   x_embed[:, 1],
                   s=markersize,
                   c=vals,
                   cmap=cmap,
                   vmin=vmin,
                   vmax=vmax,
                   edgecolors='none')
    else:
        ax.scatter(x_embed[:, 0],
                   x_embed[:, 1],
                   x_embed[:, 2],
                   s=markersize,
                   c=vals,
                   cmap=cmap,
                   vmin=vmin,
                   vmax=vmax,
                   edgecolors='none')
    if show_colorbar:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        cax = ax.inset_axes(colorbar_pos)
        cbar = plt.colorbar(sm, ax=ax, cax=cax)
        cbar.ax.get_yaxis().labelpad = 5
        cbar.ax.set_ylabel(colorbar_name, rotation=270, fontsize=colorbar_fontsize)

        if colorbar_ticklabels is not None:
            if len(colorbar_ticklabels) == 2:
                cbar.ax.get_yaxis().labelpad = 3
            cbar.ax.set_yticklabels(colorbar_ticklabels, fontsize=colorbar_tick_fontsize)
            if colorbar_ticks is None:
                cbar.set_ticks(np.linspace(vmin, vmax, len(colorbar_ticklabels)))
            else:
                cbar.set_ticks(colorbar_ticks)
        else:
            if colorbar_ticks is None:
                cbar.set_ticks([vmin, vmax])
            else:
                cbar.set_ticks(colorbar_ticks)

    if axis_off:
        ax.axis("off")

    return ax


def histeq(x, perc=0.95, Nbin=101):
    x_ub = np.quantile(x, perc)
    x_lb = x.min()
    delta_x = (x_ub - x_lb)/(Nbin-1)
    bins = [x_lb+i*delta_x for i in range(Nbin)]+[x.max()]
    pdf_x, edges = np.histogram(x, bins, density=True)
    px, edges = np.histogram(x, bins, density=False)

    # Perform histogram equalization
    cdf = np.concatenate(([0], np.cumsum(px)))
    cdf = cdf/cdf[-1]
    x_out = np.zeros((len(x)))
    for i in range(Nbin):
        mask = (x >= bins[i]) & (x < bins[i+1])
        x_out[mask] = (cdf[i] + (x[mask]-bins[i])*pdf_x[i])*np.abs(x.max())
    return x_out


def plot_heatmap(vals,
                 x_embed,
                 width=WIDTH,
                 height=HEIGHT,
                 fix='width',
                 colorbar_name="",
                 colorbar_ticklabels=None,
                 markersize=MARKERSIZE,
                 colorbar_fontsize=COLORBAR_FONTSIZE,
                 colorbar_limits=None,
                 colorbar_ticks=None,
                 colorbar_tick_fontsize=COLORBAR_TICK_FONTSIZE,
                 real_aspect_ratio=True,
                 cmap='viridis',
                 save=None):
    """Plots a quantity as a heatmap.

    Args:
        vals (:class:`numpy.ndarray`):
            Values to be plotted as a heatmap, (N,).
        x_embed (:class:`numpy.ndarray`):
            2D coordinates for visualization, (N,2).
        colorbar_name (str, optional):
            Name shown next to the colorbar. Defaults to "Latent Time".
        colorbar_ticks (str, optional):
            Name shown on the colorbar axis. Defaults to None.
        markersize (int, optional):
            Marker size. Defaults to MARKERSIZE.
        colorbar_fontsize (int, optional):
            Font size for the colorbar label. Defaults to COLORBAR_FONTSIZE.
        colorbar_limits (list, optional):
            Colorbar limits. Defaults to None.
        colorbar_ticklabels (list, optional):
            Tick labels for the colorbar. Defaults to None.
        colorbar_tick_fontsize (int, optional):
            Font size for the colorbar ticks. Defaults to COLORBAR_TICK_FONTSIZE.
        real_aspect_ratio (bool, optional):
            Whether to use real aspect ratio for the plot. Defaults to False.
        cmap (str, optional):
            Colormap name. Defaults to 'viridis'.
        save (str, optional):
            Figure name for saving (including path). Defaults to None.
    """
    figsize = _set_figsize(x_embed, width, height, real_aspect_ratio, fix)
    fig, ax = plt.subplots(figsize=figsize)
    ax = _plot_heatmap(ax,
                       vals,
                       x_embed,
                       colorbar_name,
                       colorbar_ticklabels=colorbar_ticklabels,
                       markersize=markersize,
                       show_colorbar=True,
                       colorbar_fontsize=colorbar_fontsize,
                       colorbar_limits=colorbar_limits,
                       colorbar_ticks=colorbar_ticks,
                       colorbar_tick_fontsize=colorbar_tick_fontsize,
                       cmap=cmap,
                       axis_off=True)
    save_fig(fig, save)


def _find_mode(arr):
    """Find the mode of an array."""
    if len(arr) == 0:
        return None
    vals = np.unique(arr)
    counts = np.array([np.sum(arr == val) for val in vals])
    return vals[np.argmax(counts)]


def _sample_from_bins(x_embed, sample_per_bin, bin_size):
    """Sample from bins."""
    x_min, x_max = np.min(x_embed, 0), np.max(x_embed, 0)
    xy_grid = np.meshgrid(np.linspace(x_min[0], x_max[0], int((x_max[0]-x_min[0])/bin_size)+1),
                          np.linspace(x_min[1], x_max[1], int((x_max[1]-x_min[1])/bin_size)+1))
    xy_grid = np.stack([xy_grid[0].flatten(), xy_grid[1].flatten()]).T
    samples = []
    for xy in xy_grid:
        mask = (x_embed[:, 0] >= xy[0]) & (x_embed[:, 0] <= xy[0]+bin_size) \
            & (x_embed[:, 1] >= xy[1]) & (x_embed[:, 1] <= xy[1]+bin_size)
        num = int(np.sum(mask))
        if np.sum(mask) > sample_per_bin:
            samples.append(x_embed[mask][np.random.choice(num, sample_per_bin, replace=False)])
        else:
            samples.append(x_embed[mask])
    return np.concatenate(samples, 0)


def plot_heat_density(vals,
                      x_embed,
                      radius,
                      cell_labels,
                      real_aspect_ratio=True,
                      n_grid=100,
                      scale=1.5,
                      bw_adjust=0.5,
                      levels=10,
                      sample_per_bin=None,
                      width=WIDTH,
                      height=HEIGHT,
                      fix='width',
                      markersize=MARKERSIZE,
                      legend_fontsize=LEGEND_FONTSIZE,
                      markerscale=MARKERSCALE,
                      show_colorbar=True,
                      colorbar_name='',
                      colorbar_fontsize=COLORBAR_FONTSIZE,
                      colorbar_limits=None,
                      colorbar_ticklabels=None,
                      colorbar_ticks=None,
                      colorbar_tick_fontsize=7,
                      colorbar_labelpad=2,
                      colorbar_pos=[1.04, 0.2, 0.05, 0.6],
                      bbox_to_anchor=(-0.05, 1.0),
                      ncols=1,
                      axis_off=True,
                      cmap='Reds',
                      palette=None,
                      save=None):
    """Plots a heat density map.

    Args:
        vals (:class:`numpy.ndarray`): 
            Values to be plotted as a heatmap, (N,).
        x_embed (:class:`numpy.ndarray`):
            2D coordinates for visualization, (N,2).
        radius (float):
            Radius for grid interpolation using a ball graph.
        cell_labels (:class:`numpy.ndarray`):
            Cell labels for plotting.
        real_aspect_ratio (bool, optional):
            Whether to use real aspect ratio for the plot. Defaults to False.
        n_grid (int, optional):
            Number of grid points. Defaults to 100.
        scale (float, optional):
            Scale factor for distance threshold. Defaults to 1.5.
        bw_adjust (float, optional):
            Bandwidth adjustment for KDE. Defaults to 1.0.
        levels (int, optional):
            Number of levels for the KDE plot. Defaults to 10.
        sample_per_bin (int, optional):
            Number of samples per bin. Defaults to None.
        width (int, optional):
            Figure width. Defaults to WIDTH.
        height (int, optional):
            Figure height. Defaults to HEIGHT.
        fix (str, optional):
            Whether to fix the width or height of the figure. Defaults to 'width'.
        markersize (int, optional):
            Marker size. Defaults to MARKERSIZE.
        legend_fontsize (int, optional):
            Font size for the legend. Defaults to LEGEND_FONTSIZE.
        markerscale (float, optional):
            Marker scale. Defaults to MARKERSCALE.
        show_colorbar (bool, optional):
            Whether to show the colorbar. Defaults to True.
        colorbar_name (str, optional):
            Name shown next to the colorbar. Defaults to ''.
        colorbar_fontsize (int, optional):
            Font size for the colorbar label. Defaults to COLORBAR_FONTSIZE.
        colorbar_limits (list, optional):
            Colorbar limits. Defaults to None.
        colorbar_ticklabels (list, optional):
            Tick labels for the colorbar. Defaults to None.
        colorbar_ticks (list, optional):
            Tick positions for the colorbar. Defaults to None.
        colorbar_tick_fontsize (int, optional):
            Font size for the colorbar ticks. Defaults to 7.
        colorbar_labelpad (int, optional):
            Padding for the colorbar label. Defaults to 2.
        colorbar_pos (list, optional):
            Position of the colorbar. Defaults to [1.04, 0.2, 0.05, 0.6].
        bbox_to_anchor (tuple, optional):
            Bounding box for the legend. Defaults to (-0.05, 1.0).
        ncols (int, optional):
            Number of columns for the legend. Defaults to 1.
        axis_off (bool, optional):
            Whether to turn off the axis. Defaults to True.
        cmap (str, optional):
            Colormap name. Defaults to 'Reds'.
        palette (list, optional):
            List of colors for cell types. Defaults to None.
        save (str, optional):
            Figure name for saving (including path). Defaults to None.
    """
    # Grid interpolation
    bt = BallTree(x_embed)
    x = np.linspace(x_embed[:, 0].min(), x_embed[:, 0].max(), n_grid)
    y = np.linspace(x_embed[:, 1].min(), x_embed[:, 1].max(), n_grid)
    xgrid, ygrid = np.meshgrid(x, y)
    xgrid, ygrid = xgrid.flatten(), ygrid.flatten()
    xy_grid = np.stack([xgrid, ygrid]).T

    # Find distance thredshold
    out = bt.query_radius(x_embed, radius, return_distance=True)
    _, dist = out[0], out[1]
    dist = np.concatenate(dist)
    dist_thred = dist.mean() * scale

    # Find neighbors
    out = bt.query_radius(xy_grid, radius, return_distance=True)
    spatial_nbs, dist_grid = out[0], out[1]
    mid_dist_grid = np.array([np.median(x) for x in dist_grid])
    mask = mid_dist_grid <= dist_thred

    vals_grid = np.array([np.mean(vals[nbs]) for nbs in spatial_nbs])
    labels_grid = np.array([_find_mode(cell_labels[nbs]) for nbs in spatial_nbs])
    
    figsize = _set_figsize(x_embed, width, height, real_aspect_ratio, fix)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Heatmap
    df = pd.DataFrame({'x': xgrid[mask], 'y': ygrid[mask], 'kind': labels_grid[mask]})
    
    kdeplot(data=df, x='x', y='y',
            weights=vals_grid[mask],
            common_norm=True,
            fill=True,
            cmap=cmap,
            bw_adjust=bw_adjust,
            levels=levels,
            ax=ax)
    
    # color bar
    if show_colorbar:
        if isinstance(colorbar_limits, (list, tuple)):
            vmin, vmax = colorbar_limits[0], colorbar_limits[1]
        else:
            vmin, vmax = round(vals.min(), 2), round(vals.max(), 2)
        ax = _set_colorbar(ax,
                           vmin,
                           vmax,
                           cmap,
                           colorbar_name,
                           colorbar_ticklabels=colorbar_ticklabels,
                           colorbar_fontsize=colorbar_fontsize,
                           colorbar_ticks=colorbar_ticks,
                           colorbar_pos=colorbar_pos,
                           colorbar_tick_fontsize=colorbar_tick_fontsize,
                           labelpad=colorbar_labelpad)

    cell_types = np.unique(cell_labels)
    palette = get_colors(len(cell_types)) if palette is None else palette
    for i, cell_type in enumerate(cell_types):
        if sample_per_bin is not None:
            x_sample = _sample_from_bins(x_embed[cell_labels == cell_type], sample_per_bin, radius)
            ax.scatter(x_sample[:, 0],
                       x_sample[:, 1],
                       edgecolors='none',
                       s=markersize,
                       color=palette[i],
                       label=cell_type)
        else:
            ax.scatter(x_embed[cell_labels == cell_type, 0],
                       x_embed[cell_labels == cell_type, 1],
                       edgecolors='none',
                       s=markersize,
                       color=palette[i],
                       label=cell_type)
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles,
                    labels,
                    ncol=ncols,
                    fontsize=legend_fontsize,
                    markerscale=markerscale,
                    bbox_to_anchor=bbox_to_anchor,
                    loc='upper center')
    if axis_off:
        ax.axis('off')
    save_fig(fig, save, bbox_extra_artists=(lgd,))


def plot_3d_heatmap(vals,
                    x_embed,
                    figsize=(6, 4),
                    angle=(15, 45),
                    colorbar_name="",
                    colorbar_ticklabels=None,
                    markersize=MARKERSIZE,
                    colorbar_fontsize=COLORBAR_FONTSIZE,
                    colorbar_limits=None,
                    colorbar_ticks=None,
                    colorbar_tick_fontsize=COLORBAR_TICK_FONTSIZE,
                    real_aspect_ratio=False,
                    cmap='viridis',
                    show_background=False,
                    marker_amp=3.0,
                    show_colorbar=True,
                    colorbar_pos=[1.04, 0.2, 0.05, 0.6],
                    tick_fontsize=TICK_FONTSIZE,
                    axis_off=False,
                    label_off=False,
                    ticks_off=False,
                    zoom=1.0,
                    embed='Spatial',
                    save=None):
    """Plots a quantity as a heatmap.

    Args:
        vals (:class:`numpy.ndarray`):
            Values to be plotted as a heatmap, (N,).
        x_embed (:class:`numpy.ndarray`):
            3D coordinates for visualization, (N,2).
        figsize (tuple, optional):
            Figure size. Defaults to (6, 4).
        angle (tuple, optional):
            Angle for viewing the plot. Defaults to (15, 45).
        colorbar_name (str, optional):
            Name shown next to the colorbar. Defaults to "Latent Time".
        colorbar_ticklabels (list, optional):
            Tick labels for the colorbar. Defaults to None.
        markersize (int, optional):
            Marker size. Defaults to MARKERSIZE.
        colorbar_fontsize (int, optional):
            Font size for the colorbar. Defaults to COLORBAR_FONTSIZE.
        colorbar_limits (list, optional):
            Colorbar limits. Defaults to None.
        colorbar_ticks (str, optional):
            Name shown on the colorbar axis. Defaults to None.
        colorbar_tick_fontsize (int, optional):
            Font size for the colorbar ticks. Defaults to COLORBAR_TICK_FONTSIZE.
        real_aspect_ratio (bool, optional):
            Whether to use real aspect ratio for the plot. Defaults to False.
        cmap (str, optional):
            Colormap name. Defaults to 'viridis'.
        show_background (bool, optional):
            Whether to show the background. Defaults to False.
        marker_amp (float, optional):
            Amplify the marker size for non-background dots. Defaults to 3.0.
        show_colorbar (bool, optional):
            Whether to show the colorbar. Defaults to True.
        colorbar_pos (list, optional): 
            Position of the colorbar. Defaults to [1.04, 0.2, 0.05, 0.6].
        axis_off (bool, optional):
            Whether to turn off the axis. Defaults to False.
        label_off (bool, optional):
            Whether to turn off the axis labels. Defaults to False.
        ticks_off (bool, optional):
            Whether to turn off the axis ticks. Defaults to False.
        zoom (float, optional):
            Zoom factor for the plot. Defaults to 1.0.
        embed (str, optional):
            Embedding name. Defaults to 'Spatial'.
        save (str, optional):
            Figure name for saving (including path). Defaults to None.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    ax.view_init(angle[0], angle[1])

    if isinstance(colorbar_limits, (list, tuple)):
        vmin, vmax = colorbar_limits[0], colorbar_limits[1]
    else:
        vmin = np.quantile(vals, 0.01)
        vmax = np.quantile(vals, 0.99)
        if vmin > 1e-3:
            vmin = round(vmin, 3)
        if vmax > 1e-3:
            vmax = round(vmax, 3)
    # emphasize the color of high values
    mask = vals > np.quantile(vals, 0.05)
    if show_background:
        ax.scatter(x_embed[~mask, 0],
                   x_embed[~mask, 1],
                   x_embed[~mask, 2],
                   s=markersize,
                   alpha=0.1,
                   color='gray',
                   vmin=vmin,
                   vmax=vmax,
                   edgecolors='none')
        ax.scatter(x_embed[mask, 0],
                   x_embed[mask, 1],
                   x_embed[mask, 2],
                   s=markersize * marker_amp,
                   c=vals[mask],
                   cmap=cmap,
                   vmin=vmin,
                   vmax=vmax,
                   edgecolors='none')
    else:
        ax.scatter(x_embed[:, 0],
                   x_embed[:, 1],
                   x_embed[:, 2],
                   s=markersize,
                   c=vals,
                   cmap=cmap,
                   vmin=vmin,
                   vmax=vmax,
                   edgecolors='none')
    if show_colorbar:
        _set_colorbar(ax,
                      vmin,
                      vmax,
                      cmap,
                      colorbar_name,
                      colorbar_ticklabels=colorbar_ticklabels,
                      colorbar_fontsize=colorbar_fontsize,
                      colorbar_ticks=colorbar_ticks,
                      colorbar_pos=colorbar_pos,
                      colorbar_tick_fontsize=colorbar_tick_fontsize)
    if not label_off:
        ax.set_xlabel(f'{embed} 1', fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(f'{embed} 2', fontsize=LABEL_FONTSIZE)
        ax.set_zlabel(f'{embed} 3', fontsize=LABEL_FONTSIZE)
    xrange, yrange, zrange = np.ptp(x_embed[:, 0]), np.ptp(x_embed[:, 1]), np.ptp(x_embed[:, 2])
    ax.set_xlim(x_embed[:, 0].min(), x_embed[:, 0].max())
    ax.set_ylim(x_embed[:, 1].min(), x_embed[:, 1].max())
    ax.set_zlim(x_embed[:, 2].min(), x_embed[:, 2].max())
    # xmin, xmax = x_embed[:, 0].min(), x_embed[:, 0].max()
    # ymin, ymax = x_embed[:, 1].min(), x_embed[:, 1].max()
    # zmin, zmax = x_embed[:, 2].min(), x_embed[:, 2].max()
    # ax.set_xticks([xmin, xmax], [f'{xmin:.2f}', f'{xmax:.2f}'])
    # ax.set_yticks([ymin, ymax], [f'{ymin:.2f}', f'{ymax:.2f}'])
    # ax.set_zticks([zmin, zmax], [f'{zmin:.2f}', f'{zmax:.2f}'])
    if real_aspect_ratio:
        ax.set_box_aspect((xrange, yrange, zrange), zoom=zoom)

    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    if axis_off:
        ax.axis("off")
    elif ticks_off:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    plt.tight_layout()
    if save is not None:
        save_fig(fig, save)


def plot_time(t_latent,
              x_embed,
              width=WIDTH,
              height=HEIGHT,
              fix='width',
              real_aspect_ratio=True,
              colorbar_name='Latent Time',
              colorbar_ticklabels=['early', 'late'],
              markersize=MARKERSIZE,
              show_colorbar=True,
              colorbar_fontsize=COLORBAR_FONTSIZE,
              colorbar_limits=None,
              colorbar_ticks=None,
              colorbar_tick_fontsize=COLORBAR_TICK_FONTSIZE,
              colorbar_pos=[1.04, 0.2, 0.05, 0.6],
              cmap='viridis',
              axis_off=False,
              save=None):
    """Plots mean cell time as a heatmap.

    Args:
        t_latent (`numpy.ndarray`):
            Mean latent time, (N,)
        x_embed (`numpy.ndarray`):
            2D coordinates for visualization, (N,2)
        width (int, optional):
            Figure width. Defaults to WIDTH.
        height (int, optional):
            Figure height. Defaults to HEIGHT.
        fix (str, optional):
            Whether to fix the width or height of the figure. Defaults to 'width'.
        real_aspect_ratio (bool, optional):
            Whether to use real aspect ratio for the plot. Defaults to True.
        colorbar_name (str, optional):
            Name shown next to the colorbar. Defaults to 'Latent Time'.
        colorbar_ticklabels (list, optional):
            Tick labels for the colorbar. Defaults to ['early', 'late'].
        markersize (int, optional):
            Marker size. Defaults to MARKERSIZE.
        show_colorbar (bool, optional):
            Whether to show the colorbar. Defaults to True.
        colorbar_fontsize (int, optional):
            Font size for the colorbar label. Defaults to COLORBAR_FONTSIZE.
        colorbar_limits (list, optional):
            Colorbar limits. Defaults to None.
        colorbar_ticks (list, optional):
            Tick positions for the colorbar. Defaults to None.
        colorbar_tick_fontsize (int, optional):
            Font size for the colorbar ticks. Defaults to COLORBAR_TICK_FONTSIZE.
        colorbar_pos (list, optional):
            Position of the colorbar. Defaults to [1.04, 0.2, 0.05, 0.6].
        cmap (str, optional):
            Colormap name. Defaults to 'viridis'.
        axis_off (bool, optional):
            Whether to turn off the axis. Defaults to False.
        save (str, optional):
            Figure name for saving (including path). Defaults to None.
    """
    figsize = _set_figsize(x_embed, width, height, real_aspect_ratio, fix)
    fig, ax = plt.subplots(figsize=figsize)
    _plot_heatmap(ax,
                  t_latent,
                  x_embed,
                  colorbar_name,
                  colorbar_ticklabels,
                  markersize=markersize,
                  show_colorbar=show_colorbar,
                  colorbar_fontsize=colorbar_fontsize,
                  colorbar_limits=colorbar_limits,
                  colorbar_ticks=colorbar_ticks,
                  colorbar_tick_fontsize=colorbar_tick_fontsize,
                  colorbar_pos=colorbar_pos,
                  cmap=cmap,
                  axis_off=True)
    save_fig(fig, save)


def plot_time_var(std_t,
                  x_embed,
                  t=None,
                  hist_eq=True,
                  width=WIDTH,
                  height=HEIGHT,
                  fix='width',
                  real_aspect_ratio=True,
                  colorbar_name='Time Variance',
                  colorbar_ticklabels=['low', 'high'],
                  markersize=MARKERSIZE,
                  show_colorbar=True,
                  colorbar_fontsize=COLORBAR_FONTSIZE,
                  colorbar_limits=None,
                  colorbar_ticks=None,
                  colorbar_tick_fontsize=COLORBAR_TICK_FONTSIZE,
                  colorbar_pos=[1.04, 0.2, 0.05, 0.6],
                  cmap='viridis',
                  axis_off=False,
                  save=None):
    """Plots cell time coefficient of variation as a heatmap.

    Args:
        std_t (:class:`numpy.ndarray`):
            Standard deviation of latent time, (N,)
        x_embed (:class:`numpy.ndarray`):
            2D coordinates for visualization, (N,2)
        t (:class:`numpy.ndarray`, optional):
            Mean latent time. Defaults to None.
        hist_eq (bool, optional):
            Whether to perform histogram equalization. Defaults to True.
        width (int, optional):
            Figure width. Defaults to WIDTH.
        height (int, optional):
            Figure height. Defaults to HEIGHT.
        fix (str, optional):
            Whether to fix the width or height of the figure. Defaults to 'width'.
        real_aspect_ratio (bool, optional):
            Whether to use real aspect ratio for the plot. Defaults to True.
        colorbar_name (str, optional):
            Name shown next to the colorbar. Defaults to 'Latent Time'.
        colorbar_ticklabels (list, optional):
            Tick labels for the colorbar. Defaults to ['early', 'late'].
        markersize (int, optional):
            Marker size. Defaults to MARKERSIZE.
        show_colorbar (bool, optional):
            Whether to show the colorbar. Defaults to True.
        colorbar_fontsize (int, optional):
            Font size for the colorbar label. Defaults to COLORBAR_FONTSIZE.
        colorbar_limits (list, optional):
            Colorbar limits. Defaults to None.
        colorbar_ticks (list, optional):
            Tick positions for the colorbar. Defaults to None.
        colorbar_tick_fontsize (int, optional):
            Font size for the colorbar ticks. Defaults to COLORBAR_TICK_FONTSIZE.
        colorbar_pos (list, optional):
            Position of the colorbar. Defaults to [1.04, 0.2, 0.05, 0.6].
        cmap (str, optional):
            Colormap name. Defaults to 'viridis'.
        axis_off (bool, optional):
            Whether to turn off the axis. Defaults to False.
        save (str, optional):
            Figure name for saving (including path). Defaults to None.
    """
    t_norm = np.ones((std_t.shape)) if t is None else np.abs(t) + 1e-10
    diff_entropy = np.log(std_t/t_norm)+0.5*(1+np.log(2*np.pi))
    if hist_eq:
        diff_entropy = histeq(diff_entropy, Nbin=len(diff_entropy)//50)

    figsize = _set_figsize(x_embed, width, height, real_aspect_ratio, fix)
    fig, ax = plt.subplots(figsize=figsize)
    ax = _plot_heatmap(ax,
                       diff_entropy,
                       x_embed,
                       colorbar_name,
                       colorbar_ticklabels,
                       markersize=markersize,
                       show_colorbar=show_colorbar,
                       colorbar_fontsize=colorbar_fontsize,
                       colorbar_limits=colorbar_limits,
                       colorbar_ticks=colorbar_ticks,
                       colorbar_tick_fontsize=colorbar_tick_fontsize,
                       colorbar_pos=colorbar_pos,
                       cmap=cmap,
                       axis_off=True)
    save_fig(fig, save)


def plot_state_var(std_z,
                   x_embed,
                   z=None,
                   hist_eq=True,
                   width=WIDTH,
                   height=HEIGHT,
                   fix='width',
                   real_aspect_ratio=True,
                   colorbar_name='State Uncertainty',
                   colorbar_ticklabels=['low', 'high'],
                   markersize=MARKERSIZE,
                   show_colorbar=True,
                   colorbar_fontsize=COLORBAR_FONTSIZE,
                   colorbar_limits=None,
                   colorbar_ticks=None,
                   colorbar_tick_fontsize=COLORBAR_TICK_FONTSIZE,
                   colorbar_pos=[1.04, 0.2, 0.05, 0.6],
                   cmap='viridis',
                   axis_off=False,
                   save=None):
    """Plots cell state variance (in the form of coefficient of variation) as a heatmap.

    Args:
        std_z (:class:`numpy.ndarray`):
            Standard deviation of cell state, assuming diagonal covariance, (N, dim z)
        x_embed (:class:`numpy.ndarray`):
            2D coordinates for visualization, (N, 2)
        z (:class:`numpy.ndarray`, optional):
            Mean cell state, (N, dim z). Defaults to None.
        hist_eq (bool, optional):
            Whether to perform histogram equalization. Defaults to True.
        width (int, optional):
            Figure width. Defaults to WIDTH.
        height (int, optional):
            Figure height. Defaults to HEIGHT.
        fix (str, optional):
            Whether to fix the width or height of the figure. Defaults to 'width'.
        real_aspect_ratio (bool, optional):
            Whether to use real aspect ratio for the plot. Defaults to True.
        colorbar_name (str, optional):
            Name shown next to the colorbar. Defaults to 'Latent Time'.
        colorbar_ticklabels (list, optional):
            Tick labels for the colorbar. Defaults to ['early', 'late'].
        markersize (int, optional):
            Marker size. Defaults to MARKERSIZE.
        show_colorbar (bool, optional):
            Whether to show the colorbar. Defaults to True.
        colorbar_fontsize (int, optional):
            Font size for the colorbar label. Defaults to COLORBAR_FONTSIZE.
        colorbar_limits (list, optional):
            Colorbar limits. Defaults to None.
        colorbar_ticks (list, optional):
            Tick positions for the colorbar. Defaults to None.
        colorbar_tick_fontsize (int, optional):
            Font size for the colorbar ticks. Defaults to COLORBAR_TICK_FONTSIZE.
        colorbar_pos (list, optional):
            Position of the colorbar. Defaults to [1.04, 0.2, 0.05, 0.6].
        cmap (str, optional):
            Colormap name. Defaults to 'viridis'.
        axis_off (bool, optional):
            Whether to turn off the axis. Defaults to False.
        save (str, optional):
            Figure name for saving (including path). Defaults to None.
    """
    z_norm = np.ones((std_z.shape)) if z is None else np.linalg.norm(z, axis=1).reshape(-1, 1) + 1e-10
    diff_entropy = np.sum(np.log(std_z/z_norm), 1) + 0.5*std_z.shape[1]*(1+np.log(2*np.pi))
    if hist_eq:
        diff_entropy = histeq(diff_entropy, Nbin=len(diff_entropy)//50)

    figsize = _set_figsize(x_embed, width, height, real_aspect_ratio, fix)
    fig, ax = plt.subplots(figsize=figsize)
    ax = _plot_heatmap(ax,
                       diff_entropy,
                       x_embed,
                       colorbar_name,
                       colorbar_ticklabels,
                       markersize=markersize,
                       show_colorbar=show_colorbar,
                       colorbar_fontsize=colorbar_fontsize,
                       colorbar_limits=colorbar_limits,
                       colorbar_ticks=colorbar_ticks,
                       colorbar_tick_fontsize=colorbar_tick_fontsize,
                       colorbar_pos=colorbar_pos,
                       cmap=cmap,
                       axis_off=True)
    save_fig(fig, save)


def plot_phase_axis(ax,
                    u,
                    s,
                    marker='o',
                    a=1.0,
                    downsample=1,
                    markersize=10,
                    linewidths=0.5,
                    labels=None,
                    legends=None,
                    title=None,
                    title_fontsize=7,
                    tick_fontsize=7,
                    show_legend=False,
                    palette=None):
    """Plot phase in a subplot of a figure."""
    if legends is not None:
        n_type = len(legends)
        types = np.array(range(n_type))
        if palette is None:
            palette = get_colors(len(types))
    elif labels is not None:
        types = np.unique(labels)
        n_type = len(types)
        if palette is None:
            palette = get_colors(len(types))
    try:
        if labels is None:
            ax.scatter(s[::downsample], u[::downsample],
                       marker=marker,
                       color='k',
                       s=markersize,
                       linewidths=linewidths,
                       edgecolors='none',
                       alpha=a)
        elif legends is None:
            for type_int in types:
                mask = labels == type_int
                if np.any(mask):
                    ax.scatter(s[mask][::downsample],
                               u[mask][::downsample],
                               marker=marker,
                               color=palette[type_int % n_type],
                               s=markersize,
                               linewidths=linewidths,
                               edgecolors='none',
                               alpha=a)
        else:
            for i, type_int in enumerate(types):  # type_int: label index, labels are cell types
                mask = labels == type_int
                if np.any(mask):
                    if show_legend:
                        ax.scatter(s[mask][::downsample],
                                   u[mask][::downsample],
                                   marker=marker,
                                   color=palette[type_int % n_type],
                                   s=markersize,
                                   linewidths=linewidths,
                                   edgecolors='none',
                                   alpha=a,
                                   label=legends[type_int])
                    else:
                        ax.scatter(s[mask][::downsample],
                                   u[mask][::downsample],
                                   marker=marker,
                                   s=markersize,
                                   linewidths=linewidths,
                                   edgecolors='none',
                                   color=palette[type_int % n_type],
                                   alpha=a)
                elif show_legend:
                    ax.scatter([np.nan],
                               [np.nan],
                               s=markersize,
                               marker=marker,
                               edgecolors='none',
                               color=palette[type_int % n_type],
                               alpha=a,
                               label=legends[type_int])
    except TypeError:
        return ax

    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)

    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    return ax


def plot_phase_grid(n_rows,
                    n_cols,
                    gene_list,
                    U,
                    S,
                    Labels,
                    Legends,
                    Uhat={},
                    Shat={},
                    Labels_demo={},
                    width=WIDTH,
                    height=HEIGHT,
                    alpha=0.2,
                    downsample=1,
                    obs_marker="o",
                    pred_marker="x",
                    markersize=MARKERSIZE,
                    linewidths=0.5,
                    title_fontsize=FONTSIZE,
                    show_legend=True,
                    legend_fontsize=LEGEND_FONTSIZE,
                    legend_loc="upper right",
                    bbox_to_anchor=None,
                    label_fontsize=FONTSIZE,
                    tick_fontsize=TICK_FONTSIZE,
                    palette=None,
                    hspace=0.3,
                    wspace=0.12,
                    markerscale=3.0,
                    path='figures',
                    figname=None,
                    save_format='png',
                    **kwargs):
    """Plot the phase portrait of a list of genes in an [n_rows x n_cols] grid.
    Cells are colored according to their dynamical state or cell type.
    If n_rows*n_cols < number of genes, the last few genes will be ignored.
    If n_rows*n_cols > number of genes, the last few subplots will be blank.

    Args:
        n_rows (int):
            Number of rows of the grid plot.
        n_cols (int):
            Number of columns of the grid plot.
        gene_list (list[str]):
            Genes to plot.
        U (:class:`numpy.ndarray`):
            Unspliced count matrices. The gene dimension should equal len(gene_list).
        S (:class:`numpy.ndarray`):
            Spliced count matrices. The gene dimension should equal len(gene_list).
        Labels (dict):
            Keys are method names and values are (N) cell annotations
            For the regular ODE, this can be induction/repression annotation.
            Otherwise, it's usually just the cell type annotation.
        Legends (dict):
            Keys are method names and values are the legend names to show.
            If the labels are phase labels, then the legends are usually
            {'off', induction', 'repression'}.
            If the labels are cell type annotations, the legends will be the unique
            cell type names.
        Uhat (dict, optional):
            Predicted unspliced counts.
            Keys are method names and values are arrays of size (N_pred, N_gene).
            Notice that N_pred is not necessarily the number of cells.
            This could happen if we want to save computational cost and evaluate
            the ODE just at a user-defined number of time points. Defaults to {}.
        Shat (dict, optional):
            Predicted spliced counts, similar to Uhat. Defaults to {}.
        Labels_demo (dict, optional):
            Keys are method names and values are arrays of size (N_pred).
            This is the annotation for the predictions.. Defaults to {}.
        width (int, optional):
            Width of a subplot. Defaults to WIDTH.
        height (int, optional):
            Height of a subplot. Defaults to HEIGHT.
        alpha (float, optional):
            Transparency of the data points. Defaults to 0.2.
        downsample (int, optional):
            Down-sampling factor to display the data points. Defaults to 1.
        obs_marker (str, optional):
            Marker used for plotting observations. Defaults to 'o'.
        pred_marker (str, optional):
            Marker used for plotting predictions. Defaults to 'x'.
        legend_fontsize (int/float, optional):
            Defaults to LEGEND_FONTSIZE.
        palette (str, optional):
            User-defined colormap for cell labels. Defaults to None.
        hspace (float, optional):
            Height distance proportion between subplots. Defaults to 0.3.
        wspace (float, optional):
            Width distance proportion between subplots. Defaults to 0.12.
        markerscale (float, optional):
            Marker scale for the legend. Defaults to 3.0.
        path (str, optional):
            Path to the saved figure. Defaults to 'figures'.
        figname (_type_, optional):
            Name of the saved figure, without format at the end. Defaults to None.
        save_format (str, optional):
            Figure format. Defaults to 'png'.

    """
    D = downsample
    methods = list(Uhat.keys())

    M = max(1, len(methods))

    # Detect whether multiple figures are needed
    Nfig = len(gene_list) // (n_rows*n_cols)
    if Nfig*n_rows*n_cols < Nfig:
        Nfig += 1

    if label_fontsize is None:
        label_fontsize = width * height
    for i_fig in range(Nfig):
        fig_phase, ax_phase = plt.subplots(n_rows, M*n_cols, figsize=(width * M * n_cols + 1.0, height * n_rows), facecolor='white')
        if n_rows == 1 and M * n_cols == 1:  # Single Gene, Single Method
            labels = Labels[methods[0]]
            if labels is not None:
                if labels.ndim == 2:
                    labels = labels[:, i_fig]
            title = f"{gene_list[i_fig]} ({methods[0]})"
            ax_phase = plot_phase_axis(ax_phase,
                                       U[:, i_fig],
                                       S[:, i_fig],
                                       obs_marker,
                                       alpha,
                                       downsample,
                                       markersize,
                                       linewidths,
                                       labels,
                                       Legends[methods[0]],
                                       title,
                                       title_fontsize,
                                       tick_fontsize,
                                       show_legend=show_legend,
                                       palette=palette)
            try:
                ax_phase = plot_phase_axis(ax_phase,
                                           Uhat[methods[0]][:, i_fig],
                                           Shat[methods[0]][:, i_fig],
                                           pred_marker,
                                           1.0,
                                           downsample,
                                           markersize,
                                           linewidths,
                                           Labels_demo[methods[0]],
                                           Legends[methods[0]],
                                           title,
                                           title_fontsize,
                                           tick_fontsize,
                                           show_legend=False,
                                           palette=palette)
            except (KeyError, TypeError):
                print("[** Warning **]: Skip plotting the prediction because of key value error or invalid data type.")
                pass
            ax_phase.set_xlabel("S", fontsize=label_fontsize)
            ax_phase.set_ylabel("U", fontsize=label_fontsize)
        elif n_rows == 1:  # Single Gene, Multiple Method
            for j in range(min(n_cols, len(gene_list) - i_fig * n_cols)):  # gene
                for k, method in enumerate(methods):  # method
                    labels = Labels[method]
                    if labels is not None:
                        if labels.ndim == 2:
                            labels = labels[:, i_fig*n_cols+j]
                    title = f"{gene_list[i_fig*n_cols+j]} ({method})"
                    ax_phase[M*j+k] = plot_phase_axis(ax_phase[M*j+k],
                                                      U[:, i_fig*n_cols+j],
                                                      S[:, i_fig*n_cols+j],
                                                      obs_marker,
                                                      alpha,
                                                      downsample,
                                                      markersize,
                                                      linewidths,
                                                      labels,
                                                      Legends[method],
                                                      title,
                                                      title_fontsize,
                                                      tick_fontsize,
                                                      show_legend=show_legend,
                                                      palette=palette)
                    try:
                        ax_phase[M*j+k] = plot_phase_axis(ax_phase[M*j+k],
                                                          Uhat[method][:, i_fig*n_cols+j],
                                                          Shat[method][:, i_fig*n_cols+j],
                                                          pred_marker,
                                                          1.0,
                                                          downsample,
                                                          markersize,
                                                          linewidths,
                                                          Labels_demo[method],
                                                          Legends[method],
                                                          title,
                                                          title_fontsize,
                                                          tick_fontsize,
                                                          show_legend=False,
                                                          palette=palette)
                    except (KeyError, TypeError):
                        print("[** Warning **]: "
                              "Skip plotting the prediction because of key value error or invalid data type.")
                        pass
                    ax_phase[M*j+k].set_xlabel("S", fontsize=label_fontsize)
                    ax_phase[M*j+k].set_ylabel("U", fontsize=label_fontsize)
        elif M * n_cols == 1:  # Multiple Gene, Single Method
            for i in range(min(n_rows, len(gene_list) - i_fig * n_rows)):
                labels = Labels[methods[0]]
                if labels is not None:
                    if labels.ndim == 2:
                        labels = labels[:, i_fig * n_rows + i]
                title = f"{gene_list[i_fig*n_rows+i]} ({methods[0]})"
                ax_phase[i] = plot_phase_axis(ax_phase[i],
                                              U[:, i_fig*n_rows+i],
                                              S[:, i_fig*n_rows+i],
                                              obs_marker,
                                              alpha,
                                              downsample,
                                              markersize,
                                              linewidths,
                                              labels,
                                              Legends[methods[0]],
                                              title,
                                              title_fontsize,
                                              tick_fontsize,
                                              show_legend=show_legend,
                                              palette=palette)
                try:
                    ax_phase[i] = plot_phase_axis(ax_phase[i],
                                                  Uhat[methods[0]][:, i_fig*n_rows+i],
                                                  Shat[methods[0]][:, i_fig*n_rows+i],
                                                  pred_marker,
                                                  1.0,
                                                  downsample,
                                                  markersize,
                                                  linewidths,
                                                  Labels_demo[methods[0]],
                                                  Legends[methods[0]],
                                                  title,
                                                  title_fontsize,
                                                  tick_fontsize,
                                                  show_legend=False,
                                                  palette=palette)
                except (KeyError, TypeError):
                    print("[** Warning **]: "
                          "Skip plotting the prediction because of key value error or invalid data type.")
                    pass
                ax_phase[i].set_xlabel("S", fontsize=label_fontsize)
                ax_phase[i].set_ylabel("U", fontsize=label_fontsize)
        else:
            for i in range(n_rows):
                for j in range(n_cols):  # i, j: row and column gene index
                    idx = i_fig * n_cols * n_rows + i * n_cols + j
                    if idx >= len(gene_list):
                        break
                    for k, method in enumerate(methods):
                        labels = Labels[method]
                        if labels is not None:
                            if labels.ndim == 2:
                                labels = labels[:, idx]
                        title = f"{gene_list[idx]} ({method})"
                        ax_phase[i, M * j + k] = plot_phase_axis(ax_phase[i, M * j + k],
                                                                 U[:, idx],
                                                                 S[:, idx],
                                                                 obs_marker,
                                                                 alpha,
                                                                 downsample,
                                                                 markersize,
                                                                 linewidths,
                                                                 labels,
                                                                 Legends[method],
                                                                 title,
                                                                 title_fontsize,
                                                                 tick_fontsize,
                                                                 show_legend=show_legend,
                                                                 palette=palette)
                        try:
                            ax_phase[i, M * j + k] = plot_phase_axis(ax_phase[i, M * j + k],
                                                                     Uhat[method][:, idx],
                                                                     Shat[method][:, idx],
                                                                     pred_marker,
                                                                     1.0,
                                                                     downsample,
                                                                     markersize,
                                                                     linewidths,
                                                                     Labels_demo[method],
                                                                     Legends[method],
                                                                     title,
                                                                     title_fontsize,
                                                                     tick_fontsize,
                                                                     show_legend=False,
                                                                     palette=palette)
                        except (KeyError, TypeError):
                            print("[** Warning **]:"
                                  "Skip plotting the prediction because of key value error or invalid data type.")
                            pass
                        ax_phase[i, M * j + k].set_xlabel("S", fontsize=label_fontsize)
                        ax_phase[i, M * j + k].set_ylabel("U", fontsize=label_fontsize)
        
        if show_legend:
            if n_rows == 1 and M*n_cols == 1:
                handles, labels = ax_phase.get_legend_handles_labels()
            elif ax_phase.ndim == 1:
                handles, labels = ax_phase[0].get_legend_handles_labels()
            else:
                handles, labels = ax_phase[0, 0].get_legend_handles_labels()
            n_label = len(Legends[methods[0]])

            if legend_fontsize is None:
                legend_fontsize = min(int(10*n_rows), 300*n_rows/n_label)
            if bbox_to_anchor is None:
                l_indent = 1 - 0.02/n_rows
                bbox_to_anchor = (-0.03/n_cols, l_indent)
            lgd = fig_phase.legend(handles,
                                   labels,
                                   fontsize=legend_fontsize,
                                   markerscale=markerscale,
                                   bbox_to_anchor=bbox_to_anchor,
                                   loc=legend_loc)

        fig_phase.subplots_adjust(hspace=hspace, wspace=wspace)
        fig_phase.tight_layout()

        save = None if (path is None or figname is None) else f'{path}/{figname}_{i_fig+1}.{save_format}'
        if show_legend:
            save_fig(fig_phase, save, (lgd,))
        save_fig(fig_phase, save)


def sample_scatter_plot(x, downsample, n_bins=20):
    """Sample cells for a scatter plot."""
    idx_downsample = []
    n_sample = max(1, len(x)//downsample)
    if n_bins > n_sample:
        n_bins = n_sample
    sample_per_bin = n_sample // n_bins
    n_res = n_sample - sample_per_bin * n_bins

    edges = np.linspace(x.min(), x.max(), n_bins+1)

    for i in range(len(edges)-1):
        idx_pool = np.where((x >= edges[i]) & (x <= edges[i+1]))[0]
        if len(idx_pool) > sample_per_bin + int(i+1 <= n_res):
            idx_downsample.extend(np.random.choice(idx_pool, sample_per_bin+int(i + 1 <= n_res)))
        else:  # fewer samples in the bin than needed
            idx_downsample.extend(idx_pool)

    return np.array(idx_downsample).astype(int)


def plot_sig_axis(ax,
                  t,
                  x,
                  labels=None,
                  legends=None,
                  marker='o',
                  markersize=5,
                  linewidths=0.5,
                  a=1.0,
                  downsample=1,
                  show_legend=False,
                  palette=None,
                  title=None,
                  title_fontsize=30):
    """Plot a modality versus time in a subplot."""
    if labels is None or legends is None:
        ax.scatter(t[::downsample], x[::downsample], marker, markersize=5, color='k', alpha=a)
    else:
        colors = (palette if isinstance(palette, np.ndarray) or isinstance(palette, list)
                  else get_colors(len(legends)))
        cell_types_int = np.unique(labels)
        for i, type_int in enumerate(cell_types_int):
            mask = labels == type_int
            if np.any(mask):
                idx_sample = sample_scatter_plot(x[mask], downsample)
                if show_legend:
                    ax.scatter(t[mask][idx_sample],
                               x[mask][idx_sample],
                               s=markersize,
                               marker=marker,
                               linewidth=linewidths,
                               color=colors[i % len(colors)],
                               alpha=a,
                               label=legends[i])
                else:
                    ax.scatter(t[mask][idx_sample],
                               x[mask][idx_sample],
                               s=markersize,
                               marker=marker,
                               linewidth=linewidths,
                               color=colors[i % len(colors)],
                               alpha=a)
    ymin, ymax = ax.get_ylim()
    ax.set_yticks([0, ymax])
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)
    return


def plot_sig_pred_axis(ax,
                       t,
                       x,
                       labels=None,
                       legends=None,
                       line_plot=False,
                       marker='x',
                       markersize=10,
                       linewidths=0.5,
                       title_fontsize=30,
                       a=1.0,
                       downsample=1,
                       show_legend=False,
                       title=None):
    """Plot predicted modality versus time in a subplot."""
    if labels is None or legends is None:
        if line_plot:
            ax.plot(t[::downsample], x[::downsample], 'k-', linewidth=5, alpha=a)
        else:
            ax.scatter(t[::downsample],
                       x[::downsample],
                       marker=marker,
                       s=markersize,
                       linewidths=linewidths,
                       color='k',
                       alpha=a)
    else:
        for i, label in enumerate(legends):
            mask = labels == i
            if np.any(mask):
                idx_ordered = np.argsort(t[mask][::downsample])
                if show_legend:
                    if line_plot:
                        ax.plot(t[mask][::downsample][idx_ordered],
                                x[mask][::downsample][idx_ordered],
                                'k-',
                                linewidth=5,
                                alpha=a,
                                label=label)
                    else:
                        ax.scatter(t[mask][::downsample][idx_ordered],
                                   x[mask][::downsample][idx_ordered],
                                   marker=marker,
                                   s=markersize,
                                   linewidths=linewidths,
                                   color='k',
                                   alpha=a,
                                   label=label)
                else:
                    if line_plot:
                        ax.plot(t[mask][::downsample][idx_ordered],
                                x[mask][::downsample][idx_ordered],
                                'k-',
                                linewidth=5,
                                alpha=a)
                    else:
                        ax.scatter(t[mask][::downsample][idx_ordered],
                                   x[mask][::downsample][idx_ordered],
                                   marker=marker,
                                   s=markersize,
                                   linewidths=linewidths,
                                   color='k',
                                   alpha=a)
    ymin, ymax = ax.get_ylim()
    ax.set_yticks([0, ymax])
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)
    return ax


def plot_sig_loess_axis(ax,
                        t,
                        x,
                        labels,
                        legends,
                        frac=0.5,
                        a=1.0,
                        downsample=1,
                        title_fontsize=30,
                        show_legend=False,
                        title=None,):
    """LOESS plot in a subplot."""
    from loess import loess_1d
    for i, label in enumerate(legends):
        mask = labels == i
        if np.any(mask):
            t_lb, t_ub = np.quantile(t[mask], 0.05), np.quantile(t[mask], 0.95)
            mask2 = (t <= t_ub) & (t >= t_lb) & mask
            if np.sum(mask2) >= 20:
                tout, xout, wout = loess_1d.loess_1d(t[mask2],
                                                     x[mask2],
                                                     xnew=None,
                                                     degree=1,
                                                     frac=frac,
                                                     npoints=None,
                                                     rotate=False,
                                                     sigy=None)
                torder = np.argsort(tout)
                if show_legend:
                    ax.plot(tout[torder][::downsample],
                            xout[torder][::downsample],
                            'k-',
                            linewidth=5,
                            alpha=a,
                            label=label)
                else:
                    ax.plot(tout[torder][::downsample],
                            xout[torder][::downsample],
                            'k-',
                            linewidth=5,
                            alpha=a)
    ymin, ymax = ax.get_ylim()
    ax.set_yticks([0, ymax])
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)
    return ax


def sample_quiver_plot(t, dt, x=None, n_bins=3):
    """Sample cells for a velocity quiver plot."""
    tmax, tmin = t.max()+1e-3, np.quantile(t, 0.01)
    Nbin = int(np.clip((tmax-tmin)/dt, 1, len(t)//2))
    indices = []
    for i in range(Nbin):
        batch_idx = np.where((t >= tmin + i * dt) & (t <= tmin + (i+1) * dt))[0]
        if len(batch_idx) > 0:
            if x is None:
                indices.append(batch_idx[len(batch_idx)//2])
            else:
                edges = np.linspace(np.quantile(x[batch_idx], 0.1), np.quantile(x[batch_idx], 0.9), n_bins+1)
                for j in range(n_bins):
                    mask = (x[batch_idx] >= edges[j]) & (x[batch_idx] <= edges[j+1])
                    if np.any(mask):
                        indices.append(np.random.choice(batch_idx[mask]))
    return np.array(indices).astype(int)


def plot_vel_axis(ax,
                  t,
                  x,
                  v,
                  labels=None,
                  legends=None,
                  dt=0.1,
                  show_legend=False,
                  sparsity_correction=False,
                  palette=None,
                  headwidth=5.0,
                  headlength=8.0):
    """Velocity quiver plot on a u/s-t subplot."""
    if labels is None or legends is None:
        dt_sample = (t.max()-t.min())/50
        torder = np.argsort(t)
        try:
            indices = (sample_quiver_plot(t[torder], dt_sample, x[torder], n_bins=5)
                       if sparsity_correction else
                       sample_quiver_plot(t[torder], dt_sample, n_bins=5))
        except ValueError:
            np.random.seed(42)
            indices = np.random.choice(len(t), len(t)//30, replace=False)
        if len(indices) > 0:
            ax.quiver(t[torder][indices],
                      x[torder][indices],
                      dt*np.ones((len(indices))),
                      dt*v[torder][indices],
                      angles='xy',
                      scale=None,
                      scale_units='inches',
                      headwidth=headwidth,
                      headlength=headlength,
                      color='k')
    else:
        colors = (palette if isinstance(palette, np.ndarray) or isinstance(palette, list)
                  else get_colors(len(legends)))
        cell_types_int = np.unique(labels)
        for i,  type_int in enumerate(cell_types_int):
            mask = labels == type_int
            t_type = t[mask]
            dt_sample = (t_type.max()-t_type.min())/30
            if np.any(mask):
                torder = np.argsort(t_type)
                try:
                    indices = (sample_quiver_plot(t_type[torder], dt_sample, x[mask][torder], n_bins=4)
                               if sparsity_correction else
                               sample_quiver_plot(t_type[torder], dt_sample, n_bins=4))
                except ValueError:
                    np.random.seed(42)
                    indices = np.random.choice(len(t_type), len(t_type)//30+1, replace=False)
                if len(indices) == 0:  # edge case handling
                    continue
                v_type = v[mask][torder][indices]
                v_type = np.clip(v_type, np.quantile(v_type, 0.01), np.quantile(v_type, 0.95))
                # Actual Quiver Plot
                if show_legend:
                    ax.quiver(t_type[torder][indices],
                              x[mask][torder][indices],
                              dt*np.ones((len(indices))),
                              dt*v_type,
                              label=legends[i],
                              angles='xy',
                              scale=None,
                              scale_units='inches',
                              headwidth=headwidth,
                              headlength=headlength,
                              color=colors[i % len(colors)])
                else:
                    ax.quiver(t_type[torder][indices],
                              x[mask][torder][indices],
                              dt*np.ones((len(indices))),
                              dt*v_type,
                              angles='xy',
                              scale=None,
                              scale_units='inches',
                              headwidth=headwidth,
                              headlength=headlength,
                              color=colors[i % len(colors)])
    ymin, ymax = ax.get_ylim()
    ax.set_yticks([max(0, ymin * 0.9), ymax * 1.1])
    return ax


def plot_sig_grid(n_rows,
                  n_cols,
                  gene_list,
                  T,
                  U,
                  S,
                  Labels,
                  Legends,
                  That={},
                  Uhat={},
                  Shat={},
                  V={},
                  Labels_demo={},
                  width=WIDTH,
                  height=HEIGHT,
                  alpha=1.0,
                  downsample=1,
                  loess_downsample=None,
                  sparsity_correction=False,
                  plot_loess=False,
                  frac=0.5,
                  marker=".",
                  markersize=MARKERSIZE,
                  linewidths=0.5,
                  palette=None,
                  show_legend=True,
                  legend_fontsize=None,
                  title_fontsize=FONTSIZE,
                  headwidth=5.0,
                  headlength=8.0,
                  label_fontsize=FONTSIZE,
                  y_label_pos_x=-0.03,
                  y_label_pos_y=0.5,
                  show_xticks=False,
                  tick_fontsize=TICK_FONTSIZE,
                  hspace=0.3,
                  wspace=0.12,
                  markerscale=5.0,
                  bbox_to_anchor=None,
                  legend_loc='upper right',
                  path='figures',
                  figname=None,
                  save_format='png'):
    """Plot u/s of a list of genes vs. time in an [n_rows x n_cols] grid of subplots.
    Cells are colored according to their dynamical state or cell type.

    Args:
        n_rows (int):
            Number of rows of the grid plot.
        n_cols (int):
            Number of columns of the grid plot.
        gene_list (array like):
            Genes to plot. If the length exceeds n_rows*n_cols, multiple figures will
            be generated. If length is less than n_rows*n_cols, some subplots will be
            blank.
        T (dict):
            Keys are methods (string) and values are time arrays.
            For some methods, the value is an (N,G) array
            instead of an (N) array because of local fitting.
        U (:class:`numpy.ndarray`):
            Unspliced count data.
            Contain just the genes for plotting.
        S (:class:`numpy.ndarray`):
            Spliced count data.
            Contain just the genes for plotting.
        Labels (dict):
            Keys are methods and values are arrays of cell annotation.
            Usually the values are cell type annotations.
        Legends (dict):
            Keys are methods and values are legend names.
            Usually the legend names are unique values of cell annotation.
            In our application, these are unique cell types.
        That (dict, optional):
            Keys are methods and values are (N_eval) of cell time.
            Time used in evaluation. N_eval is generally unequal to number of cells
            in the original data and the time points don't necessarily match the original
            cell because we often need fewer time points to evaluate a parametric model.
            For scVelo, the value is an (N_eval,G) array instead of an (N_eval) array
            because of local fitting. Defaults to {}.
        Uhat (dict, optional):
            Dictionary with method names as keys and arrays of predicted u as values.
            Defaults to {}.
        Shat (dict, optional):
            Dictionary with method names as keys and arrays of predicted s as values.
            Defaults to {}.
        V (dict, optional):
            Keys are methods and values are (N,G) arrays of velocity.
            Defaults to {}.
        Labels_demo (dict, optional):
            Keys are methods and values are cell type annotations of the prediction.
            Defaults to {}.
        width (int, optional):
            Subplot width. Defaults to 6.
        height (int, optional):
            Subplot height. Defaults to 3.
        alpha (float, optional):
            Transparency of the data points.. Defaults to 1.0.
        downsample (int, optional):
            Down-sampling factor to reduce the overlapping of data points. Defaults to 1.
        sparsity_correction (bool, optional):
            Whether to sample u/s uniformly in the range to avoid
            sapling most zeros in sparse expression profiles.
            Defaults to False.
        plot_loess (bool, optional):
            Whether to plot a line fit. Defaults to False.
        frac (float, optional):
            Hyper-parameter for the LOESS plot.
            This is the window length of the local regression.
            Defaults to 0.5.
        marker (str, optional):
            Marker for the data points. Defaults to 'o'.
        markersize (int, optional):
            Size of the markers. Defaults to MARKERSIZE.
        linewidths (float, optional):
            Width of the marker edge. Defaults to 0.5.
        palette (:class:`numpy.ndarray`, optional):
            User-defined colormap for different cell types. Defaults to None.
        show_legend (bool, optional):
            Whether to show the legend. Defaults to True.
        legend_fontsize (int, optional):
            Defaults to None.
        title_fontsize (int, optional):
            Defaults to FONTSIZE.
        headwidth (float, optional):
            Width of the arrow head. Defaults to 5.0.
        headlength (float, optional):
            Length of the arrow head. Defaults to 8.0.
        label_fontsize (int, optional):
            x/y axis label fontsize. Defaults to FONTSIZE.
        y_label_pos_x (float, optional):
            x position of the y-axis label relative to the y axis. Defaults to -0.03.
        y_label_pos_y (float, optional):
            y position of the y-axis label relative to the y axis. Defaults to 0.5.
        show_xticks (bool, optional):
            Whether to show xticks. Defaults to False.
        tick_fontsize (int, optional):
            Fontsize of the ticks. Defaults to TICK_FONTSIZE.
        hspace (float, optional):
            Height distance proportion between subplots. Defaults to 0.3.
        wspace (float, optional):
            Width distance proportion between subplots. Defaults to 0.12.
        markerscale (float, optional):
            Marker scale for the legend. Defaults to 5.0.
        bbox_to_anchor (tuple, optional):
            Bbox to anchor the legend. Defaults to None.
        legend_loc (str, optional):
            Location of the legend. Defaults to 'upper right'.
        path (str, optional):
            Saving path. Defaults to 'figures'.
        figname (str, optional):
            Name if the figure.
            Because there can be multiple figures generated in this function.
            We will append a number to figname when saving the figures.
            Figures will not be saved if set to None. Defaults to None.
        save_format (str, optional):
            Figure format, could be png, pdf, svg, eps and ps. Defaults to 'png'.

    """
    methods = list(Uhat.keys())
    M = max(1, len(methods))

    # Detect whether multiple figures are needed
    Nfig = len(gene_list) // (n_rows*n_cols)
    if Nfig * n_rows * n_cols < len(gene_list):
        Nfig += 1

    # Plotting
    for i_fig in range(Nfig):
        fig_sig, ax_sig = plt.subplots(3 * n_rows, M * n_cols, figsize=(width * M * n_cols + 1.0, 3 * height * n_rows), facecolor='white')
        if M * n_cols == 1:
            for i in range(min(n_rows, len(gene_list) - i_fig * n_rows)):
                idx = i_fig*n_rows+i
                t = T[methods[0]][:, idx] if T[methods[0]].ndim == 2 else T[methods[0]]
                if np.any(np.isnan(t)):
                    continue
                that = That[methods[0]][:, idx] if That[methods[0]].ndim == 2 else That[methods[0]]
                title = f"{gene_list[idx]} ({methods[0]})"
                plot_sig_axis(ax_sig[3*i],
                              t,
                              U[:, idx],
                              Labels[methods[0]],
                              Legends[methods[0]],
                              marker,
                              markersize,
                              linewidths,
                              alpha,
                              downsample,
                              show_legend,
                              palette,
                              title,
                              title_fontsize)
                plot_sig_axis(ax_sig[3*i+1],
                              t,
                              S[:, idx],
                              Labels[methods[0]],
                              Legends[methods[0]],
                              marker,
                              markersize,
                              linewidths,
                              alpha,
                              downsample,
                              False,
                              palette)

                try:
                    if ('VeloVAE' in methods[0])\
                        or ('TopoVelo' in methods[0])\
                        or (methods[0] in ['DeepVelo',
                                           'Discrete PyroVelocity',
                                           'PyroVelocity',
                                           'VeloVI',
                                           'cellDancer']):
                        if loess_downsample is None:
                            loess_downsample = min(10, max(len(that)//5000, 1))
                        if plot_loess:
                            plot_sig_loess_axis(ax_sig[3 * i],
                                                that[::loess_downsample],
                                                Uhat[methods[0]][:, idx][::loess_downsample],
                                                Labels_demo[methods[0]][::loess_downsample],
                                                Legends[methods[0]],
                                                frac=frac,
                                                title_fontsize=title_fontsize)
                            plot_sig_loess_axis(ax_sig[3 * i + 1],
                                                that[::loess_downsample],
                                                Shat[methods[0]][:, idx][::loess_downsample],
                                                Labels_demo[methods[0]][::loess_downsample],
                                                Legends[methods[0]],
                                                frac=frac,
                                                title_fontsize=title_fontsize)
                        elif 'Discrete' in methods[0]:
                            uhat_plot = np.random.poisson(Uhat[methods[0]][:, idx])
                            shat_plot = np.random.poisson(Shat[methods[0]][:, idx])
                            plot_sig_pred_axis(ax_sig[3*i],
                                               that,
                                               uhat_plot,
                                               markersize=markersize,
                                               linewidths=linewidths,
                                               title_fontsize=title_fontsize,
                                               a=alpha,
                                               downsample=downsample,
                                               show_legend=False)
                            plot_sig_pred_axis(ax_sig[3*i+1],
                                               that,
                                               shat_plot,
                                               markersize=markersize,
                                               linewidths=linewidths,
                                               title_fontsize=title_fontsize,
                                               a=alpha,
                                               downsample=downsample,
                                               show_legend=False)
                        plot_vel_axis(ax_sig[3 * i + 2],
                                      t,
                                      Shat[methods[0]][:, idx],
                                      V[methods[0]][:, idx],
                                      Labels[methods[0]],
                                      Legends[methods[0]],
                                      sparsity_correction=sparsity_correction,
                                      palette=palette,
                                      headwidth=headwidth,
                                      headlength=headlength)
                    else:  # plot line prediction
                        plot_sig_pred_axis(ax_sig[3*i],
                                           that,
                                           Uhat[methods[0]][:, idx],
                                           Labels_demo[methods[0]],
                                           Legends[methods[0]],
                                           line_plot=True,
                                           a=1.0,
                                           downsample=downsample)
                        plot_sig_pred_axis(ax_sig[3*i+1],
                                           that,
                                           Shat[methods[0]][:, idx],
                                           Labels_demo[methods[0]],
                                           Legends[methods[0]],
                                           line_plot=True,
                                           a=1.0,
                                           downsample=downsample)
                        plot_vel_axis(ax_sig[3*i+2],
                                      t,
                                      S[:, idx],
                                      V[methods[0]][:, idx],
                                      Labels[methods[0]],
                                      Legends[methods[0]],
                                      sparsity_correction=sparsity_correction,
                                      palette=palette,
                                      headwidth=headwidth,
                                      headlength=headlength)
                except (KeyError, TypeError):
                    print("[** Warning **]: "
                          "Skip plotting the prediction because of key value error or invalid data type.")
                    return
                if np.all(~np.isnan(t)):
                    ax_sig[3*i].set_xlim(t.min(), np.quantile(t, 0.999)+0.1)
                    ax_sig[3*i+1].set_xlim(t.min(), np.quantile(t, 0.999)+0.1)
                    ax_sig[3*i+2].set_xlim(t.min(), np.quantile(t, 0.999)+0.1)

                ax_sig[3*i].set_ylabel("U", fontsize=label_fontsize, rotation=0)
                ax_sig[3*i].yaxis.set_label_coords(y_label_pos_x, y_label_pos_y)

                ax_sig[3*i+1].set_ylabel("S", fontsize=label_fontsize, rotation=0)
                ax_sig[3*i+1].yaxis.set_label_coords(y_label_pos_x, y_label_pos_y)

                ax_sig[3*i+2].set_ylabel("S", fontsize=label_fontsize, rotation=0)
                ax_sig[3*i+2].yaxis.set_label_coords(y_label_pos_x, y_label_pos_y)

                for r in range(3):
                    ax_sig[3*i+r].set_xticks([])
                    ax_sig[3*i+r].set_yticks([])
                    ax_sig[3*i+r].tick_params(axis='both', which='major', labelsize=tick_fontsize)
                    ax_sig[3*i+r].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
        else:
            legends = []
            for i in range(n_rows):
                for j in range(n_cols):  # i, j: row and column gene index
                    idx = i_fig * n_rows * n_cols + i * n_cols + j  # which gene
                    if idx >= len(gene_list):
                        break
                    for k, method in enumerate(methods):  # k: method index
                        # Pick time according to the method
                        if T[method].ndim == 2:
                            t = T[method][:, idx]
                            that = That[method][:, idx]
                        else:
                            t = T[method]
                            that = That[method]

                        title = f"{gene_list[idx]} ({method})"
                        plot_sig_axis(ax_sig[3*i, M*j+k],
                                      t,
                                      U[:, idx],
                                      Labels[method],
                                      Legends[method],
                                      marker,
                                      markersize,
                                      linewidths,
                                      alpha,
                                      downsample,
                                      show_legend,
                                      palette,
                                      title,
                                      title_fontsize)
                        plot_sig_axis(ax_sig[3*i+1, M*j+k],
                                      t,
                                      S[:, idx],
                                      Labels[method],
                                      Legends[method],
                                      marker,
                                      markersize,
                                      linewidths,
                                      alpha,
                                      downsample,
                                      False,
                                      palette)

                        if len(Legends[method]) > len(legends):
                            legends = Legends[method]
                        try:
                            if ('VeloVAE' in method)\
                                or ('TopoVelo' in method)\
                                or (method in ['DeepVelo',
                                               'Discrete PyroVelocity',
                                               'PyroVelocity',
                                               'VeloVI',
                                               'cellDancer']):
                                # These methods don't have line prediction
                                if loess_downsample is None:
                                    loess_downsample = min(10, max(len(that)//5000, 1))
                                if plot_loess:
                                    plot_sig_loess_axis(ax_sig[3*i, M*j+k],
                                                        that[::loess_downsample],
                                                        Uhat[method][:, idx][::loess_downsample],
                                                        Labels_demo[method][::loess_downsample],
                                                        Legends[method],
                                                        frac=frac)
                                    plot_sig_loess_axis(ax_sig[3*i+1, M*j+k],
                                                        that[::K],
                                                        Shat[method][:, idx][::loess_downsample],
                                                        Labels_demo[method][::loess_downsample],
                                                        Legends[method], frac=frac)
                                elif 'Discrete' in method:
                                    uhat_plot = np.random.poisson(Uhat[method][:, idx])
                                    shat_plot = np.random.poisson(Shat[method][:, idx])
                                    plot_sig_pred_axis(ax_sig[3*i, M*j+k],
                                                       that,
                                                       uhat_plot,
                                                       markersize=markersize,
                                                       linewidths=linewidths,
                                                       title_fontsize=title_fontsize,
                                                       a=alpha,
                                                       downsample=downsample,
                                                       show_legend=False)
                                    plot_sig_pred_axis(ax_sig[3*i+1, M*j+k],
                                                       that,
                                                       shat_plot,
                                                       markersize=markersize,
                                                       linewidths=linewidths,
                                                       title_fontsize=title_fontsize,
                                                       a=alpha,
                                                       downsample=downsample,
                                                       show_legend=False)
                                plot_vel_axis(ax_sig[3*i+2, M*j+k],
                                              t,
                                              Shat[method][:, idx],
                                              V[method][:, idx],
                                              Labels[method],
                                              Legends[method],
                                              sparsity_correction=sparsity_correction,
                                              palette=palette,
                                              headwidth=headwidth,
                                              headlength=headlength)
                            else:  # plot line prediction
                                plot_sig_pred_axis(ax_sig[3*i, M*j+k],
                                                   that,
                                                   Uhat[method][:, idx],
                                                   Labels_demo[method],
                                                   Legends[method],
                                                   line_plot=True,
                                                   a=1.0,
                                                   downsample=downsample)
                                plot_sig_pred_axis(ax_sig[3*i+1, M*j+k],
                                                   that,
                                                   Shat[method][:, idx],
                                                   Labels_demo[method],
                                                   Legends[method],
                                                   line_plot=True,
                                                   a=1.0,
                                                   downsample=downsample)
                                plot_vel_axis(ax_sig[3*i+2, M*j+k],
                                              t,
                                              S[:, idx],
                                              V[method][:, idx],
                                              Labels[method],
                                              Legends[method],
                                              sparsity_correction=sparsity_correction,
                                              palette=palette,
                                              headwidth=headwidth,
                                              headlength=headlength)
                        except (KeyError, TypeError):
                            print("[** Warning **]: "
                                  "Skip plotting the prediction because of key value error or invalid data type.")
                            pass
                        if np.all(~np.isnan(t)):
                            for r in range(3):
                                ax_sig[3*i+r,  M*j+k].set_xlim(t.min(), np.quantile(t, 0.999)+0.1)
                                if not show_xticks:
                                    ax_sig[3*i+r,  M*j+k].set_xticks([])
                                ax_sig[3*i,  M*j+k].set_xlabel("Time", fontsize=label_fontsize)

                        ax_sig[3*i,  M*j+k].set_ylabel("U", fontsize=label_fontsize, rotation=0)
                        ax_sig[3*i,  M*j+k].yaxis.set_label_coords(y_label_pos_x, y_label_pos_y)

                        ax_sig[3*i+1,  M*j+k].set_ylabel("S", fontsize=label_fontsize, rotation=0)
                        ax_sig[3*i+1,  M*j+k].yaxis.set_label_coords(y_label_pos_x, y_label_pos_y)

                        ax_sig[3*i+2,  M*j+k].set_ylabel("S", fontsize=label_fontsize, rotation=0)
                        ax_sig[3*i+2,  M*j+k].yaxis.set_label_coords(y_label_pos_x, y_label_pos_y)

                        for r in range(3):
                            ax_sig[3*i+r, M*j+k].tick_params(axis='both', which='major', labelsize=tick_fontsize)
                            ax_sig[3*i+r, M*j+k].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
        if show_legend:
            if ax_sig.ndim == 1:
                handles, labels = ax_sig[0].get_legend_handles_labels()
            else:
                handles, labels = ax_sig[0, 0].get_legend_handles_labels()

            if legend_fontsize is None:
                legend_fontsize = np.min([int(30*n_rows), 300*n_rows/len(Legends[methods[0]]), int(30*n_cols)])
            if bbox_to_anchor is None:
                l_indent = 1 - 0.02/n_rows
                bbox_to_anchor = (-0.03/n_cols, l_indent)
            lgd = fig_sig.legend(handles,
                                 labels,
                                 fontsize=legend_fontsize,
                                 markerscale=markerscale,
                                 bbox_to_anchor=bbox_to_anchor,
                                 loc=legend_loc)

        fig_sig.subplots_adjust(hspace=hspace, wspace=wspace)
        plt.tight_layout()

        save = None if (path is None or figname is None) else f'{path}/{figname}_{i_fig+1}.{save_format}'
        if show_legend:
            save_fig(fig_sig, save, (lgd,))
        save_fig(fig_sig, save)


def plot_time_grid(T,
                   X_emb,
                   capture_time=None,
                   std_t=None,
                   downsample=1,
                   max_quantile=0.99,
                   color_map='viridis',
                   width=WIDTH,
                   height=HEIGHT,
                   fix='width',
                   marker='.',
                   markersize=MARKERSIZE,
                   linewidths=0.5,
                   grid_size=None,
                   real_aspect_ratio=True,
                   title_off=False,
                   title_fontsize=None,
                   show_colorbar=True,
                   colorbar_fontsize=COLORBAR_FONTSIZE,
                   colorbar_ticklabels=['min', 'max'],
                   colorbar_tick_fontsize=COLORBAR_TICK_FONTSIZE,
                   colorbar_labelpad=20,
                   colorbar_pos=[1.04, 0.2, 0.05, 0.6],
                   path='figures',
                   figname='time',
                   save_format='png'):
    """Plot the latent time of different methods.

    Args:
        T (dict):
            Keys are method names and values are (N) arrays containing time
        X_emb (:class:`numpy.ndarray`):
            2D embedding for visualization, (N,2)
        capture_time (:class:`numpy.ndarray`, optional):
            Capture time, (N,). Defaults to None.
        std_t (dict, optional):
            Keys are method names and values are (N) arrays
            containing standard deviations of cell time.
            Not applicable to some methods. Defaults to None.
        downsample (int, optional):
            Down-sampling factor to reduce data point overlapping.. Defaults to 1.
        max_quantile (float, optional):
            Top quantile for clipping extreme values. Defaults to 0.99.
        width (int, optional):
            Subplot width. Defaults to WIDTH. Ignored when real_aspect_ratio is True.
        height (int, optional):
            Subplot height. Defaults to HEIGHT. Ignored when real_aspect_ratio is True.
        fix (str, optional):
            Fix the width or height of the figure. Defaults to 'width'.
        marker (str, optional):
            Marker type. Defaults to 'o'.
        markersize (int, optional):
            Size of the dots. Defaults to MARKERSIZE.
        linewidths (float, optional):
            Width of the marker edges. Defaults to 0.5.
        grid_size (tuple, optional):
            Grid size. Defaults to None.
        real_aspect_ratio (bool, optional):
            Whether to use real aspect ratio. Defaults to True.
        title_off (bool, optional):
            Whether to show the title. Defaults to False.
        title_fontsize (int, optional):
            Font size of the title. Defaults to None.
        show_colorbar (bool, optional):
            Whether to show colorbar. Defaults to True.
        colorbar_fontsize (int, optional):
            Font size of the colorbar. Defaults to COLORBAR_FONTSIZE.
        colorbar_tick_fontsize (int, optional):
            Font size of the colorbar ticks. Defaults to COLORBAR_TICK_FONTSIZE.
        colorbar_labelpad (int, optional):
            Label pad of the colorbar. Defaults to 20.
        color_map (str, optional):
            Colormap. Defaults to 'plasma_r'.
        path (str, optional):
            Saving path. Defaults to 'figures'.
        figname (str, optional):
            Name of the figure. Defaults to 'time_grid'.
        save_format (str, optional):
            Figure format, could be png, pdf, svg, eps and ps. Defaults to 'png'.
    """
    if capture_time is not None:
        methods =  list(T.keys()) + ["Capture Time"]
    else:
        methods = list(T.keys())
    M = len(methods)

    if grid_size is None:
        grid_size = (1, M)
    n_row, n_col = grid_size

    # Calculate figure size
    if real_aspect_ratio:
        panel_figsize = compute_figsize(X_emb, width, height, real_aspect_ratio, fix)
        figsize = (panel_figsize[0]*n_col, panel_figsize[1]*n_row)
    else:
        figsize = (width*n_col, height*n_row)
    
    if title_fontsize is None:
        title_fontsize = 4 * width

    if std_t is not None:
        fig_time, ax = plt.subplots(2*n_row, n_col, figsize=figsize, facecolor='white')
        for i, method in enumerate(methods):
            row = i // n_col
            col = i - row * n_col
            t = capture_time if method == "Capture Time" else T[method]
            t = np.clip(t, None, np.quantile(t, max_quantile))
            t = t - t.min()
            t = t/(t.max() + (t.max() == 0))
            if n_col > 1:
                ax[2*row, col].scatter(X_emb[::downsample, 0],
                                       X_emb[::downsample, 1],
                                       s=markersize,
                                       marker=marker,
                                       c=t[::downsample],
                                       cmap=color_map,
                                       linewidths=linewidths)
                if not title_off:
                    if method == "Capture Time":
                        ax[2*row, col].set_title("Expected Temporal Order", fontsize=title_fontsize)
                    else:
                        ax[2*row, col].set_title(method, fontsize=title_fontsize)
            else:
                ax[2*row].scatter(X_emb[::downsample, 0],
                                  X_emb[::downsample, 1],
                                  s=markersize,
                                  marker=marker,
                                  edgecolors='none',
                                  c=t[::downsample],
                                  cmap=color_map,
                                  linewidths=linewidths)
                if not title_off:
                    if method == "Capture Time":
                        ax[2*row].set_title("Expected Temporal Order", fontsize=title_fontsize)
                    else:
                        ax[2*row].set_title(method, fontsize=title_fontsize)

            # Plot the Time Variance in a Colormap
            var_t = std_t[method]**2

            if np.any(var_t > 0):
                if M > 1:
                    ax[2*row+1, col].scatter(X_emb[::downsample, 0],
                                             X_emb[::downsample, 1],
                                             s=markersize,
                                             marker=marker,
                                             edgecolors='none',
                                             c=var_t[::downsample],
                                             cmap='Reds',
                                             linewidths=linewidths)
                    norm1 = matplotlib.colors.Normalize(vmin=np.min(var_t), vmax=np.max(var_t))
                    sm1 = matplotlib.cm.ScalarMappable(norm=norm1, cmap='Reds')
                    cbar1 = fig_time.colorbar(sm1, ax=ax[2*row+1, col])
                    cbar1.ax.get_yaxis().labelpad = 15
                    cbar1.ax.set_ylabel('Time Variance', rotation=270, fontsize=colorbar_fontsize)
                else:
                    ax[2*row+1].scatter(X_emb[::downsample, 0],
                                        X_emb[::downsample, 1],
                                        s=markersize,
                                        marker=marker,
                                        c=var_t[::downsample],
                                        cmap='Reds',
                                        linewidths=linewidths)
                    norm1 = matplotlib.colors.Normalize(vmin=np.min(var_t), vmax=np.max(var_t))
                    sm1 = matplotlib.cm.ScalarMappable(norm=norm1, cmap='Reds')
                    cbar1 = fig_time.colorbar(sm1, ax=ax[2*row+1])
                    cbar1.ax.get_yaxis().labelpad = colorbar_labelpad
                    cbar1.ax.set_ylabel('Time Variance', rotation=270, fontsize=colorbar_fontsize)
    else:
        fig_time, ax = plt.subplots(n_row, n_col, figsize=figsize, facecolor='white')
        for i, method in enumerate(methods):
            row = i // n_col
            col = i - row * n_col
            t = capture_time if method == "Capture Time" else T[method]
            t = np.clip(t, None, np.quantile(t, max_quantile))
            t = t - t.min()
            t = t/(t.max() + (t.max() == 0))
            if n_col > 1 and n_row > 1:
                ax[row, col].scatter(X_emb[::downsample, 0],
                                     X_emb[::downsample, 1],
                                     s=markersize,
                                     marker=marker,
                                     c=t[::downsample],
                                     cmap=color_map,
                                     linewidths=linewidths)
                if not title_off:
                    if method == "Capture Time":
                        ax[row, col].set_title("Expected Temporal Order", fontsize=title_fontsize)
                    else:
                        ax[row, col].set_title(method, fontsize=title_fontsize)
                ax[row, col].axis('off')
            elif n_col > 1:
                ax[col].scatter(X_emb[::downsample, 0],
                                X_emb[::downsample, 1],
                                s=markersize,
                                marker=marker,
                                c=t[::downsample],
                                cmap=color_map,
                                linewidths=linewidths)
                if not title_off:
                    if method == "Capture Time":
                        ax[col].set_title("Expected Temporal Order", fontsize=title_fontsize)
                    else:
                        ax[col].set_title(method, fontsize=title_fontsize)
                ax[col].axis('off')
            elif n_row > 1:
                ax[row].scatter(X_emb[::downsample, 0],
                                X_emb[::downsample, 1],
                                s=markersize,
                                marker=marker,
                                c=t[::downsample],
                                cmap=color_map,
                                linewidths=linewidths)
                if not title_off:
                    if method == "Capture Time":
                        ax[row].set_title("Expected Temporal Order", fontsize=title_fontsize)
                    else:
                        ax[row].set_title(method, fontsize=title_fontsize)
                ax[row].axis('off')
            else:
                ax.scatter(X_emb[::downsample, 0],
                           X_emb[::downsample, 1],
                           s=markersize,
                           marker=marker,
                           c=t[::downsample],
                           cmap=color_map,
                           linewidths=linewidths)
                if not title_off:
                    if method == "Capture Time":
                        ax.set_title("Expected Temporal Order", fontsize=title_fontsize)
                    else:
                        ax.set_title(method, fontsize=title_fontsize)
                ax.axis('off')
    if show_colorbar:
        if isinstance(ax, np.ndarray):
            _set_colorbar(ax[-1], 0, 1, color_map, 'Cell Time', colorbar_ticklabels, colorbar_fontsize, None, colorbar_pos, colorbar_tick_fontsize)
        else:
            _set_colorbar(ax, 0, 1, color_map, 'Cell Time', colorbar_ticklabels, colorbar_fontsize, None, colorbar_pos, colorbar_tick_fontsize)
    
    save = None if (path is None or figname is None) else f'{path}/{figname}.{save_format}'
    save_fig(fig_time, save)


def _adj_mtx_to_map(w):
    """Convert adjacency matrix to a mapping (adjacency list)."""
    # w[i,j] = 1 if j is the parent of i
    G = {}
    for i in range(w.shape[1]):
        G[i] = []
        for j in range(w.shape[0]):
            if w[j, i] > 0:
                G[i].append(j)
    return G


def get_depth(graph):
    """Get the depths of all nodes in a tree-like graph."""
    depth = np.zeros((len(graph.keys())))
    roots = []
    for u in graph:
        if u in graph[u]:
            roots.append(u)
    for root in roots:
        queue = [root]
        depth[root] = 0
        while len(queue) > 0:
            v = queue.pop(0)
            for u in graph[v]:
                if u == root:
                    continue
                queue.append(u)
                depth[u] = depth[v] + 1
    return depth


def _plot_branch(ax, t, x, graph, label_dic_rev, plot_depth=True, color_map=None):
    """Plot some attributes of all nodes in a tree-like graph. """
    colors = get_colors(len(t), color_map)
    if plot_depth:
        depth = get_depth(graph)
        for i in range(len(t)):
            ax.scatter(depth[i:i+1], x[i:i+1], s=80, color=colors[i], label=label_dic_rev[i])
        for parent in graph:
            for child in graph[parent]:
                ax.plot([depth[child], depth[parent]], [x[child], x[parent]], "k-", alpha=0.2, linewidth=3)
    else:
        for i in range(len(t)):
            ax.scatter(t[i:i+1], x[i:i+1], s=80, color=colors[i], label=label_dic_rev[i])
        for parent in graph:
            for child in graph[parent]:
                ax.plot([t[child], t[parent]], [x[child], x[parent]], "k-", alpha=0.2, linewidth=3)
    return ax


def plot_rate_grid(adata,
                   key,
                   gene_list,
                   n_rows,
                   n_cols,
                   width=WIDTH,
                   height=HEIGHT,
                   legend_ncol=8,
                   legend_fontsize=LEGEND_FONTSIZE,
                   markerscale=MARKERSCALE,
                   title_fontsize=TITLE_FONTSIZE,
                   label_fontsize=LABEL_FONTSIZE,
                   bbox_to_anchor=None,
                   legend_loc='upper right',
                   plot_depth=True,
                   color_map=None,
                   path="figures",
                   figname="genes",
                   save_format="png"):
    """Plot cell-type-specific rate parameters inferred from branching ODE.

    Args:
        adata (:class:`anndata.AnnData`):
            AnnData object.
        key (str):
            Key used to extract the corresponding rate parameters.
            For example, f"{key}_alpha" will be used to extract the transcription rate from .varm
        gene_list (array like):
            List of genes to plot
        n_rows (int):
            Number of rows of the subplot grid.
        n_cols (int):
            Number of columns of the subplot grid.
        width (int, optional):
            Subplot width. Defaults to WIDTH.
        height (int, optional):
            Subplot width. Defaults to HEIGHT.
        legend_ncol (int, optional):
            Number of columns in the legend. Defaults to 8.
        legend_fontsize (int, optional):
            Fontsize of the legend. Defaults to LEGEND_FONTSIZE.
        markerscale (float, optional):
            Marker scale for the legend. Defaults to MARKERSCALE.
        title_fontsize (int, optional):
            Fontsize of the title. Defaults to TITLE_FONTSIZE.
        label_fontsize (int, optional):
            Fontsize of the labels. Defaults to LABEL_FONTSIZE.
        bbox_to_anchor (tuple, optional):
            Bbox to anchor the legend. Defaults to None.
        legend_loc (str, optional):
            Location of the legend. Defaults to 'upper right'.
        plot_depth (bool, optional):
            Whether to plot the depth in transition graph as a surrogate of time.
            Set to true by default for better visualization. Defaults to True.
        color_map (str, optional):
            Defaults to None.
        path (str, optional):
            Path to the folder for saving the figure. Defaults to "figures".
        figname (str, optional):
            Name of the saved figure. Defaults to "genes".
        save_format (str, optional):
            Figure format, could be png, pdf, svg, eps and ps. Defaults to 'png'. Defaults to "png".

    """
    num_fig = len(gene_list) // (n_rows*n_cols)
    if num_fig * n_rows * n_cols < len(gene_list):
        num_fig += 1
    graph = _adj_mtx_to_map(adata.uns['brode_w'])
    label_dic = adata.uns['brode_label_dic']
    label_dic_rev = {}
    for type_ in label_dic:
        label_dic_rev[label_dic[type_]] = type_

    # Plotting
    for i_fig in range(num_fig):
        fig, ax = plt.subplots(3*n_rows, n_cols, figsize=(width*n_cols, height*3*n_rows), facecolor='white')
        if n_cols == 1:
            for i in range(n_cols):
                idx = i_fig*n_rows * n_cols + i
                gidx = np.where(adata.var_names == gene_list[idx])[0][0]
                alpha = adata.varm[f"{key}_alpha"][gidx]
                beta = adata.varm[f"{key}_beta"][gidx]
                gamma = adata.varm[f"{key}_gamma"][gidx]
                t_trans = adata.uns[f"{key}_t_trans"]

                ax[3*i] = _plot_branch(ax[3*i],
                                       t_trans,
                                       alpha,
                                       graph,
                                       label_dic_rev,
                                       plot_depth,
                                       color_map=color_map)
                ax[3*i+1] = _plot_branch(ax[3*i+1],
                                         t_trans,
                                         beta,
                                         graph,
                                         label_dic_rev,
                                         plot_depth,
                                         color_map=color_map)
                ax[3*i+2] = _plot_branch(ax[3*i+2],
                                         t_trans,
                                         gamma,
                                         graph,
                                         label_dic_rev,
                                         plot_depth,
                                         color_map=color_map)

                ax[3*i].set_ylabel(r"$\alpha$", fontsize=label_fontsize, rotation=0)
                ax[3*i+1].set_ylabel(r"$\beta$", fontsize=label_fontsize, rotation=0)
                ax[3*i+2].set_ylabel(r"$\gamma$", fontsize=label_fontsize, rotation=0)
                for k in range(3):
                    ax[3*i+k].set_xticks([])
                    ax[3*i+k].set_yticks([])
                    if plot_depth:
                        ax[3*i+k].set_xlabel("Depth", fontsize=label_fontsize)
                    else:
                        ax[3*i+k].set_xlabel("Time", fontsize=label_fontsize)
                    ax[3*i+k].yaxis.set_label_coords(-0.03, 0.5)
                    ax[3*i+k].set_title(gene_list[idx], fontsize=title_fontsize)
            handles, labels = ax[0].get_legend_handles_labels()
        else:
            for i in range(n_rows):
                for j in range(n_cols):  # i, j: row and column gene index
                    idx = i_fig*n_rows*n_cols+i*n_cols+j  # which gene
                    if idx >= len(gene_list):
                        break
                    idx = i_fig*n_rows*n_cols+i*n_cols+j
                    gidx = np.where(adata.var_names == gene_list[idx])[0][0]
                    alpha = adata.varm[f"{key}_alpha"][gidx]
                    beta = adata.varm[f"{key}_beta"][gidx]
                    gamma = adata.varm[f"{key}_gamma"][gidx]
                    t_trans = adata.uns[f"{key}_t_trans"]

                    ax[3*i, j] = _plot_branch(ax[3*i, j],
                                              t_trans,
                                              alpha,
                                              graph,
                                              label_dic_rev,
                                              color_map=color_map)
                    ax[3*i+1, j] = _plot_branch(ax[3*i+1, j],
                                                t_trans,
                                                beta,
                                                graph,
                                                label_dic_rev,
                                                color_map=color_map)
                    ax[3*i+2, j] = _plot_branch(ax[3*i+2, j],
                                                t_trans,
                                                gamma,
                                                graph,
                                                label_dic_rev,
                                                color_map=color_map)

                    ax[3*i, j].set_ylabel(r"$\alpha$", fontsize=label_fontsize, rotation=0)
                    ax[3*i+1, j].set_ylabel(r"$\beta$", fontsize=label_fontsize, rotation=0)
                    ax[3*i+2, j].set_ylabel(r"$\gamma$", fontsize=label_fontsize, rotation=0)
                    for k in range(3):
                        ax[3*i+k, j].set_xticks([])
                        ax[3*i+k, j].set_yticks([])
                        ax[3*i+k, j].set_xlabel("Time", fontsize=label_fontsize)
                        ax[3*i+k, j].yaxis.set_label_coords(-0.03, 0.5)
                        ax[3*i+k, j].set_title(gene_list[idx], fontsize=title_fontsize)
            handles, labels = ax[0, 0].get_legend_handles_labels()
        plt.tight_layout()

        if bbox_to_anchor is None:
            l_indent = 1 - 0.02/n_rows
            bbox_to_anchor = (-0.03/n_cols, l_indent)
        if legend_fontsize is None:
            legend_fontsize = np.min([int(30*n_rows), int(10*n_cols)])
        # min(n_rows*10, n_rows*120/len(graph.keys()))
        lgd = fig.legend(handles,
                         labels,
                         fontsize=legend_fontsize,
                         markerscale=markerscale,
                         bbox_to_anchor=bbox_to_anchor,
                         loc=legend_loc)

        save = None if figname is None else f'{path}/{figname}_brode_rates_{i_fig+1}.{save_format}'
        save_fig(fig, save, (lgd,))
    return


def plot_trajectory_3d(x_embed,
                       t,
                       cell_labels,
                       plot_arrow=False,
                       n_grid=10,
                       n_time=20,
                       k=30,
                       k_grid=8,
                       scale=1.5,
                       angle=(15, 45),
                       figsize=(8, 6),
                       markersize=MARKERSIZE,
                       label_fontsize=LABEL_FONTSIZE,
                       legend_fontsize=LEGEND_FONTSIZE,
                       markerscale=MARKERSCALE,
                       bbox_to_anchor=(0.0, 1.0, 1.0, -0.05),
                       eps_t=None,
                       color_map=None,
                       embed='umap',
                       save=None,
                       **kwargs):
    """3D quiver plot. x-y plane is a 2D embedding such as UMAP.
    z axis is the cell time. Arrows follow the direction of time to nearby points.

    Args:
        x_embed (:class:`numpy.ndarray`):
            2D embedding for visualization
        t (:class:`numpy.ndarray`):
            Cell time.
        cell_labels (:class:`numpy.ndarray`):
            Cell type annotations.
        plot_arrow (bool, optional):
            Whether to add a quiver plot upon the background 3D scatter plot.
            Defaults to False.
        n_grid (int, optional):
            Grid size of the x-y plane. Defaults to 50.
        n_time (int, optional):
            Grid size of the z (time) axis. Defaults to 20.
        k (int, optional):
            Number of neighbors when computing velocity of each grid point. Defaults to 30.
        k_grid (int, optional):
            Number of neighbors when averaging across the 3D grid. Defaults to 8.
        scale (float, optional):
            Parameter to control boundary detection. Defaults to 1.5.
        angle (tuple, optional):
            Angle of the 3D plot. Defaults to (15, 45).
        figsize (tuple, optional):
            Defaults to (WIDTH, HEIGHT).
        markersize (int, optional):
            Defaults to MARKERSIZE.
        label_fontsize (int, optional):
            Defaults to LABEL_FONTSIZE.
        legend_fontsize (int, optional):
            Defaults to LEGEND_FONTSIZE.
        markerscale (int, optional):
            Defaults to 3.
        bbox_to_anchor (tuple, optional):
            Defaults to (0.0, 1.0, 1.0, -0.05).
        eps_t (float, optional):
            Parameter to control the relative time order of cells. Defaults to None.
        color_map (str, optional):
            Defaults to None.
        embed (str, optional):
            Name of the embedding.. Defaults to 'umap'.
        save (str, optional):
            Figure name for saving (including path). Defaults to None.

    """
    t_clip = np.clip(t, np.quantile(t, 0.01), np.quantile(t, 0.99))
    range_z = np.max(x_embed.max(0) - x_embed.min(0))
    w = range_z/(t_clip.max()-t_clip.min())
    x_3d = np.concatenate((x_embed, (t_clip - t_clip.min()).reshape(-1, 1)*w), 1)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    ax.view_init(angle[0], angle[1])
    # Plot cells by label
    cell_types = np.unique(cell_labels)
    colors = get_colors(len(cell_types), color_map)
    for i, type_ in enumerate(cell_types):
        cell_mask = cell_labels == type_
        d = max(1, np.sum(cell_mask)//3000)
        ax.scatter(x_3d[:, 0][cell_mask][::d],
                   x_3d[:, 1][cell_mask][::d],
                   x_3d[:, 2][cell_mask][::d],
                   s=markersize,
                   color=colors[i],
                   label=type_,
                   edgecolor='none')
    if plot_arrow:
        # Used for filtering target grid points
        knn_model_2d = pynndescent.NNDescent(x_embed, n_neighbors=k)

        # Compute the time on a grid
        knn_model = pynndescent.NNDescent(x_3d, n_neighbors=k+20)
        ind, dist = knn_model.neighbor_graph
        dist_thred = dist.mean() * scale

        x = np.linspace(x_3d[:, 0].min(), x_3d[:, 0].max(), n_grid)
        y = np.linspace(x_3d[:, 1].min(), x_3d[:, 1].max(), n_grid)
        z = np.linspace(x_3d[:, 2].min(), x_3d[:, 2].max(), n_time)

        xgrid, ygrid, zgrid = np.meshgrid(x, y, z)
        xgrid, ygrid, zgrid = xgrid.flatten(), ygrid.flatten(), zgrid.flatten()
        xyz_grid = np.stack([xgrid, ygrid, zgrid]).T

        neighbors_grid, dist_grid = knn_model.query(xyz_grid, k=k)
        mask = np.quantile(dist_grid, 0.5, 1) <= dist_thred

        # transition probability on UMAP
        def transition_prob(dist, sigma):
            P = np.exp(-(np.clip(dist/sigma, -5, None))**2)
            psum = P.sum(1).reshape(-1, 1)
            psum[psum == 0] = 1.0
            P = P/psum
            return P

        P = transition_prob(dist_grid, dist_thred)
        tgrid = np.sum(np.stack([t[neighbors_grid[i]] for i in range(len(xgrid))])*P, 1)
        tgrid = tgrid[mask]

        # Compute velocity based on grid time
        # filter out distant grid points
        knn_grid = pynndescent.NNDescent(xyz_grid[mask], n_neighbors=k_grid, metric="l2")
        neighbor_grid, dist_grid = knn_grid.neighbor_graph

        if eps_t is None:
            eps_t = (t_clip.max()-t_clip.min())/len(t)*10
        delta_t = tgrid[neighbor_grid] - tgrid.reshape(-1, 1) - eps_t

        sigma_t = (t_clip.max()-t_clip.min())/n_grid
        ind_grid_2d, dist_grid_2d = knn_model_2d.query(xyz_grid[mask], k=k_grid)
        dist_thred_2d = (dist_grid_2d.mean(1)+dist_grid_2d.std(1)).reshape(-1, 1)
        # Filter out backflow and distant points in 2D space
        P = (np.exp((np.clip(delta_t/sigma_t, -5, 5))**2))*((delta_t >= 0) & (dist_grid_2d <= dist_thred_2d))
        psum = P.sum(1).reshape(-1, 1)
        psum[psum == 0] = 1.0
        P = P/psum

        delta_x = (xgrid[mask][neighbor_grid] - xgrid[mask].reshape(-1, 1))
        delta_y = (ygrid[mask][neighbor_grid] - ygrid[mask].reshape(-1, 1))
        delta_z = (zgrid[mask][neighbor_grid] - zgrid[mask].reshape(-1, 1))
        norm = np.sqrt(delta_x**2+delta_y**2+delta_z**2)
        norm[norm == 0] = 1.0
        vx_grid_filter = ((delta_x/norm)*P).sum(1)
        vy_grid_filter = ((delta_y/norm)*P).sum(1)
        vz_grid_filter = ((delta_z/norm)*P).sum(1)
        # KNN Smoothing
        vx_grid_filter = vx_grid_filter[neighbor_grid].mean(1)
        vy_grid_filter = vy_grid_filter[neighbor_grid].mean(1)
        vz_grid_filter = vz_grid_filter[neighbor_grid].mean(1)

        vx_grid = np.zeros((n_grid*n_grid*n_time))
        vy_grid = np.zeros((n_grid*n_grid*n_time))
        vz_grid = np.zeros((n_grid*n_grid*n_time))
        vx_grid[mask] = vx_grid_filter
        vy_grid[mask] = vy_grid_filter
        vz_grid[mask] = vz_grid_filter

        range_x = np.mean(x_embed.max(0) - x_embed.min(0))
        ax.quiver(xgrid.reshape(n_grid, n_grid, n_time),
                  ygrid.reshape(n_grid, n_grid, n_time),
                  zgrid.reshape(n_grid, n_grid, n_time),
                  vx_grid.reshape(n_grid, n_grid, n_time),
                  vy_grid.reshape(n_grid, n_grid, n_time),
                  vz_grid.reshape(n_grid, n_grid, n_time),
                  color='k',
                  length=(0.8*range_x/n_grid + 0.8*range_x/n_time),
                  normalize=True)

    ax.set_xlabel(f'{embed} 1', fontsize=label_fontsize)
    ax.set_ylabel(f'{embed} 2', fontsize=label_fontsize)
    ax.set_zlabel('Time', fontsize=label_fontsize)

    ncol = kwargs['ncol'] if 'ncol' in kwargs else 4
    lgd = ax.legend(fontsize=legend_fontsize, ncol=ncol, markerscale=markerscale, bbox_to_anchor=bbox_to_anchor, loc='center')
    fig.tight_layout()
    if 'axis_off' in kwargs:
        ax.axis('off')
    save_fig(fig, save, (lgd,))


def get_inv_degree_mtx(graph):
    """Get the inversed degree matrix of a graph.

    Args:
        graph (array-like):
            An adjacency matrix.
    """
    return np.diag(1/np.sum(graph, 1))


def plot_trajectory_4d(x_spatial,
                       v,
                       cell_labels,
                       radius,
                       principal_curve=None,
                       plot_anchors=True,
                       plot_arrow=False,
                       n_grid=10,
                       scale=1.5,
                       smooth_factor=0.05,
                       palette=None,
                       angle=(15, 45),
                       figsize=(8, 6),
                       arrow_length=5,
                       arrow_length_ratio=0.5,
                       markersize=MARKERSIZE,
                       label_fontsize=12,
                       legend_fontsize=12,
                       legend_loc='center',
                       markerscale=2.0,
                       alpha=1.0,
                       tick_fontsize=10,
                       linewidth=0.5,
                       labelpad=(-10, -10, -10),
                       tick_labelpad=-3,
                       bbox_to_anchor=(0.0, 1.0, 1.0, -0.05),
                       zoom=1.0,
                       embed='spatial',
                       real_aspect_ratio=True,
                       save=None,
                       **kwargs):
    """3D quiver plot. Data are visualized in a 3D embedding such as UMAP.
    Additionally, we use time information to pick a future direction.

    Args:
        x_spatial (:class:`numpy.ndarray`):
            3D embedding for visualization
        v (:class:`numpy.ndarray`):
            Velocity of cells.
        cell_labels (:class:`numpy.ndarray`):
            Cell type annotations.
        radius (float):
            Radius for building an epsilon-ball graph.
        principal_curve (:class:`numpy.ndarray`, optional):
            Principal curve. Defaults to None.
        plot_anchors (bool, optional):
            Whether to plot the line connecting anchors of the principal curve. Defaults to True.
        plot_arrow (bool, optional):
            Whether to add a quiver plot upon the background 3D scatter plot.
            Defaults to False.
        n_grid (int, optional):
            Grid size of the x-y plane. Defaults to 50.
        scale (float, optional):
            Parameter to control boundary detection. Defaults to 1.5.
        smooth_factor (float, optional):
            Gaussian smoothing factor. Defaults to 0.05.
        palette (array-like, optional):
            Color palette for cell types. Defaults to None.
        angle (tuple, optional):
            Angle of the 3D plot. Defaults to (15, 45).
        figsize (tuple, optional):
            Defaults to (WIDTH, HEIGHT).
        arrow_length (int, optional):
            Length of the arrow. Defaults to 10.
        arrow_length_ratio (float, optional):
            Ratio of the arrow length. Defaults to 0.5.
        markersize (int, optional):
            Defaults to MARKERSIZE.
        alpha (float, optional):
            Transparency of the dots. Defaults to 1.0.
        label_fontsize (int, optional):
            Defaults to LABEL_FONTSIZE.
        legend_fontsize (int, optional):
            Defaults to LEGEND_FONTSIZE.
        legend_loc (str, optional):
            Location of the legend. Defaults to 'center'.
        markerscale (int, optional):
            Defaults to 3.
        tick_fontsize (int, optional):
            Defaults to 10.
        linewidth (float, optional):
            Defaults to 0.5.
        bbox_to_anchor (tuple, optional):
            Defaults to (0.0, 1.0, 1.0, -0.05).
        zoom (float, optional):
            Defaults to 1.0.
        embed (str, optional):
            Name of the embedding.. Defaults to 'umap'.
        real_aspect_ratio (bool, optional):
            Whether to use the aspect ratio proportional to the real coordinate values.
        save (str, optional):
            Figure name for saving (including path). Defaults to None.

    Returns:
        tuple: A tuple of 3D grid coordinates and smoothed velocity.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    ax.view_init(angle[0], angle[1])
    # Plot cells by label
    cell_types = np.unique(cell_labels)
    if palette is None:
        palette = get_colors(len(cell_types))
    for i, type_ in enumerate(cell_types):
        cell_mask = cell_labels == type_
        d = max(1, np.sum(cell_mask)//3000)
        ax.scatter(x_spatial[:, 0][cell_mask][::d],
                   x_spatial[:, 1][cell_mask][::d],
                   x_spatial[:, 2][cell_mask][::d],
                   s=markersize,
                   color=palette[i],
                   alpha=alpha,
                   label=type_,
                   edgecolor='none')

    if isinstance(principal_curve, np.ndarray):
        if not plot_anchors:
            ax.scatter(principal_curve[:, 0], principal_curve[:, 1], principal_curve[:, 2], s=markersize*2, color='k', edgecolor='none')
        else:
            ax.plot(principal_curve[:, 0], principal_curve[:, 1], principal_curve[:, 2], 'k-', linewidth=5)
    xyz_grid, v_grid_sm = None, None
    if plot_arrow:
        # Compute the velocity on a grid
        bt = BallTree(x_spatial)

        x = np.linspace(x_spatial[:, 0].min(), x_spatial[:, 0].max(), n_grid)
        y = np.linspace(x_spatial[:, 1].min(), x_spatial[:, 1].max(), n_grid)
        z = np.linspace(x_spatial[:, 2].min(), x_spatial[:, 2].max(), n_grid)
        grid_dist = np.ptp(x) / (n_grid - 1)

        xgrid, ygrid, zgrid = np.meshgrid(x, y, z)
        xgrid, ygrid, zgrid = xgrid.flatten(), ygrid.flatten(), zgrid.flatten()
        xyz_grid = np.stack([xgrid, ygrid, zgrid]).T
        # Find distance thredshold
        out = bt.query_radius(x_spatial, radius, return_distance=True)
        spatial_nbs, dist = out[0], out[1]
        dist = np.concatenate(dist)
        dist_thred = dist.mean() * scale
        
        # Find neighbors
        out = bt.query_radius(xyz_grid, radius, return_distance=True)
        spatial_nbs, dist_grid = out[0], out[1]
        mid_dist_grid = np.array([np.median(x) for x in dist_grid])
        mask = mid_dist_grid <= dist_thred

        # transition probability on UMAP
        def transition_prob(dist, sigma):
            _dist = np.clip(dist/sigma, -5, None)
            p = np.exp(-_dist)
            psum = p.sum()
            psum += int(psum == 0)
            p = p/psum
            return p
        v_grid = np.zeros((len(xyz_grid), 3))
        for i, (nbs, dist) in enumerate(zip(spatial_nbs, dist_grid)):
            if mask[i]:
                p = transition_prob(dist, dist_thred).reshape(-1, 1)
                v_grid[i] = np.sum(v[nbs] * p, 0)

        # gaussian smoothing
        v_grid = v_grid.reshape(n_grid, n_grid, n_grid, 3)
        v_grid_sm = np.zeros((n_grid*n_grid*n_grid, 3))
        for i in range(3):
            v_grid_sm[:, i] = gaussian_filter(v_grid[:, :, :, i],
                                              grid_dist*smooth_factor,
                                              mode="nearest",
                                              radius=max(int(n_grid*0.1), 3)).flatten()
        v_grid_sm[~mask] = 0

        ax.quiver(xgrid.reshape(n_grid, n_grid, n_grid),
                  ygrid.reshape(n_grid, n_grid, n_grid),
                  zgrid.reshape(n_grid, n_grid, n_grid),
                  v_grid_sm[:, 0].reshape(n_grid, n_grid, n_grid),
                  v_grid_sm[:, 1].reshape(n_grid, n_grid, n_grid),
                  v_grid_sm[:, 2].reshape(n_grid, n_grid, n_grid),
                  color='k',
                  normalize=True,
                  length=arrow_length,
                  arrow_length_ratio=arrow_length_ratio,
                  linewidth=linewidth)
    if real_aspect_ratio:
        ax.set_box_aspect((np.ptp(x_spatial[:, 0]), np.ptp(x_spatial[:, 1]), np.ptp(x_spatial[:, 2])), zoom=zoom)
    if not 'labels_off' in kwargs:
        ax.set_xlabel(f'{embed} 1', fontsize=label_fontsize, labelpad=labelpad[0])
        ax.set_ylabel(f'{embed} 2', fontsize=label_fontsize, labelpad=labelpad[1])
        ax.set_zlabel(f'{embed} 3', fontsize=label_fontsize, labelpad=labelpad[2])
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, pad=tick_labelpad)

    ncol = kwargs['ncol'] if 'ncol' in kwargs else 4
    lgd = ax.legend(fontsize=legend_fontsize, ncol=ncol, markerscale=markerscale, bbox_to_anchor=bbox_to_anchor, loc=legend_loc)
    plt.tight_layout()
    if 'ticks_off' in kwargs:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    if save is not None:
        save_fig(fig, save)
    return xyz_grid, v_grid_sm


def plot_rate_hist(adata,
                   model,
                   key,
                   tprior='tprior',
                   figsize=(WIDTH * 3, HEIGHT),
                   label_fontsize=LABEL_FONTSIZE,
                   save="figures/hist.png"):
    """Convert rate parameters to real interpretable units and plot the histogram

    Args:
        adata (:class:`anndata.AnnData`):
            AnnData object.
        model (str):
            Model name.
        key (str):
            Key for retreiving model predictions/inferred parameters and other data.
        tprior (str, optional):
            Key for capture time.
            This is used to convert rates to transcript/minute.
            If not provided or doesn't exist in adata, we assume the experiment lasts
            one day. Defaults to 'tprior'.
        figsize (tuple, optional):
            Defaults to (18, 4).
        save (str, optional):
            Figure name for saving (including path). Defaults to "figures/hist.png".
    """
    if 'Discrete' in model:
        U, S = adata.layers[f"{key}_uhat"], adata.layers[f"{key}_shat"]
    else:
        U, S = adata.layers["Mu"], adata.layers["Ms"]
    x_median = np.median(U.sum(1)) + np.median(S.sum(1))
    sparsity_u = [1 - np.sum(U[:, i] <= U[:, i].max()*0.01)/adata.n_obs/adata.n_vars for i in range(adata.n_vars)]
    sparsity_s = [1 - np.sum(S[:, i] <= S[:, i].max()*0.01)/adata.n_obs/adata.n_vars for i in range(adata.n_vars)]
    u_median = np.array([np.quantile(U[:, i], 0.5+0.5*sparsity_u[i]) for i in range(adata.n_vars)])
    s_median = np.array([np.quantile(S[:, i], 0.5+0.5*sparsity_s[i]) for i in range(adata.n_vars)])
    sparsity_scale = (360000 / x_median)
    t = adata.obs[f"{key}_time"].to_numpy()
    if tprior in adata.obs:
        tprior = adata.obs[tprior].to_numpy()
        t_scale = (tprior.max()-tprior.min()) / (np.quantile(t, 0.99)-np.quantile(t, 0.01))
    else:
        print('Warning: No multiple capture times detected! Assume the experiment lasts one day.')
        t_scale = 1 / (np.quantile(t, 0.99) - np.quantile(t, 0.01))
    if "Full VB" in model:
        std_alpha = np.exp(adata.var[f"{key}_logstd_alpha"].to_numpy())
        std_beta = np.exp(adata.var[f"{key}_logstd_beta"].to_numpy())
        std_gamma = np.exp(adata.var[f"{key}_logstd_gamma"].to_numpy())
        alpha = np.exp(adata.var[f"{key}_logmu_alpha"].to_numpy()+0.5*std_alpha**2)\
            / (1440*t_scale) * sparsity_scale
        beta = np.exp(adata.var[f"{key}_logmu_beta"].to_numpy()+0.5*std_beta**2) * u_median\
            / (1440*t_scale) * sparsity_scale
        gamma = np.exp(adata.var[f"{key}_logmu_gamma"].to_numpy()+0.5*std_gamma**2) * s_median\
            / (1440*t_scale) * sparsity_scale
    elif "VeloVAE" or "TopoVelo" in model:
        alpha = (adata.var[f"{key}_alpha"]).to_numpy() / (1440*t_scale) * sparsity_scale
        beta = (adata.var[f"{key}_beta"]).to_numpy() * u_median / (1440*t_scale) * sparsity_scale
        gamma = (adata.var[f"{key}_gamma"]).to_numpy() * s_median / (1440*t_scale) * sparsity_scale
    ub = [min(np.quantile(alpha, 0.95), alpha.mean()*4),
          min(np.quantile(beta, 0.95), beta.mean()*4),
          min(np.quantile(gamma, 0.95), gamma.mean()*4)]

    fig, ax = plt.subplots(1, 3, figsize=figsize)
    ax[0].hist(alpha, bins=np.linspace(0, ub[0], 50), color="orange", label=r"$\alpha$")
    ax[1].hist(beta, bins=np.linspace(0, ub[1], 50), color="green", label=r"$\beta$")
    ax[2].hist(gamma, bins=np.linspace(0, ub[2], 50), color="blue", label=r"$\gamma$")
    ax[0].set_xlabel(r"$\alpha$ (transcript / min)", fontsize=label_fontsize)
    ax[1].set_xlabel(r"$\beta$u (transcript / min)", fontsize=label_fontsize)
    ax[2].set_xlabel(r"$\gamma$s (transcript / min)", fontsize=label_fontsize)
    ax[0].set_ylabel("Number of Genes", fontsize=label_fontsize)
    plt.tight_layout()
    save_fig(fig, save)
