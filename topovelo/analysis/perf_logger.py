import numpy as np
import pandas as pd
import os
from ..plotting import get_colors
import re
import matplotlib.pyplot as plt

MARKERS = ["o", "v", "x", "s", "+", "d", "1", "*", "^", "p", "h"]


class PerfLogger:
    """Class for saving the performance metrics
    """
    def __init__(self, save_path='perf', checkpoints=None):
        """Constructor

        Args:
            save_path (str, optional):
                Path for saving the data frames to .csv files. Defaults to 'perf'.
            checkpoints (list[str], optional):
                Existing results to load (.csv). Defaults to None.
        """
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        self.n_dataset = 0
        self.n_model = 0
        self.metrics = ["MSE Train",
                        "MSE Test",
                        "MAE Train",
                        "MAE Test",
                        "LL Train",
                        "LL Test",
                        "CBDir (Gene Space)",
                        "CBDir",
                        "Time Score",
                        "In-Cluster Coherence",
                        "Velocity Consistency",
                        "Spatial Velocity Consistency",
                        "Spatial Time Consistency",
                        "Time Correlation"]
        self.multi_metrics = ["K-CBDir (Gene Space)",
                              "K-CBDir",
                              "Mann-Whitney Test (Gene Space)",
                              "Mann-Whitney Test",
                              "Mann-Whitney Test Stats (Gene Space)",
                              "Mann-Whitney Test Stats"]
        self.metrics_type = ["CBDir",
                             "CBDir (Embed)",
                             "Time Score"]
        if checkpoints is None:
            self._create_empty_df()
        else:
            self.df = pd.read_csv(checkpoints[0], header=[0], index_col=[0, 1])
            self.df_type = pd.read_csv(checkpoints[1], header=[0, 1], index_col=[0, 1])
            self.df_multi = pd.read_csv(checkpoints[2], header=[0, 1], index_col=[0, 1])
            self.df_multi_type = pd.read_csv(checkpoints[3], header=[0, 1, 2], index_col=[0, 1])

    def _create_empty_df(self):
        row_mindex = pd.MultiIndex.from_arrays([[], []], names=["Metrics", "Model"])
        col_index = pd.Index([], name='Dataset')
        col_mindex = pd.MultiIndex.from_arrays([[], []], names=["Dataset", "Transition"])
        self.df = pd.DataFrame(index=row_mindex, columns=col_index)
        self.df_type = pd.DataFrame(index=row_mindex, columns=col_mindex)

        col_mindex_2 = pd.MultiIndex.from_arrays([[], []], names=["Dataset", "Step"])
        self.df_multi = pd.DataFrame(index=row_mindex, columns=col_mindex_2)

        col_mindex_3 = pd.MultiIndex.from_arrays([[], [], []], names=["Dataset", "Transition", "Step"])
        self.df_multi_type = pd.DataFrame(index=row_mindex, columns=col_mindex_3)

    def insert(self, data_name, res, res_type, multi_res, multi_res_type):
        """Insert the performance evaluation results from topovelo.post_analysis

        Args:
            data_name (str):
                Name of the dataset
            res (:class:`pandas.DataFrame`):
                Contains performance metrics for the entire dataset.
                Rows are the performance metrics.
                Columns are model names.
            res_type (:class:`pandas.DataFrame`):
                Contains the velocity and time metrics for each pair of
                cell type transition. Rows are different performance metrics (1 level),
                while columns are indexed by method and cell type transitions (2 levels).
            multi_res (:class:`pandas.DataFrame`):
                Similar to "res" except that the performance metrics are multi-dimensional.
                Column index has 2 levels (method and number of steps)
            multi_res_type (:class:`pandas.DataFrame`):
                Similar to "res_type" except that the performance metrics are multi-dimensional.
                Column index has 3 levels (method, transition pair and number of steps)
        """
        self.n_dataset += 1
        methods = res.columns.unique(0)

        # Collapse the dataframe to 1D series with multi-index
        res_1d = pd.Series(res.values.flatten(), index=pd.MultiIndex.from_product([res.index, res.columns]))
        for x in res_1d.index:
            self.df.loc[x, data_name] = res_1d.loc[x]

        # Reshape the data in res_type to match the multi-row-index in self.df_type
        if res_type.shape[1] > 0:
            res_reshape = pd.DataFrame(res_type.values.reshape(res_type.shape[0] * len(methods), -1),
                                    index=pd.MultiIndex.from_product([res_type.index, methods]),
                                    columns=pd.MultiIndex.from_product([[data_name], res_type.columns.unique(1)]))
            self.df_type = pd.concat([self.df_type, res_reshape], axis=1)
        else:
            print('Warning: no cell type transition pair found. Skipped insertion to df_type.')

        # Multi-dimensional metrics
        res_reshape = pd.DataFrame(multi_res.values.reshape(multi_res.shape[0] * len(methods), -1),
                                   index=pd.MultiIndex.from_product([multi_res.index, methods]),
                                   columns=pd.MultiIndex.from_product([[data_name],
                                                                       multi_res.columns.unique(1)]))
        self.df_multi = pd.concat([self.df_multi, res_reshape], axis=1)

        # Multi-dimensional metrics for each transition pair
        if multi_res_type.shape[1] > 0:
            res_reshape = pd.DataFrame(multi_res_type.values.reshape(multi_res_type.shape[0] * len(methods), -1),
                                       index=pd.MultiIndex.from_product([multi_res_type.index, methods]),
                                       columns=pd.MultiIndex.from_product([[data_name],
                                                                          multi_res_type.columns.unique(1),
                                                                          multi_res_type.columns.unique(2)]))
            self.df_multi_type = pd.concat([self.df_multi_type, res_reshape], axis=1)
        else:
            print('Warning: no cell type transition pair found. Skipped insertion to df_multi_type.')

        # update number of models
        self.n_model = len(self.df.index.unique(1))

        return

    def plot_summary(self, metrics=[], methods=None, figure_path=None, dpi=100):
        """Generate boxplots showing the overall performance metrics.
        Each plot shows one metric over all datasets, with methods as x-axis labels and 

        Args:
            metrics (list[str], optional):
                Performance metric to plot.
                If set to None, all metrics will be plotted.
            methods (list[str], optional):
                Methods to compare.
                If set to None, all existing methods will be included.
            figure_path (str, optional):
                Path to the folder for saving figures.
                If set to None, figures will not be saved.
                Defaults to None.
            bbox_to_anchor (tuple, optional):
                Location of the legend. Defaults to (1.25, 1.0).
        """
        n_model = self.n_model if methods is None else len(methods)
        colors = get_colors(n_model)
        if methods is None:
            methods = np.array(self.df.index.unique(1)).astype(str)
        for metric in metrics:
            if metric in self.df.index:
                df_plot = self.df.loc[metric]
            elif metric in self.df_multi.index.unique(0):
                df_plot = self.df_multi.loc[metric]
            else:
                continue
            if methods is not None:
                df_plot = df_plot.loc[methods]
            vals = df_plot.values.T
            fig, ax = plt.subplots(figsize=(1.6*n_model+3, 4))
            # rectangular box plot
            bplot = ax.boxplot(vals,
                               vert=True,  # vertical box alignment
                               patch_artist=True,  # fill with color
                               labels=df_plot.index.to_numpy())  # will be used to label x-ticks
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
            for line in bplot['medians']:
                line.set_color('black')
            for line in bplot['means']:
                line.set_color('black')
            ax.set_xlabel("")          
            ax.set_title(metric)
            ax.set_xticks(range(1, n_model+1), methods, rotation=0)
            ax.tick_params(axis='both', which='major', labelsize=15)
            # ax.grid()
            fig = ax.get_figure()
            fig.tight_layout()
            if figure_path is not None:
                fig_name = re.sub(r'\W+', ' ', metric.lower())
                fig_name = '_'.join(fig_name.rstrip().split())
                fig.savefig(f'{figure_path}/{metric}_summary.png', dpi=dpi, bbox_inches='tight')

        return

    def plot_transition_pairs(self, metrics=[], methods=None, figure_path=None, bbox_to_anchor=(1.25, 1.0), dpi=100):
        """Plot performance metrics for each transition pair given knowledge about cell type transition in a dataset.

        Args:
            metrics (list[str], optional):
                Performance metrics to plot. Defaults to [].
            figure_path (str, optional):
                Path to the folder for saving figures.
                If set to None, figures will not be saved.
                Defaults to None.
            bbox_to_anchor (tuple, optional):
                . Defaults to (1.25, 1.0).
        """
        datasets = self.df_type.columns.unique(0)
        if methods is None:
            methods = np.array(self.df.index.unique(1)).astype(str)
        for metric in metrics:
            if metric not in self.metrics_type:
                continue
            fig_name = re.sub(r'\W+', ' ', metric.lower())
            fig_name = '_'.join(fig_name.rstrip().split())
            for dataset in datasets:
                if np.all(np.isnan(self.df_type.loc[metric, dataset].values)):
                    continue
                colors = get_colors(self.df_type.loc[metric, dataset].shape[0])
                df_plot = self.df_type.loc[metric, dataset].loc[methods].T
                ax = df_plot.plot.bar(color=colors, figsize=(12, 6), fontsize=14)
                ax.set_title(metric, fontsize=20)
                if isinstance(bbox_to_anchor, tuple):
                    ax.legend(fontsize=16, loc=1, bbox_to_anchor=bbox_to_anchor)
                transition_pairs = self.df_type[dataset].columns.unique(0)
                ax.set_xticklabels(transition_pairs, rotation=0)
                ax.set_xlabel("")
                # ax.grid()
                fig = ax.get_figure()
                fig.tight_layout()
                if figure_path is not None:
                    fig.savefig(f'{figure_path}/perf_{fig_name}_{dataset}.png', bbox_inches='tight', dpi=dpi)
        return

    def _plot_velocity_metrics_ax(self,
                                  metric,
                                  dataset,
                                  methods,
                                  show_legend=True,
                                  ax=None,
                                  **kwargs):
        colors = get_colors(len(methods))
        steps = self.df_multi.columns.unique(1)
        for i, model in enumerate(methods):
            ax = self.df_multi.loc[(metric, model), dataset].plot(marker=MARKERS[i],
                                                                  markersize=kwargs['markersize'],
                                                                  color=colors[i],
                                                                  label=model,
                                                                  ax=ax,
                                                                  linewidth=kwargs['linewidth'],
                                                                  figsize=kwargs['figsize'])
        
        ax.set_xticks(range(len(steps)), range(1, len(steps)+1), rotation=0)
        if show_legend:
            ncols = 1 if 'ncols' not in kwargs else kwargs['ncols']
            if 'bbox_to_anchor' in kwargs:
                ax.legend(fontsize=kwargs['legend_fontsize'],
                            ncol=ncols,
                            loc='center',
                            bbox_to_anchor=kwargs['bbox_to_anchor'])
            else:
                ax.legend(fontsize=kwargs['legend_fontsize'], ncol=ncols)
        ax.set_title(dataset, fontsize=kwargs['title_fontsize'])
        # ax.grid()
        ax.set_xlabel("Step Size", fontsize=kwargs['ylabel_fontsize'])
        ax.set_ylabel(metric, fontsize=kwargs['ylabel_fontsize'], labelpad=kwargs['labelpad'])
        ax.tick_params(axis='both', which='major', labelsize=kwargs['tick_fontsize'])
        return ax

    def plot_velocity_metrics(self,
                              metrics=[],
                              datasets=[],
                              methods=[],
                              figure_path=None,
                              figsize=(5, 6),
                              markersize=6,
                              linewidth=1.0,
                              title_fontsize=20,
                              legend_fontsize=8,
                              tick_fontsize=12,
                              ylabel_fontsize=15,
                              labelpad=10,
                              legend_ncols=None,
                              bbox_to_anchor=(0, 1, 1, 0.2),
                              dpi=100):
        """Generate markered line plots of K-CBDir and related test results.
        Each plot only considers one single dataset.

        Args:
            dataset (str, optional):
                Dataset to plot.
                If set to None, the functions will generate a single plot for each dataset.
                Defaults to None.
            methods (list[str], optional):
                Methods to compare.
                If set to None, all existing methods will be included.
            figure_path (str, optional):
                Path to the folder for saving figures.
                If set to None, figures will not be saved.
                Defaults to None.
            bbox_to_anchor (tuple, optional):
                Location of the legend. Defaults to (0, 1, 1, 0.1).
        """
        if len(datasets) == 0:
            datasets = list(self.df.columns.unique(0))
        if len(methods) == 0:
            methods = np.array(self.df.index.unique(1)).astype(str)
        if len(metrics) == 0:
            metrics = self.multi_metrics
        if legend_ncols is None:
            legend_ncols = len(methods)

        for metric in metrics:
            metric_name = re.sub(r'\W+', ' ', metric.lower())
            metric_name = '_'.join(metric_name.rstrip().split())
            for dataset in datasets:
                data_name = '-'.join(dataset.rstrip().split())
                fig_name = metric_name+"_"+data_name
                ax = self._plot_velocity_metrics_ax(metric,
                                                    dataset,
                                                    methods,
                                                    figsize,
                                                    figsize=figsize,
                                                    markersize=markersize,
                                                    linewidth=linewidth,
                                                    title_fontsize=title_fontsize,
                                                    legend_fontsize=legend_fontsize,
                                                    tick_fontsize=tick_fontsize,
                                                    ylabel_fontsize=ylabel_fontsize,
                                                    labelpad=labelpad,
                                                    ncols=legend_ncols,
                                                    bbox_to_anchor=bbox_to_anchor)
                fig = ax.get_figure()
                fig.tight_layout()
                if figure_path is not None:
                    fig.savefig(f'{figure_path}/{fig_name}_{dataset}.png', bbox_inches='tight', dpi=dpi)
                plt.show(fig)
                plt.close()
        return

    def _plot_metrics_ax(self,
                         metric,
                         datasets,
                         methods,
                         show_legend=True,
                         ax=None,
                         **kwargs):
        colors = get_colors(len(methods))
        df_plot = self.df.loc[metric].loc[methods].T
        ax = df_plot.plot.bar(color=colors,
                              figsize=kwargs['figsize'],
                              legend=show_legend,
                              ax=ax)
        ax.set_xlabel("")
        ax.set_xticklabels(datasets, rotation=0)
        ax.set_title(metric, fontsize=kwargs['title_fontsize'])
        # ax.grid()
        ax.tick_params(axis='both', which='major', labelsize=kwargs['tick_fontsize'])
        if show_legend:
            ncols = 1 if 'ncols' not in kwargs else kwargs['ncols']
            if 'bbox_to_anchor' in kwargs:
                loc = 'center' if len(kwargs['bbox_to_anchor']) == 4 else 1
                ax.legend(fontsize=kwargs['legend_fontsize'], loc=loc, ncols=ncols, bbox_to_anchor=kwargs['bbox_to_anchor'])
            else:
                ax.legend(fontsize=kwargs['legend_fontsize'], ncols=ncols)
        return ax

    def plot_metrics(self,
                     metrics=[],
                     datasets=[],
                     methods=[],
                     figure_path=None,
                     figsize=(12, 6),
                     title_fontsize=20,
                     legend_fontsize=16,
                     tick_fontsize=15,
                     bbox_to_anchor=(1.25, 1.0),
                     dpi=100):
        """Generate bar plots showing all scalar performance metrics.
        Each plot has different datasets as x-axis labels and different bars represent methods.

        Args:
            figure_path (str, optional):
                Path to the folder for saving figures.
                If set to None, figures will not be saved.
                Defaults to None.
            bbox_to_anchor (tuple, optional):
                Location of the legend. Defaults to (1.25, 1.0).
        """
        if len(datasets) == 0:
            datasets = self.df.columns.unique(0)
        if len(methods) == 0:
            methods = np.array(self.df.index.unique(1)).astype(str)
        if len(metrics) == 0:
            metrics = self.metrics

        for metric in metrics:
            if metric not in self.metrics:
                continue
            fig_name = re.sub(r'\W+', ' ', metric.lower())
            fig_name = '_'.join(fig_name.rstrip().split())
            if np.all(np.isnan(self.df.loc[metric, :].values)):
                continue
            ax = self._plot_metrics_ax(metric,
                                       datasets,
                                       methods,
                                       figsize=figsize,
                                       title_fontsize=title_fontsize,
                                       legend_fontsize=legend_fontsize,
                                       tick_fontsize=tick_fontsize,
                                       bbox_to_anchor=bbox_to_anchor)
            fig = ax.get_figure()
            fig.tight_layout()
            if figure_path is not None:
                fig.savefig(f'{figure_path}/perf_{fig_name}.png', bbox_inches='tight', dpi=dpi)
        return

    def _decompose_num(self, n, max_ratio=4):
        # max_ratio defines the upper limit of ncols/nrows 
        if n <= 4:
            return [n]
        ncols = int(np.ceil(np.sqrt(n)))
        # Find any divider of n fulfilling the max_ratio criterium
        for i in range(int(np.floor(np.sqrt(n))), 0, -1):
            if n % i == 0 and (n // i) / i <= max_ratio:
                return [n//i]*i
        # If on such i exists, we cannot divide n subfigures into a rectangular grid
        r = n % ncols
        d = n // ncols
        if r <= ncols // 2:
            out = [ncols]*d
            for i in range(1, r+1):
                out[-i] += 1
        else:
            out = [ncols]*(d+1)
            out[-1] = out[0] - 1
            if r < ncols - 1:
                for i in range(2, ncols - r):
                    out[-i] -= 1

        return out

    def _auto_grid_size(self, datasets, metrics):
        n_scalar = np.sum([metric in self.metrics for metric in metrics])
        n_multi = len(metrics) - n_scalar
        n_dataset = len(datasets)
        ncols_scalar, ncols_multi = self._decompose_num(n_scalar), self._decompose_num(n_dataset*n_multi)
        nrows = len(ncols_scalar) + len(ncols_multi)
        return n_scalar, nrows, ncols_scalar, ncols_multi

    def plot(self,
             metrics,
             legend_metric,
             datasets=[],
             methods=[],
             grid_size_params=None,
             figure_path=None,
             figure_name='perf',
             figsize=(7.5, 5),
             markersize=3,
             linewidth=1.0,
             title_fontsize=8,
             legend_fontsize=5,
             ylabel_fontsize=6,
             labelpad=5,
             tick_fontsize=6,
             ncols_legend=None,
             hspace=0.3,
             wspace=0.12,
             bbox_to_anchor=(0, 1, 1, 0.8),
             dpi=300,
             save_format='png'):
        if len(datasets) == 0:
            datasets = self.df.columns.unique(0)

        # Automatically determine the grid size
        if grid_size_params is None:
            n_scalar, nrows, ncols_scalar, ncols_multi = self._auto_grid_size(datasets, metrics)
        else:
            ncols_scalar = grid_size_params['ncols_scalar']
            n_scalar = len(ncols_scalar)
            ncols_multi = grid_size_params['ncols_multi']
            nrows = len(ncols_scalar) + len(ncols_multi)

        fig = plt.figure(figsize=figsize, facecolor='white')  # (nrows*figsize[0], max_ncol*figsize[1])

        counter_scalar, ptr_scalar = 1, 0
        counter_multi, ptr_multi = 1, 0
        for i, metric in enumerate(metrics):
            if metric in self.metrics:
                if len(methods) == 0:
                    methods_ = np.array(self.df.loc[metric].index.unique(0)).astype(str)
                else:
                    methods_ = methods
                idx = ptr_scalar*ncols_scalar[ptr_scalar] + counter_scalar
                ax = fig.add_subplot(nrows, ncols_scalar[ptr_scalar], idx)
                self._plot_metrics_ax(metric,
                                      datasets,
                                      methods_,
                                      show_legend=(metric == legend_metric),
                                      ax=ax,
                                      figsize=(figsize[0], figsize[1]),
                                      title_fontsize=title_fontsize,
                                      legend_fontsize=legend_fontsize,
                                      tick_fontsize=tick_fontsize)
                if (metric == legend_metric):
                    handles, labels = ax.get_legend_handles_labels()
                    ax.get_legend().remove()
                counter_scalar += 1
                if counter_scalar > ncols_scalar[ptr_scalar]:
                    ptr_scalar += 1
                    counter_scalar = 1
            elif metric in self.multi_metrics:
                if len(methods) == 0:
                    methods_ = np.array(self.df_multi.loc[metric].index.unique(0)).astype(str)
                else:
                    methods_ = methods
                for j, dataset in enumerate(datasets):
                    idx = (n_scalar + ptr_multi)*ncols_multi[ptr_multi] + counter_multi
                    ax = fig.add_subplot(nrows, ncols_multi[ptr_multi], idx)
                    self._plot_velocity_metrics_ax(metric,
                                                   dataset,
                                                   methods_,
                                                   show_legend=(metric == legend_metric),
                                                   ax=ax,
                                                   figsize=(figsize[0], figsize[1]),
                                                   markersize=markersize,
                                                   linewidth=linewidth,
                                                   title_fontsize=title_fontsize,
                                                   legend_fontsize=legend_fontsize,
                                                   ylabel_fontsize=ylabel_fontsize,
                                                   labelpad=labelpad,
                                                   tick_fontsize=tick_fontsize)
                    if j > 0:
                        ax.set_ylabel('')
                    # _fig = ax.get_figure()
                    # print(_fig.get_size_inches())
                    if (metric == legend_metric):
                        handles, labels = ax.get_legend_handles_labels()
                        ax.get_legend().remove()
                    counter_multi += 1
                    if counter_multi > ncols_multi[ptr_multi]:
                        ptr_multi += 1
                        counter_multi = 1
        loc = 'center' if len(bbox_to_anchor) == 4 else 1
        if len(methods) == 0:
            if legend_metric in self.multi_metrics:
                methods = np.array(self.df_multi.loc[legend_metric].index.unique(0)).astype(str)
            else:
                methods = np.array(self.df.loc[legend_metric].index.unique(0)).astype(str)
        ncols_legend = np.max(self._decompose_num(len(methods))) if ncols_legend is None else ncols_legend
        fig.legend(handles, labels, bbox_to_anchor=bbox_to_anchor, loc=loc, ncol=ncols_legend, fontsize=legend_fontsize)
        fig.tight_layout()
        fig.subplots_adjust(hspace=hspace, wspace=wspace)
        if figure_path is not None:
            fig.savefig(f'{figure_path}/{figure_name}.{save_format}', bbox_inches='tight', dpi=dpi)
        return

    def save(self, file_name=None):
        """Save data frames to .csv files.

        Args:
            file_name (str, optional):
                Name of the csv file for saving. Does not need the path
                as the path is specified when an object is created.
                If set to None, will pick 'perf' as the default name.
                Defaults to None.
        """
        if file_name is None:
            file_name = "perf"
        self.df.to_csv(f"{self.save_path}/{file_name}.csv")
        self.df_type.to_csv(f"{self.save_path}/{file_name}_type.csv")
        self.df_multi.to_csv(f"{self.save_path}/{file_name}_multi.csv")
        self.df_multi_type.to_csv(f"{self.save_path}/{file_name}_multi_type.csv")