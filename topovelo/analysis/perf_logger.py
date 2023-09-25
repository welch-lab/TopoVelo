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
                        "CBDir",
                        "CBDir (Embed)",
                        "Time Score",
                        "In-Cluster Coherence",
                        "Velocity Consistency",
                        "Spatial Consistency",
                        "Time Correlation"]
        self.multi_metrics = ["K-CBDir",
                              "K-CBDir (Embed)",
                              "Mann-Whitney Test",
                              "Mann-Whitney Test (Embed)",
                              "Mann-Whitney Test Stats",
                              "Mann-Whitney Test Stats (Embed)"]
        self.metrics_type = ["CBDir",
                             "CBDir (Embed)",
                             "Time Score"]
        if checkpoints is None:
            self._create_empty_df()
        else:
            self.df = pd.read_csv(checkpoints[0], header=[0], index_col=[0, 1])
            self.df_type = pd.read_csv(checkpoints[1], header=[0, 1], index_col=[0, 1])

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
        """Insert the performance evaluation results from velovae.post_analysis

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
        # Collapse the dataframe to 1D series with multi-index
        res_1d = pd.Series(res.values.flatten(), index=pd.MultiIndex.from_product([res.index, res.columns]))
        for x in res_1d.index:
            self.df.loc[x, data_name] = res_1d.loc[x]

        # Reshape the data in res_type to match the multi-row-index in self.df_type
        methods = np.unique(res_type.columns.unique(0))
        res_reshape = pd.DataFrame(res_type.values.reshape(res_type.shape[0] * len(methods), -1),
                                   index=pd.MultiIndex.from_product([res_type.index, methods]),
                                   columns=pd.MultiIndex.from_product([[data_name], res_type.columns.unique(1)]))
        self.df_type = pd.concat([self.df_type, res_reshape], axis=1)

        # Multi-dimensional metrics
        res_reshape = pd.DataFrame(multi_res.values.reshape(multi_res.shape[0] * len(methods), -1),
                                   index=pd.MultiIndex.from_product([multi_res.index, methods]),
                                   columns=pd.MultiIndex.from_product([[data_name],
                                                                       multi_res.columns.unique(1)]))
        self.df_multi = pd.concat([self.df_multi, res_reshape], axis=1)

        # Multi-dimensional metrics for each transition pair
        res_reshape = pd.DataFrame(multi_res_type.values.reshape(multi_res_type.shape[0] * len(methods), -1),
                                   index=pd.MultiIndex.from_product([multi_res_type.index, methods]),
                                   columns=pd.MultiIndex.from_product([[data_name],
                                                                       multi_res_type.columns.unique(1),
                                                                       multi_res_type.columns.unique(2)]))
        self.df_multi_type = pd.concat([self.df_multi_type, res_reshape], axis=1)

        # update number of models
        self.n_model = len(self.df.index.unique(1))
        self.df.sort_index(inplace=True)
        self.df_type.sort_index(inplace=True)
        self.df_multi.sort_index(inplace=True)
        self.df_multi_type.sort_index(inplace=True)
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
            ax.grid()
            fig = ax.get_figure()
            fig.tight_layout()
            if figure_path is not None:
                fig_name = re.sub(r'\W+', ' ', metric.lower())
                fig_name = '_'.join(fig_name.rstrip().split())
                fig.savefig(f'{figure_path}/{metric}_summary.png', dpi=dpi, bbox_inches='tight')
        
        return

    def plot_transition_pairs(self, metrics=[], figure_path=None, bbox_to_anchor=(1.25, 1.0)):
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
        datasets = np.unique(self.df_type.columns.unique(0))
        for metric in metrics:
            if metric not in self.metrics_type:
                continue
            fig_name = re.sub(r'\W+', ' ', metric.lower())
            fig_name = '_'.join(fig_name.rstrip().split())
            for dataset in datasets:
                if np.all(np.isnan(self.df_type.loc[metric, dataset].values)):
                    continue
                colors = get_colors(self.df_type.loc[metric, dataset].shape[0])
                ax = self.df_type.loc[metric, dataset].T.plot.bar(color=colors, figsize=(12, 6), fontsize=14)
                ax.set_title(metric, fontsize=20)
                if isinstance(bbox_to_anchor, tuple):
                    ax.legend(fontsize=16, loc=1, bbox_to_anchor=bbox_to_anchor)
                transition_pairs = self.df_type[dataset].columns.unique(0)
                ax.set_xticklabels(transition_pairs, rotation=0)
                ax.set_xlabel("")
                ax.grid()
                fig = ax.get_figure()
                fig.tight_layout()
                if figure_path is not None:
                    fig.savefig(f'{figure_path}/perf_{fig_name}_{dataset}.png', bbox_inches='tight')
        return

    def plot(self, metrics=[], figure_path=None, bbox_to_anchor=(1.25, 1.0)):
        """Generate bar plots showing all performance metrics.
        Each plot has different datasets as x-axis labels and different bars represent methods.

        Args:
            figure_path (str, optional):
                Path to the folder for saving figures.
                If set to None, figures will not be saved.
                Defaults to None.
            bbox_to_anchor (tuple, optional):
                Location of the legend. Defaults to (1.25, 1.0).
        """
        datasets = self.df.columns.unique(0)
        for metric in metrics:
            if metric not in self.metrics:
                continue
            colors = get_colors(self.df.loc[metric, :].shape[0])
            fig_name = re.sub(r'\W+', ' ', metric.lower())
            fig_name = '_'.join(fig_name.rstrip().split())
            if np.all(np.isnan(self.df.loc[metric, :].values)):
                continue
            ax = self.df.loc[metric, :].T.plot.bar(color=colors, figsize=(12, 6), fontsize=14)
            ax.set_xlabel("")
            ax.set_xticklabels(datasets, rotation=0)
            ax.set_title(metric, fontsize=20)
            ax.grid()
            if isinstance(bbox_to_anchor, tuple):
                ax.legend(fontsize=16, loc=1, bbox_to_anchor=bbox_to_anchor)
            fig = ax.get_figure()
            fig.tight_layout()
            if figure_path is not None:
                fig.savefig(f'{figure_path}/perf_{fig_name}.png', bbox_inches='tight')
        return

    def plot_velocity_metrics(self,
                              dataset=None,
                              methods=None,
                              figure_path=None,
                              bbox_to_anchor=(0, 1, 1, 0.1),
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
        if methods == None:
            methods = list(self.df.index.unique(0))
        datasets = list(self.df.columns.unique(0))
        models = list(self.df_multi.index.unique(1))
        steps = self.df_multi.columns.unique(1)
        colors = get_colors(len(models))
        for metric in self.multi_metrics:
            metric_name = re.sub(r'\W+', ' ', metric.lower())
            metric_name = '_'.join(metric_name.rstrip().split())
            for dataset in datasets:
                data_name = '-'.join(dataset.rstrip().split())
                fig_name = metric_name+"_"+data_name
                for i, model in enumerate(models):
                    ax = self.df_multi.loc[(metric, model), dataset].plot(marker=MARKERS[i],
                                                                          markersize=10,
                                                                          figsize=(5, 6),
                                                                          color=colors[i],
                                                                          label=model)
                    ax.set_xlabel("")
                    ax.set_xticks(range(len(steps)), steps, rotation=0)
                    ax.legend(fontsize=13.5, ncol=4, loc='center', bbox_to_anchor=bbox_to_anchor)
                ax.grid()
                ax.set_ylabel(metric, fontsize=15)
                ax.tick_params(axis='both', which='major', labelsize=10)
                fig = ax.get_figure()
                fig.tight_layout()
                if figure_path is not None:
                    fig.savefig(f'{figure_path}/{fig_name}_{dataset}.png', bbox_inches='tight', dpi=dpi)
                plt.show(fig)
                plt.close()
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