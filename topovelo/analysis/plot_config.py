DEFAULT = {
    "time": {
        "downsample": 1,
        "max_quantile": 0.99,
        "color_map": "viridis",
        "width": 6,
        "height": 4,
        "marker": "o",
        "markersize": 50,
        "linewidths": 0.5,
        "grid_size": None,
        "real_aspect_ratio": True,
        "title_fontsize": None,
        "show_colorbar": True,
        "colorbar_fontsize": 20,
        "colorbar_labelpad": 20,
        "colorbar_pos": [1.04, 0.2, 0.05, 0.6],
        "path": None,
        "figname": "time",
        "save_format": "png"
    },
    "stream": {
        "width": 6,
        "height": 4,
        "density": 2,
        "markersize": 50,
        "alpha": 0.5,
        "linewidth": 1.0,
        "arrow_size": 1.0,
        "arrow_color": "black",
        "legend_loc": "right margin",
        "palette": None,
        "cutoff_perc": 0.0,
        "perc": [2, 98],
        "legend_fontsize": None
    },
    "phase": {
        "width": 6,
        "height": 6,
        "alpha": 0.8,
        "downsample": 1,
        "obs_marker": "o",
        "pred_marker": "x",
        "markersize": 10,
        "linewidths": 0.5,
        "title_fontsize": 30,
        "show_legend": True,
        "legend_fontsize": None,
        "legend_loc": "upper right",
        "bbox_to_anchor": None,
        "label_fontsize": None,
        "palette": None,
        "hspace": 0.3,
        "wspace": 0.12,
        "markerscale": 5.0,
        "path": None,
        "figname": "phase",
        "save_format": "png"
    },
    "gene": {
        "width": 6,
        "height": 3,
        "alpha": 1.0,
        "downsample": 1,
        "loess_downsample": None,
        "sparsity_correction": False,
        "plot_loess": False,
        "frac": 0.5,
        "marker": "o",
        "markersize": 5,
        "linewidths": 0.5,
        "palette": None,
        "show_legend": True,
        "legend_fontsize": None,
        "title_fontsize": 30,
        "headwidth": 5.0,
        "headlength": 8.0,
        "label_fontsize": 30,
        "y_label_pos_x": -0.03,
        "y_label_pos_y": 0.5,
        "show_xticks": False,
        "tick_fontsize": 15,
        "hspace": 0.3,
        "wspace": 0.12,
        "markerscale": 5.0,
        "bbox_to_anchor": None,
        "legend_loc": "upper right",
        "path": None,
        "figname": "gene",
        "save_format": "png"
    }
}


class PlotConfig():
    """Class to store the configuration of a plot.
    """
    def __init__(self, plot_type):
        """Initialise the PlotConfig object.

        Args:
            plot_type (str): The type of plot to configure.
        """
        self.config = DEFAULT[plot_type].copy()
    
    def set(self, key, value):
        """Set a configuration value.

        Args:
            key (str): The key to set the value of.
            value: The value to set.
        """
        self.config[key] = value
    
    def set_multiple(self, config):
        """Set multiple configuration values at once.

        Args:
            config (dict): A dictionary of configuration values.
        """
        for key, value in config.items():
            self.config[key] = value
    
    def get(self, key):
        """Get the value of a key in the configuration.

        Args:
            key (str): The key to get the value of.

        Returns:
            The value of the key.
        """
        return self.config[key]
    
    def get_all(self):
        """Get all the configuration values.

        Returns:
            list: The configuration values.
        """
        return list(self.config.values())