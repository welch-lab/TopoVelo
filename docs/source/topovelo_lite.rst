TopoVelo API
============
This page contains the main API documentation for the TopoVelo package. The functions on this page are expected to fulfill most users' needs.
For a more detailed documentation, please refer to the full API documentation.

Preprocessing
-------------

.. automodule:: topovelo.preprocessing
   :members: preprocess, build_spatial_graph
   :undoc-members: 
   :show-inheritance:

VAE Model
---------

.. autoclass:: topovelo.model.VAE
   :members: __init__, train, save_model, save_anndata
   :undoc-members: 
   :show-inheritance:

Evaluation
----------

.. automodule:: topovelo.analysis
   :members: post_analysis
   :undoc-members:
   :show-inheritance:

Plotting Functions
------------------

.. automodule:: topovelo.plotting
   :members: get_colors, compute_figsize, plot_spatial_graph, plot_heatmap, plot_heat_density, plot_trajectory_4d
   :undoc-members:
   :show-inheritance:
