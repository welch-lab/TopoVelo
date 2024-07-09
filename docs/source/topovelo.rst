API for Developers
==================
This page contains more detailed functionality of each module. Some are merely utility functions and still in development.

Preprocessing
-------------

.. automodule:: topovelo.preprocessing
   :members:
   :undoc-members: 
   :show-inheritance:

VAE Model
---------

.. autoclass:: topovelo.model.VAE
   :members: __init__,
             train,
             save_model,
             save_anndata,
             forward,
             eval_model,
             pred_xy,
             pred_all,
             reload_training,
             resume_train_stage_3,
             test,
             test_spatial,
             get_enc_att,
             get_dec_att,
             get_enc_att_all_pairs,
             get_dec_att_all_pairs,
             xy_velocity,
             vae_risk_gaussian,
             vae_risk_poisson,
             vae_risk_nb,
             vae_spatial_loss,
             update_x0,
             set_mode,
             split_train_validation_test,
             encode_batch,
             encode_slice,
   :undoc-members: 
   :show-inheritance:

Evaluation
----------

.. automodule:: topovelo.analysis
   :members:
   :undoc-members:
   :show-inheritance:

Plotting Functions
------------------

.. automodule:: topovelo.plotting
   :members: get_colors,
             set_figure_size,
             set_fontsize,
             compute_figsize,
             plot_spatial_graph,
             plot_heatmap,
             plot_heat_density,
             plot_trajectory_3d,
             plot_trajectory_4d,
             plot_phase_grid,
             plot_sig_grid,
             plot_time_grid,
             plot_rate_grid,
             plot_rate_hist,
             plot_phase_vel,
             plot_velocity,
             plot_spatial_extrapolation,
             plot_legend,
             plot_3d_heatmap,
             plot_time,
             plot_time_var,
             plot_state_var,

   :undoc-members:
   :show-inheritance:
