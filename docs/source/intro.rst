Introduction
============

Welcome to the documentation of the TopoVelo package! 
This package provides computational tools for inferring RNA velocity and spatial cellular dynamics from spatial transcriptomic data.
``TopoVelo`` applies a graph variational autoencoder to learn spatially-coupled RNA velocity from unspliced and spliced mRNA count matrices.
It also computes the spatial migration of cells based on the learned RNA velocity.
The package includes functions for data preprocessing, model training, evaluation and visualization of the results.
The package is implemented in Python and is available on GitHub at `https://github.com/welch-lab/TopoVelo`_.
