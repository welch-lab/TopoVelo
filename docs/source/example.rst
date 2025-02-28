Examples
=============

Installation
*************
The package is available on PyPI. Users can install it using pip:
::

    pip install topovelo

Alternatively, you can download the package and manually install it:
::

    git clone https://github.com/welch-lab/VeloVAE.git
    cd topovelo
    pip install .

If you do not want to install it into your python environment, you can just download the package and import it locally by adding the path of the package to the system path:
::

    import sys
    sys.path.append(< path to the package >)
    import topovelo

Usage
*************
To train a VAE model, we need an input dataset in the form of an `AnnData <https://anndata.readthedocs.io/en/latest/index.html>`_
object. Any raw count matrix needs to be preprocessed. This package has a basic preprocessing function following the scVelo pipeline,
but we suggest users start from scVelo for more comprehensive preprocessing functionality.

**Note**: Sometimes the results are susceptible to gene selection. In this case, we suggest users consider a different strategy for selecting genes.
The following code block shows the major steps for using the tool:
::

    import scanpy as sc
    import scvelo as scv
    import topovelo as tpv
    import torch

    # Load .h5ad to AnnData
    adata = sc.read(< path to .h5ad >)

    # Preprocessing
    spatial_key = 'X_spatial'
    tpv.preprocess(adata, n_gene=200, spatial_key=spatial_key)
    tpv.build_spatial_graph(adata, spatial_key)

    # Create a VAE object 
    device = torch.device('cpu') if torch.cuda.is_available() else torch.device('cpu')
    vae = tpv.VAE(adata, tmax=20, dim_z=5, device=device)

    # Training
    vae.train(adata, adata.obsp['spatial_graph'], spatial_key)

    # Save results
    vae.save_model(< path to model parameters >, 'encoder', 'decoder')
    vae.save_anndata(adata, 'gat', < path to output >, file_name="adata_out.h5ad")

There are some hyper-parameters users can tune before training.
Notebook with illustrative examples can be found `here <https://github.com/welch-lab/TopoVelo/tree/main/notebooks>`_.
