# TopoVelo

TopoVelo stands for **Topo**logical **velo**city inference. It is a tool for jointly modeling temporal gene expression and spatial cellular dynamics from spatial transcriptomic data.
The method applies a graph variational autoencoder to recover spatially-coupled RNA velocity and reveal the spatial migration of cells during tissue growth.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)

## Installation

To install the TopoVelo package, you can use pip:
```
git clone https://github.com/welch-lab/TopoVelo.git
cd TopoVelo
pip install .
```

## Usage

The package provides function modules for training and evaluating a TopoVelo model and plotting results. The default pipeline with minimal hyperparameter tuning is shown as follows:
```
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
```
Examples of advanced usage, model evaluation and plotting can be found in the [tutorial](./notebooks/tutorial/).

## License

GNU General Public License v3.0

## Contact

The package is currently maintained by Yichen Gu. Please contact the maintainer via gyichen@umich.edu.
