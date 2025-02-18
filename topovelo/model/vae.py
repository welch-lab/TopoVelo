from typing import Dict, List, Literal, Tuple, Optional, Union
import logging
import os
from anndata import AnnData
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions.negative_binomial import NegativeBinomial
from torch.distributions.poisson import Poisson

from torch_geometric import seed_everything
from torch_geometric.nn import GCNConv, GATConv

import time
from ..plotting import plot_sig, plot_time, plot_cluster
from ..plotting import plot_train_loss, plot_test_loss
from ..plotting import plot_spatial_extrapolation

from .model_util import hist_equal, init_params, init_params_parallel, get_ts_global, reinit_params
from .model_util import convert_time, get_gene_index
from .model_util import pred_su, knnx0_index, knnx0_index_parallel, get_x0, knnx0_index_batch
from .model_util import elbo_collapsed_categorical
from .model_util import assign_gene_mode, find_dirichlet_param, assign_gene_mode_tprior
from .model_util import dge2array
from .model_util import encode_type
from .training_data import SCGraphData, Index
from .vanilla_vae import VanillaVAE, kl_gaussian
from .velocity import rna_velocity_vae
P_MAX = 1e4
GRAD_MAX = 1e5
logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    """Encoder class for the graph VAE model
    """
    def __init__(self,
                 Cin,
                 dim_z,
                 dim_cond=0,
                 dim_edge=None,
                 n_hidden=500,
                 attention=True,
                 n_head=5,
                 xavier_gain=0.05,
                 checkpoint=None):
        super(Encoder, self).__init__()
        self.dim_z = dim_z
        self.dim_cond = dim_cond
        self.n_hidden = n_hidden
        self.n_head = n_head
        self.xavier_gain = xavier_gain
        if attention:
            self.conv1 = GATConv(Cin, n_hidden, n_head, edge_dim=dim_edge)
            self.fc_mu_t = nn.Linear(n_head*n_hidden+dim_cond, 1).float()
            self.spt1 = nn.Softplus().float()
            self.fc_std_t = nn.Linear(n_head*n_hidden+dim_cond, 1).float()
            self.spt2 = nn.Softplus().float()

            self.fc_mu_z = nn.Linear(n_head*n_hidden+dim_cond, dim_z).float()
            self.fc_std_z = nn.Linear(n_head*n_hidden+dim_cond, dim_z).float()
            self.spt3 = nn.Softplus().float()
        else:
            self.conv1 = GCNConv(Cin, n_hidden)
            self.fc_mu_t = nn.Linear(n_hidden+dim_cond, 1).float()
            self.spt1 = nn.Softplus().float()
            self.fc_std_t = nn.Linear(n_hidden+dim_cond, 1).float()
            self.spt2 = nn.Softplus().float()

            self.fc_mu_z = nn.Linear(n_hidden+dim_cond, dim_z).float()
            self.fc_std_z = nn.Linear(n_hidden+dim_cond, dim_z).float()
            self.spt3 = nn.Softplus().float()
        if checkpoint is not None:
            self.load_state_dict(torch.load(checkpoint))
        else:
            self.init_weights()

    def init_weights(self):
        for m in [self.conv1]:
            if isinstance(m, GCNConv):
                nn.init.xavier_uniform_(m.lin.weight, self.xavier_gain)
                nn.init.constant_(m.bias, 0)
            else:
                try:
                    nn.init.xavier_uniform_(m.lin.weight, self.xavier_gain)
                except AttributeError:
                    # deprecated in recent torch_geometric
                    nn.init.xavier_uniform_(m.lin_src.weight, self.xavier_gain)
                    nn.init.xavier_uniform_(m.lin_dst.weight, self.xavier_gain)
                nn.init.xavier_uniform_(m.att_src, self.xavier_gain)
                nn.init.xavier_uniform_(m.att_dst, self.xavier_gain)
                nn.init.constant_(m.bias, 0)
        for m in [self.fc_mu_t,
                  self.fc_std_t,
                  self.fc_mu_z,
                  self.fc_std_z]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, data_in, edge_index, edge_weight=None, condition=None):
        if isinstance(self.conv1, GCNConv):
            h = F.relu(self.conv1(data_in, edge_index, edge_weight))
        else:
            h = F.relu(self.conv1(data_in, edge_index))
        if condition is not None:
            h = torch.cat((h, condition), 1)
        mu_tx, std_tx = self.spt1(self.fc_mu_t(h)), self.spt2(self.fc_std_t(h))
        mu_zx, std_zx = self.fc_mu_z(h), self.spt3(self.fc_std_z(h))
        return mu_tx, std_tx, mu_zx, std_zx


class MLPDecoder(nn.Module):
    """MLP decoder for learning spatial rates.
    """
    def __init__(self,
                 Cin,
                 dim_out,
                 dim_cond=0,
                 hidden_size=(250, 500),
                 xavier_gain=1.0,
                 enable_sigmoid=True,
                 checkpoint=None):
        super(MLPDecoder, self).__init__()
        self.xavier_gain = xavier_gain
        self.enable_sigmoid = enable_sigmoid
        self.n_hidden = len(hidden_size)

        self.bn_in = nn.BatchNorm1d(num_features=Cin)

        self.fcs = nn.ModuleList([nn.Linear(Cin+dim_cond, hidden_size[0])])
        for i in range(1, self.n_hidden):
            self.fcs.append(nn.Linear(hidden_size[i-1], hidden_size[i]))
        self.dpts = nn.ModuleList([nn.Dropout(p=0.2) for i in range(self.n_hidden)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_features=h) for h in hidden_size])
        self.fc_out = nn.Linear(hidden_size[-1], dim_out)
        self.acts = nn.ModuleList([nn.LeakyReLU() for i in range(self.n_hidden)])
        self.sigm = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        for m in self.fcs:
            nn.init.xavier_uniform_(m.weight, self.xavier_gain)
            nn.init.constant_(m.bias, 0.0)
        for m in self.bns:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        nn.init.xavier_normal_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, 0.0)

    def forward(self, z_in):
        h = self.dpts[0](self.acts[0](self.bns[0](self.fcs[0](self.bn_in(z_in)))))
        for i in range(1, self.n_hidden):
            h = self.dpts[i](self.acts[i](self.bns[i](self.fcs[i](h))))
        if not self.enable_sigmoid:
            return self.fc_out(h)
        return self.sigm(self.fc_out(h))


class GraphDecoder(nn.Module):
    """Graph decoder for learning spatial rates.
    """
    def __init__(self,
                 Cin,
                 dim_out,
                 dim_cond=0,
                 dim_edge=None,
                 n_hidden=500,
                 attention=True,
                 n_head=5,
                 xavier_gain=1.0,
                 enable_sigmoid=True,
                 checkpoint=None):
        super(GraphDecoder, self).__init__()
        self.dim_out = dim_out
        self.dim_cond = dim_cond
        self.n_head = 5
        self.xavier_gain = xavier_gain
        self.enable_sigmoid = enable_sigmoid
        if attention:
            self.conv1 = GATConv(Cin, n_hidden, n_head, edge_dim=dim_edge)
            self.fc_out = nn.Linear(n_head*n_hidden, dim_out).float()
            self.sigm = nn.Sigmoid().float()
        else:
            self.conv1 = GCNConv(Cin, n_hidden)
            self.fc_out = nn.Linear(n_hidden, dim_out).float()
            self.sigm = nn.Sigmoid().float()
        self.act = nn.LeakyReLU()
        self.init_weights()

    def init_weights(self):
        for m in [self.conv1]:
            if isinstance(m, GCNConv):
                nn.init.xavier_uniform_(m.lin.weight, self.xavier_gain)
                nn.init.constant_(m.bias, 0)
            else:
                try:
                    nn.init.xavier_uniform_(m.lin.weight, self.xavier_gain)
                except AttributeError:
                    # old torch_geometric version 
                    nn.init.xavier_uniform_(m.lin_src.weight, self.xavier_gain)
                    nn.init.xavier_uniform_(m.lin_dst.weight, self.xavier_gain)
                nn.init.xavier_uniform_(m.att_src, self.xavier_gain)
                nn.init.xavier_uniform_(m.att_dst, self.xavier_gain)
                nn.init.constant_(m.bias, 0)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, 0)

    def forward(self, data_in, edge_index, edge_weight=None, condition=None):
        if isinstance(self.conv1, GCNConv):
            h = self.act(self.conv1(data_in, edge_index, edge_weight))
        else:
            h = self.act(self.conv1(data_in, edge_index))
        if condition is not None:
            h = torch.cat((h, condition), 1)
        if not self.enable_sigmoid:
            return self.fc_out(h)
        return self.sigm(self.fc_out(h))


class MultiLayerGraphDecoder(nn.Module):
    """Graph decoder for learning spatial rates.
    """
    def __init__(self,
                 Cin,
                 dim_out,
                 dim_cond=0,
                 dim_edge=None,
                 hidden_size=(500, 250),
                 attention=True,
                 n_head=5,
                 xavier_gain=1.0,
                 enable_sigmoid=True,
                 checkpoint=None):
        super(MultiLayerGraphDecoder, self).__init__()
        self.dim_out = dim_out
        self.dim_cond = dim_cond
        self.n_head = 5
        self.xavier_gain = xavier_gain
        self.enable_sigmoid = enable_sigmoid
        self.n_layers = len(hidden_size)

        # self.bn_layers = nn.ModuleList([BatchNorm(Cin)])
        # for i in range(self.n_layers):
        #    self.bn_layers.append(BatchNorm(hidden_size[i]))
        if attention:
            self.conv_layers = nn.ModuleList([GATConv(Cin, hidden_size[0], n_head, edge_dim=dim_edge)])
            for i in range(1, self.n_layers):
                self.conv_layers.append(GATConv(hidden_size[i-1]*n_head, hidden_size[i], n_head))
            self.fc_out = nn.Linear(n_head*hidden_size[-1], dim_out).float()
            self.sigm = nn.Sigmoid().float()
        else:
            self.conv_layers = nn.ModuleList([GCNConv(Cin, hidden_size[0])])
            for i in range(1, self.n_layers):
                self.conv_layers.append(GCNConv(hidden_size[i-1], hidden_size[i]))
            self.fc_out = nn.Linear(hidden_size[-1], dim_out).float()
            self.sigm = nn.Sigmoid().float()
        self.acts = nn.ModuleList([nn.LeakyReLU() for i in range(len(hidden_size))])
        self.init_weights()

    def init_weights(self):
        for m in self.conv_layers:
            if isinstance(m, GCNConv):
                nn.init.xavier_uniform_(m.lin.weight, self.xavier_gain)
                nn.init.constant_(m.bias, 0)
            else:
                try:
                    nn.init.xavier_uniform_(m.lin.weight, self.xavier_gain)
                except AttributeError:
                    nn.init.xavier_uniform_(m.lin_src.weight, self.xavier_gain)
                    nn.init.xavier_uniform_(m.lin_dst.weight, self.xavier_gain)
                nn.init.xavier_uniform_(m.att_src, self.xavier_gain)
                nn.init.xavier_uniform_(m.att_dst, self.xavier_gain)
                nn.init.constant_(m.bias, 0)
        # for bn in self.bn_layers:
        #     bn.reset_parameters()
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, 0)

    def forward(self, data_in, edge_index, edge_weight=None, condition=None):
        #h = self.bn_layers[0](data_in)
        if isinstance(self.conv_layers[0], GCNConv):
            h = self.acts[0](self.conv_layers[0](data_in, edge_index, edge_weight))
            for i in range(1, self.n_layers):
                h = self.acts[i](self.conv_layers[i](h, edge_index, edge_weight))
        else:
            h = self.acts[0](self.conv_layers[0](data_in, edge_index))
            for i in range(1, self.n_layers):
                h = self.acts[i](self.conv_layers[i](h, edge_index))
        if condition is not None:
            h = torch.cat((h, condition), 1)
        if not self.enable_sigmoid:
            return self.fc_out(h)
        return self.sigm(self.fc_out(h))


class Decoder(nn.Module):
    """Decoder class for the TopoVelo model"""
    def __init__(self,
                 adata,
                 tmax,
                 train_idx,
                 dim_z,
                 full_vb=False,
                 discrete=False,
                 graph_decoder=False,
                 attention=False,
                 n_head=5,
                 dim_cond=0,
                 dim_coord=2,
                 dim_edge=None,
                 batch_idx=None,
                 ref_batch=None,
                 N1=250,
                 N2=500,
                 spatial_hidden_size=(128, 64),
                 p=98,
                 min_sigma_u=0.1,
                 min_sigma_s=0.1,
                 init_ton_zero=False,
                 filter_gene=False,
                 xavier_gain=0.05,
                 device=torch.device('cpu'),
                 init_method="steady",
                 init_key=None,
                 checkpoint=None,
                 **kwargs):
        super(Decoder, self).__init__()
        self.n_gene = adata.n_vars
        self.tmax = tmax
        self.train_idx = train_idx
        self.dim_z = dim_z
        self.is_full_vb = full_vb
        self.is_discrete = discrete
        self.graph_decoder = graph_decoder
        self.attention = attention
        self.dim_cond = dim_cond
        self.cvae = True if dim_cond > 1 else False
        self.batch = batch_idx
        self.ref_batch = ref_batch
        self.min_sigma_u = min_sigma_u
        self.min_sigma_s = min_sigma_s
        self.init_ton_zero = init_ton_zero
        self.filter_gene = filter_gene
        self.device = device
        self.init_method = init_method
        self.init_key = init_key
        self.checkpoint = checkpoint
        self.construct_nn(adata, dim_z, dim_cond, dim_coord, dim_edge, N1, N2, spatial_hidden_size, p, n_head, xavier_gain, **kwargs)

    def construct_nn(self, adata, dim_z, dim_cond, dim_coord, dim_edge, N1, N2, spatial_hidden_size, p, n_head, xavier_gain, **kwargs):
        """Construct neural networks for the decoder."""
        self.set_shape(self.n_gene, dim_cond)
        if self.graph_decoder:
            self.net_rho = GraphDecoder(dim_z+dim_cond,
                                        self.n_gene,
                                        n_hidden=N2,
                                        dim_edge=dim_edge,
                                        attention=self.attention,
                                        n_head=n_head,
                                        xavier_gain=xavier_gain).to(self.device)
            self.net_rho2 = GraphDecoder(dim_z+dim_cond,
                                         self.n_gene,
                                         n_hidden=N2,
                                         dim_edge=dim_edge,
                                         attention=self.attention,
                                         n_head=n_head,
                                         xavier_gain=xavier_gain).to(self.device)
        else:
            self.net_rho = MLPDecoder(dim_z, self.n_gene, dim_cond, hidden_size=(N1, N2)).to(self.device)
            self.net_rho2 = MLPDecoder(dim_z, self.n_gene, dim_cond, hidden_size=(N1, N2)).to(self.device)
        
        self.net_coord = MultiLayerGraphDecoder(dim_z+dim_cond+2+dim_coord,
                                                dim_coord,
                                                hidden_size=spatial_hidden_size,
                                                attention=self.attention,
                                                n_head=n_head,
                                                xavier_gain=xavier_gain,
                                                enable_sigmoid=False).to(self.device)

        if self.checkpoint is not None:
            self.alpha = nn.Parameter(torch.empty(self.params_shape))
            self.beta = nn.Parameter(torch.empty(self.params_shape))
            self.gamma = nn.Parameter(torch.empty(self.params_shape))
            self.register_buffer('sigma_u', torch.empty(self.n_gene))
            self.register_buffer('sigma_s', torch.empty(self.n_gene))
            self.register_buffer('zero_vec', torch.empty(self.n_gene))

            self.ton = nn.Parameter(torch.empty(self.n_gene))
            self.u0 = nn.Parameter(torch.empty(self.n_gene))
            self.s0 = nn.Parameter(torch.empty(self.n_gene))
            self.logit_pw = nn.Parameter(torch.empty(self.n_gene, 2))

            if self.cvae:
                self.scaling_u = nn.Parameter(torch.empty((self.dim_cond, self.n_gene)))
                self.scaling_s = nn.Parameter(torch.empty((self.dim_cond, self.n_gene)))
                self.register_buffer('one_mat', torch.empty((self.dim_cond, self.n_gene)))
                self.register_buffer('zero_mat', torch.empty((self.dim_cond, self.n_gene)))
            else:
                self.scaling_u = nn.Parameter(torch.empty(self.n_gene))
                self.scaling_s = nn.Parameter(torch.empty(self.n_gene))
            self.load_state_dict(torch.load(self.checkpoint))
        else:
            self.init_ode(adata, p, **kwargs)
        
    def set_shape(self, G, dim_cond):
        if self.is_full_vb:
            if self.cvae:
                self.params_shape = (dim_cond, 2, G)
            else:
                self.params_shape = (2, G)
        else:
            if self.cvae:
                self.params_shape = (dim_cond, G)
            else:
                self.params_shape = G

    def to_param(self, x):
        if self.is_full_vb:
            if self.cvae:
                return nn.Parameter(torch.tensor(np.tile(np.stack([np.log(x),
                                                                   np.log(0.05)*np.ones(self.params_shape[2])]),
                                                         (self.params_shape[0], 1, 1)), device=self.device))
            else:
                return nn.Parameter(torch.tensor(np.stack([np.log(x),
                                                           np.log(0.05)*np.ones(self.params_shape[1])]), device=self.device))
        else:
            if self.cvae:
                return nn.Parameter(torch.tensor(np.tile(np.log(x), (self.params_shape[0], 1)), device=self.device))
            else:
                return nn.Parameter(torch.tensor(np.log(x), device=self.device))

    def _check_param_validity(self):
        for param, val in self.named_parameters():
            if torch.any(torch.isnan(val)):
                raise AssertionError(f'NAN detected in {param}.')
            if torch.any(torch.isinf(val)):
                raise AssertionError(f'Infinity detected in {param}.')
        return

    def init_ode(self, adata, p, **kwargs):
        """Initialize the parameters of the ODE model."""
        G = adata.n_vars
        logger.info("Initialization using the steady-state and dynamical models.")
        u = adata.layers['Mu'][self.train_idx]
        s = adata.layers['Ms'][self.train_idx]

        # Compute gene scaling for multiple batches
        if self.cvae:
            logger.info("Computing scaling_u factors for each batch class.")
            scaling_u = np.ones((self.dim_cond, G))
            scaling_s = np.ones((self.dim_cond, G))
            if self.ref_batch is None:
                self.ref_batch = 0
            ui = u[self.batch[self.train_idx] == self.ref_batch]
            si = s[self.batch[self.train_idx] == self.ref_batch]
            filt = (si > 0) * (ui > 0)
            logger.debug(f"{np.mean(np.sum(filt, 0) > 0)*100:.2f}% of the genes in the reference batch have non-zero u and s in the reference batch.")
            ui[~filt] = np.nan
            si[~filt] = np.nan
            std_u_ref, std_s_ref = np.nanstd(ui, axis=0), np.nanstd(si, axis=0)
            scaling_u[self.ref_batch] = np.clip(std_u_ref / std_s_ref, 1e-6, 1e6)
            scaling_s[self.ref_batch] = 1.0
            scaling_u_full = np.zeros((len(self.train_idx), G))
            scaling_s_full = np.zeros((len(self.train_idx), G))
            for i in range(self.dim_cond):
                if i != self.ref_batch:
                    ui = u[self.batch[self.train_idx] == i]
                    si = s[self.batch[self.train_idx] == i]
                    filt = (si > 0) * (ui > 0)

                    ui[~filt] = np.nan
                    si[~filt] = np.nan
                    std_u, std_s = np.nanstd(ui, axis=0), np.nanstd(si, axis=0)
                    scaling_u[i] = np.clip(std_u / (std_s_ref*(~np.isnan(std_s_ref)) + std_s*np.isnan(std_s_ref)),
                                           1e-6, 1e6)
                    scaling_s[i] = np.clip(std_s / (std_s_ref*(~np.isnan(std_s_ref)) + std_s*np.isnan(std_s_ref)),
                                           1e-6, 1e6)
                scaling_u_full[self.batch[self.train_idx] == i] = scaling_u[i]
                scaling_s_full[self.batch[self.train_idx] == i] = scaling_s[i]
            # Handle inf and nan
            p_nan = np.mean(np.isnan(scaling_u) | np.isnan(scaling_s), 1)*100
            p_inf = np.mean(np.isinf(scaling_u) | np.isinf(scaling_s), 1)*100
            if np.any(np.isnan(scaling_u) | np.isinf(scaling_u)):
                scaling_u[np.isnan(scaling_u) | np.isinf(scaling_u)] = 1.0
            if np.any(np.isnan(scaling_s) | np.isinf(scaling_s)):
                scaling_s[np.isnan(scaling_s) | np.isinf(scaling_s)] = 1.0
            if np.any(np.isnan(scaling_u_full) | np.isinf(scaling_u_full)):
                scaling_u_full[np.isnan(scaling_u_full) | np.isinf(scaling_u_full)] = 1.0
            if np.any(np.isnan(scaling_s_full) | np.isinf(scaling_s_full)):
                scaling_s_full[np.isnan(scaling_s_full) | np.isinf(scaling_s_full)] = 1.0
            
            (alpha, beta, gamma,
             _,
             toff,
             u0, s0,
             sigma_u, sigma_s,
             T, gene_score) = init_params(u/scaling_s_full, s/scaling_s_full, p, fit_scaling=True,
                                          min_sigma_u=self.min_sigma_u, min_sigma_s=self.min_sigma_s)
        else:
            (alpha, beta, gamma,
             scaling_u,
             toff,
             u0, s0,
             sigma_u, sigma_s,
             T, gene_score) = init_params(u, s, p, fit_scaling=True,
                                          min_sigma_u=self.min_sigma_u, min_sigma_s=self.min_sigma_u)
            # Scaling equals 1 in case of a single batch
            scaling_s = np.ones_like(scaling_u)
        
            if np.any(np.isinf(scaling_u)):
                logger.warning('scaling_u invalid inf')
                scaling_u[np.isinf(scaling_u)] = 1.0
            if np.any(np.isnan(scaling_u)):
                logger.warning('scaling_u invalid nan')
                scaling_u[np.isnan(scaling_u)] = 1.0
            scaling_u_full = scaling_u
            scaling_s_full = scaling_s
        
        # Gene dyncamical mode initialization
        if self.init_method == 'tprior':
            w = assign_gene_mode_tprior(adata, self.init_key, self.train_idx)
        else:
            dyn_mask = (T > self.tmax*0.01) & (np.abs(T-toff) > self.tmax*0.01)
            w = np.sum(((T < toff) & dyn_mask), 0) / (np.sum(dyn_mask, 0) + 1e-10)
            assign_type = kwargs['assign_type'] if 'assign_type' in kwargs else 'auto'
            thred = kwargs['ks_test_thred'] if 'ks_test_thred' in kwargs else 0.05
            n_cluster_thred = kwargs['n_cluster_thred'] if 'n_cluster_thred' in kwargs else 3
            std_prior = kwargs['std_alpha_prior'] if 'std_alpha_prior' in kwargs else 0.1
            if 'reverse_gene_mode' in kwargs:
                w = (1 - assign_gene_mode(adata, w, assign_type, thred, std_prior, n_cluster_thred)
                     if kwargs['reverse_gene_mode'] else
                     assign_gene_mode(adata, w, assign_type, thred, std_prior, n_cluster_thred))
            else:
                w = assign_gene_mode(adata, w, assign_type, thred, std_prior, n_cluster_thred)
        self.w_init = w
        logger.debug(f"Initial induction: {np.sum(w >= 0.5)}, repression: {np.sum(w < 0.5)}/{G}")
        adata.var["w_init"] = w
        logit_pw = 0.5*(np.log(w+1e-10) - np.log(1-w+1e-10))
        logit_pw = np.stack([logit_pw, -logit_pw], 1)
        self.logit_pw = nn.Parameter(torch.tensor(logit_pw, device=self.device).float())
        
        # Reinitialize parameters with a unified time 
        if self.init_method == "tprior":
            logger.info("Initialization using prior time.")
            t_prior = adata.obs[self.init_key].to_numpy()
            t_prior = t_prior[self.train_idx]
            std_t = (np.std(t_prior)+1e-3)*0.05
            std_t = std_t*(self.tmax/(t_prior.max() - t_prior.min()))
            self.t_init = np.random.uniform(t_prior-std_t, t_prior+std_t)
            self.t_init -= self.t_init.min()
            self.t_init = self.t_init
            self.t_init = self.t_init/self.t_init.max()*self.tmax
            self.toff_init = get_ts_global(self.t_init, u/scaling_u_full, s/scaling_s_full, 95)
            self.alpha_init, self.beta_init, self.gamma_init, self.ton_init = reinit_params(u/scaling_u_full,
                                                                                            s/scaling_s_full,
                                                                                            self.t_init,
                                                                                            self.toff_init)
        else:
            logger.info("Initialization using the steady-state and dynamical models.")
            T = T+np.random.rand(T.shape[0], T.shape[1]) * 1e-3
            T_eq = np.zeros(T.shape)
            n_bin = T.shape[0]//50+1
            for i in range(T.shape[1]):
                T_eq[:, i] = hist_equal(T[:, i], self.tmax, 0.9, n_bin)
            if "init_t_quant" in kwargs:
                self.t_init = np.quantile(T_eq, kwargs["init_t_quant"], 1)
            else:
                self.t_init = np.quantile(T_eq, 0.5, 1)

            # Spatially smooth the initial time
            if 'spatial_graph_key' in kwargs:
                adj_mtx = adata.obsp[kwargs['spatial_graph_key']].A[self.train_idx][:, self.train_idx]
                total_nbs = adj_mtx.sum(1).reshape(-1, 1)
                total_nbs[total_nbs <= 0] = 1
                adj_mtx = adj_mtx / total_nbs
                self.t_init = np.matmul(adj_mtx, self.t_init)

            self.toff_init = get_ts_global(self.t_init, u/scaling_u_full, s/scaling_s_full, 95)
            self.alpha_init, self.beta_init, self.gamma_init, self.ton_init = reinit_params(u/scaling_u_full,
                                                                                            s/scaling_s_full,
                                                                                            self.t_init,
                                                                                            self.toff_init)
        if self.cvae:
            del scaling_u_full
            del scaling_s_full
        self.alpha = self.to_param(self.alpha_init)
        self.beta = self.to_param(self.beta_init)
        self.gamma = self.to_param(self.gamma_init)
        self.scaling_u = nn.Parameter(torch.tensor(np.log(scaling_u), device=self.device))
        self.scaling_s = nn.Parameter(torch.tensor(np.log(scaling_s), device=self.device))

        self.u0 = nn.Parameter(torch.tensor(np.log(u0+1e-10)))
        self.s0 = nn.Parameter(torch.tensor(np.log(s0+1e-10)))

        if self.init_ton_zero:
            self.ton = nn.Parameter(torch.zeros(G, device=self.device))
        else:
            self.ton = nn.Parameter(torch.tensor(self.ton_init+1e-10, device=self.device))
        self.register_buffer('sigma_u', torch.tensor(np.log(sigma_u), device=self.device))
        self.register_buffer('sigma_s', torch.tensor(np.log(sigma_s), device=self.device))
        self.register_buffer('zero_vec', torch.zeros_like(self.u0, device=self.device))
        if self.cvae:
            self.register_buffer('one_mat', torch.ones_like(self.scaling_u, device=self.device))
            self.register_buffer('zero_mat', torch.zeros_like(self.scaling_u, device=self.device))
        self._check_param_validity()

    def get_param(self, x):
        """Retrieve the parameter of the ODE model."""
        if x == 'ton':
            out = self.ton
        elif x == 'u0':
            out = self.u0
        elif x == 's0':
            out = self.s0
        elif x == 'alpha':
            out = self.alpha
        elif x == 'beta':
            out = self.beta
        elif x == 'gamma':
            out = self.gamma
        elif x == 'scaling_u':
            out = self.scaling_u
        elif x == 'scaling_s':
            out = self.scaling_s
        return out

    def get_param_1d(self,
                     x,
                     condition=None,
                     sample=True,
                     mask_idx=None,
                     mask_to=1,
                     detach=False):
        """Retreive the parameter of the ODE model with a shape of (num_gene) or (num_batch, num_gene)."""
        param = self.get_param(x)
        if detach:
            param = param.detach()
        if self.is_full_vb and x in ['alpha', 'beta', 'gamma']:
            if sample:
                G = self.n_gene
                eps = torch.randn(G, device=self.device)
                if condition is not None:
                    y = param[:, 0] + eps*(param[:, 1].exp())
                else:
                    y = param[0] + eps*(param[1].exp())
            else:
                if condition is not None:
                    y = param[:, 0]
                else:
                    y = param[0]
        else:
            y = param

        y = y.exp()

        if condition is not None and y.ndim > 1:
            if mask_idx is not None:
                mask = torch.ones_like(condition)
                mask[:, mask_idx] = 0
                mask_flip = (~mask.bool()).int()
                y = torch.einsum('ij,jk->ik', condition * mask, y)\
                    + torch.einsum('ij,jk->ik', condition * mask_flip, self.one_mat if mask_to == 1 else self.zero_mat)
            else:
                y = torch.einsum('ij,jk->ik', condition, y)

        return y

    def _sample_ode_param(self, condition=None, sample=True):
        """Sample rate parameters for full vb or output fixed rate parameters when
        (1) random is set to False
        (2) full vb is not enabled
        """
        alpha = self.get_param_1d('alpha', condition, sample=sample)
        beta = self.get_param_1d('beta', condition, sample=sample)
        gamma = self.get_param_1d('gamma', condition, sample=sample)
        scaling_u = self.get_param_1d('scaling_u', condition, sample=sample)
        scaling_s = self.get_param_1d('scaling_s', condition, sample=sample)
        
        return alpha, beta, gamma, scaling_u, scaling_s

    def _clip_rate(self, rate, max_val):
        clip_fn = nn.Hardtanh(-16, np.log(max_val))
        return clip_fn(rate)

    def _compute_rho(self,
                     z,
                     which_net=1,
                     edge_index=None,
                     edge_weight=None,
                     condition=None):
        """Compute cellwise transcription rates of each gene."""
        net_rho = self.net_rho if which_net == 1 else self.net_rho2
        if condition is None:
            rho = (net_rho(z, edge_index, edge_weight)
                   if isinstance(net_rho, GraphDecoder) else
                   net_rho(z))
        else:
            rho = (net_rho(torch.cat((z, condition), 1), edge_index, edge_weight)
                   if isinstance(net_rho, GraphDecoder) else
                   net_rho(torch.cat((z, condition), 1)))
        return rho

    def _compute_xy(self,
                    t,
                    z,
                    t0,
                    xy0,
                    edge_index=None,
                    edge_weight=None,
                    condition=None):
        """Compute spatial coordinates of each cell."""
        if condition is None:
            coord = (self.net_coord(torch.cat((t, z, t0, xy0), 1), edge_index, edge_weight)
                     if isinstance(self.net_coord, MultiLayerGraphDecoder) else
                     self.net_coord(torch.cat((t, z, t0, xy0), 1)))
        else:
            coord = (self.net_coord(torch.cat((t, z, t0, xy0, condition), 1), edge_index, edge_weight)
                     if isinstance(self.net_coord, MultiLayerGraphDecoder) else
                     self.net_coord(torch.cat((t, z, t0, xy0, condition), 1)))
        return coord

    def _solve_ode(self,
                   t,
                   alpha,
                   beta,
                   gamma,
                   scaling_u,
                   scaling_s,
                   u0=None,
                   s0=None,
                   t0=None,
                   condition=None,
                   return_vel=False,
                   detach=True,
                   neg_slope=0.0):
        """Evaluate ODE solution, compatible with 2 stages of TopoVelo
        as well as batch correction

        Args:
            t (:class:`torch.tensor`): Time
            alpha (:class:`torch.tensor`):
                Cellwise transcription rate, (num_cell, num_gene)
            beta (:class:`torch.tensor`):
                Splicing rate, (num_gene)
            gamma (:class:`torch.tensor`):
                Degradation rate, (num_gene)
            scaling_u (:class:`torch.tensor`):
                Gene scaling factor of unspliced reads
            scaling_s (:class:`torch.tensor`):
                Gene scaling factor of spliced reads (used for multi-batch integration)
            u0 (:class:`torch.tensor`, optional):
                Initial condition of u in stage 2, (num_cell, num_gene). Defaults to None.
            s0 (:class:`torch.tensor`, optional):
                Initial condition of s in stage 2. Defaults to None.
            t0 (:class:`torch.tensor`, optional):
                Initial time in stage 2. Defaults to None.
            condition (:class:`torch.tensor`, optional):
                Batch information, (num_cell, num batch). Defaults to None.
            return_vel (bool, optional):
                Whether to return velocity. Defaults to False.
            detach (bool, optional):
                Whether to return detached tensors. Defaults to True.
            neg_slope (float, optional):
                Leaky ReLU parameter used for time clipping. Defaults to 0.0.

        Returns:
            Predicted u, s and velocity
        """
        if u0 is None or s0 is None or t0 is None:
            t0 = self.get_param_1d('ton', condition, sample=False, detach=detach)
            u0 = torch.stack([torch.zeros(self.u0.shape, device=alpha.device, dtype=float),
                              self.get_param_1d('u0', condition, sample=False, detach=detach)])
            s0 = torch.stack([torch.zeros(self.u0.shape, device=alpha.device, dtype=float),
                              self.get_param_1d('s0', condition, sample=False, detach=detach)])
            alpha = torch.stack([alpha,
                                 torch.zeros(alpha.shape, device=alpha.device, dtype=float)], 1)
            if condition is not None:
                beta = beta.unsqueeze(1)
                gamma = gamma.unsqueeze(1)
                scaling_u = scaling_u.unsqueeze(1)
                scaling_s = scaling_s.unsqueeze(1)
            tau = torch.stack([F.leaky_relu(t-t0, neg_slope) for i in range(2)], 1)
            uhat, shat = pred_su(tau, u0, s0, alpha, beta, gamma)
        else:
            tau = F.leaky_relu(t-t0, neg_slope)
            uhat, shat = pred_su(tau, u0/scaling_u, s0/scaling_s, alpha, beta, gamma)
        uhat = F.relu(uhat)
        shat = F.relu(shat)
        # Compute velocity
        vu, vs = None, None
        if return_vel:
            vu = alpha - beta * uhat
            vs = beta * uhat - gamma * shat
        uhat = uhat * scaling_u
        shat = shat * scaling_s
        
        return uhat, shat, vu, vs

    def forward_basis(self,
                      t, z,
                      batch_sample,
                      edge_index=None,
                      edge_weight=None,
                      condition=None,
                      eval_mode=False,
                      return_vel=False,
                      neg_slope=0.0):
        """Evaluates the basis functions (ODE solutions).
        Outputs a (n sample, n basis, n gene) tensor"""
        rho = self._compute_rho(z, 1, edge_index, edge_weight, condition)
        
        if batch_sample is not None:
            rho = rho[batch_sample]
        condition_batch = None if condition is None else condition[batch_sample]
        alpha, beta, gamma, scaling_u, scaling_s = self._sample_ode_param(condition_batch, sample=not eval_mode)
        if condition is not None:
            beta = beta.unsqueeze(1)
            gamma = gamma.unsqueeze(1)
            scaling_u = scaling_u.unsqueeze(1)
            scaling_s = scaling_s.unsqueeze(1)
        # tensor shape (n_cell, n_basis, n_gene)
        alpha = torch.stack([alpha*rho,
                             torch.zeros(rho.shape, device=rho.device, dtype=float)], 1)
        u0 = torch.stack([torch.zeros(self.u0.shape, device=rho.device, dtype=float),
                          self.u0.exp()])
        s0 = torch.stack([torch.zeros(self.u0.shape, device=rho.device, dtype=float),
                          self.s0.exp()])
        if batch_sample is not None:
            tau = torch.stack([F.leaky_relu(t[batch_sample] - self.ton.exp(), neg_slope) for i in range(2)], 1)
        else:
            tau = torch.stack([F.leaky_relu(t - self.ton.exp(), neg_slope) for i in range(2)], 1)
        
        Uhat, Shat = pred_su(tau,
                             u0,
                             s0,
                             alpha,
                             beta,
                             gamma)
        Uhat = F.relu(Uhat)
        Shat = F.relu(Shat)
        # Compute velocity
        vu, vs = None, None
        if return_vel:
            vu = alpha - beta * Uhat
            vs = beta * Uhat - gamma * Shat
        Uhat = Uhat * scaling_u
        Shat = Shat * scaling_s

        return Uhat, Shat, vu, vs

    def forward(self,
                t, z,
                batch_sample,
                edge_index=None,
                edge_weight=None,
                u0=None,
                s0=None,
                t0=None,
                condition=None,
                eval_mode=False,
                return_vel=False,
                neg_slope=0.0):
        """Top-level forward function for the decoder class"""
        if u0 is None or s0 is None or t0 is None:
            return self.forward_basis(t, z,
                                      batch_sample,
                                      edge_index,
                                      edge_weight,
                                      condition,
                                      eval_mode,
                                      return_vel,
                                      neg_slope)
        else:
            if batch_sample is not None:
                tau = F.leaky_relu(t[batch_sample]-t0, neg_slope)
            else:
                tau = F.leaky_relu(t-t0, neg_slope)
            rho = self._compute_rho(z, 2, edge_index, edge_weight, condition)
            
            if batch_sample is not None:
                rho = rho[batch_sample]
            condition_batch = None if condition is None else condition[batch_sample]
            alpha, beta, gamma, scaling_u, scaling_s = self._sample_ode_param(condition_batch, sample=not eval_mode)
            alpha = alpha*rho
            
            Uhat, Shat = pred_su(tau,
                                 u0/scaling_u,
                                 s0/scaling_s,
                                 alpha,
                                 beta,
                                 gamma)
            Uhat = F.relu(Uhat)
            Shat = F.relu(Shat)
            # Compute velocity
            Vu, Vs = None, None
            if return_vel:
                Vu = alpha - beta * Uhat
                Vs = beta * Uhat - gamma * Shat
            Uhat = Uhat * scaling_u
            Shat = Shat * scaling_s

        return Uhat, Shat, Vu, Vs


class VAE(VanillaVAE):
    """TopoVelo Model
    """
    def __init__(self,
                 adata: AnnData,
                 tmax: float,
                 dim_z: int,
                 dim_cond: int = 0,
                 dim_coord: int = 2,
                 dim_edge: Optional[int] = None,
                 device: str = 'cpu',
                 hidden_size: Tuple = (500, 250, 500),
                 spatial_hidden_size: Tuple = (128, 64),
                 full_vb: bool = False,
                 discrete: bool = False,
                 graph_decoder: bool = True,
                 spatial_decoder: bool = False,
                 attention: bool = True,
                 n_head: int = 5,
                 batch_key: Optional[str] = None,
                 ref_batch: Optional[int] = None,
                 slice_key: Optional[str] = None,
                 discrete_dim: Optional[int] = None,
                 init_method: str = "steady",
                 init_key: Optional[str] = None,
                 tprior: Optional[float] = None,
                 train_test_split: Tuple[float, float, float] = (0.7, 0.2, 0.1),
                 test_samples: Optional[np.ndarray] = None,
                 init_ton_zero: bool = True,
                 filter_gene: bool = False,
                 count_distribution: Literal["Poisson", "NB", "auto"] = "Poisson",
                 time_overlap: float = 0.05,
                 std_z_prior: float = 0.01,
                 min_sigma_u: float = 0.1,
                 min_sigma_s: float = 0.1,
                 xavier_gain: float = 0.05,
                 checkpoints: Tuple[Optional[str], Optional[str]] = [None, None],
                 rate_prior: Dict[str, Tuple] = {
                     'alpha': (0.0, 1.0),
                     'beta': (0.0, 0.5),
                     'gamma': (0.0, 0.5)
                 },
                 random_state: int = 2022,
                 verbose_level=logging.WARNING,
                 **kwargs):
        """TopoVelo Model

        Args:
            adata (:class:`anndata.AnnData`):
                AnnData object containing all relevant data and meta-data
            tmax (float):
                Time range.
                This is used to restrict the cell time within certain range. In the
                case of a Gaussian model without capture time, tmax/2 will be the mean of prior.
                If capture time is provided, then they are scaled to match a range of tmax.
                In the case of a uniform model, tmax is strictly the maximum time.
            dim_z (int):
                Dimension of the latent cell state
            dim_cond : int, optional
                Dimension of additional information for the conditional VAE.
                Set to zero by default, equivalent to a VAE.
                This feature is not stable now.
            dim_coord (int, optional):
                Dimension of the spatial coordinates.
            dim_edge (int, optional):
                Dimension of the edge features
            device ({'gpu','cpu'}, optional):
                Training device
            hidden_size (tuple[int], optional):
                Width of the hidden layers. Should be a tuple of the form
                (encoder layer 1, decoder layer 1, decoder layer 2)
                Defaults to (500, 250, 500).
            spatial_hidden_size (tuple[int], optional):
                Width of the hidden layers in the spatial decoder.
                Should be a tuple of the form (spatial layer 1, spatial layer 2).
                Defaults to (128, 64).
            full_vb (bool, optional):
                Enable the full variational Bayes. Defaults to False.
            discrete (bool, optional):
                Enable the discrete count model. Defaults to False.
            graph_decoder (bool, optional):
                Enable GNN as the VAE decoder. Defaults to True.
            spatial_decoder (bool, optional):
                Enable the spatial decoder. Defaults to False.
            attention (bool, optional):
                Enable the graph attention mechanism. Defaults to True.
            n_head (int, optional):
                Number of heads in the graph attention mechanism. Defaults to 5.
            batch_key (str, optional):
                Column in the AnnData object containing the batch information.
                Defaults to None.
            ref_batch (int, optional):
                Reference batch for the batch correction. Defaults to None.
            slice_key (str, optional):
                Column in the AnnData object containing the slice information.
                Defaults to None.
            discrete_dim (int, optional):
                Number of discrete dimensions, used in stage 2 KNN search.
                Defaults to None.
            init_method ({'tprior', 'steady'}, optional):
                Initialization method.
                Should choose from
                (1) tprior: use the capture time to estimate rate parameters. Cell time will be
                            randomly sampled with the capture time as the mean. The variance can
                            be controlled by changing 'time_overlap' in config.
                (2) steady: use the steady-state model to estimate gamma, alpha and assume beta = 1.
                            After this, a global cell time is estimated by taking the quantile over
                            all local times. Finally, rate parameters are reinitialized using the
                            global cell time.
                Defaults to 'steady'.
            init_key (str, optional):
                Column in the AnnData object containing the capture time.
                Defaults to None.
            tprior (str, optional):
                Key in adata.obs that stores the capture time.
                Used for informative time prior.
            train_test_split (tuple[float], optional):
                Train-test split ratio. Defaults to (0.7, 0.2, 0.1).
            test_samples (np.ndarray, optional):
                Indices of test samples. Defaults to None.
            init_ton_zero (bool, optional):
                Whether to add a non-zero switch-on time for each gene.
                It's set to True if there's no capture time. Defaults to True.
            filter_gene (bool, optional):
                Whether to remove non-velocity genes. Defaults to False.
            count_distriution ({'auto', 'Poisson', 'NB'}, optional):
                Count distribution, effective only when discrete=True
                The current version only assumes Poisson or negative binomial distributions.
                When set to 'auto', the program determines a proper one based on over dispersion
            time_overlap (float, optional):
                Overlap of two adjacent time intervals when using the capture time as the prior.
                This is used to control the variance of the prior. Defaults to 0.05.
            std_z_prior (float, optional):
                Standard deviation of the prior (isotropical Gaussian) of cell state.
            min_sigma_u (float, optional):
                Minimum value of Gaussian noise of unspliced counts. Defaults to 0.1.
            min_sigma_s (float, optional):
                Minimum value of Gaussian noise of spliced counts. Defaults to 0.1.
            xavier_gain (float, optional):
                Gain of the Xavier initialization. Default to 0.05.
            checkpoints (list of 2 strings, optional):
                Contains the path to saved encoder and decoder models.
                Should be a .pt file.
            rate_prior (dict, optional):
                Prior distribution of rate parameters.
                Keys are always `alpha',`beta',`gamma'
                Values are length-2 tuples (mu, sigma), representing the mean and standard deviation
                of log rates.
            random_state (int, optional):
                Random seed. Notice that pytorch_geometric graph attention is not entirely reproducible
                even with the same random seed. See https://github.com/pytorch/pytorch/issues/75179.
            verbose_level (int, optional):
                Logging level. Default to logging.WARNING.  
        """
        t_start = time.time()
        self.timer = 0
        self.is_discrete = discrete
        self.is_full_vb = full_vb
        early_stop_thred = adata.n_vars*1e-4 if self.is_discrete else adata.n_vars*1e-3

        # Training Configuration
        self.config = {
            # Model Parameters
            "dim_z": dim_z,
            "dim_edge": dim_edge,
            "hidden_size": hidden_size,
            "tmax": tmax,
            "init_method": init_method,
            "init_key": init_key,
            "tprior": tprior,
            "std_z_prior": std_z_prior,
            "tail": 0.01,
            "time_overlap": time_overlap,
            "n_neighbors": 10,
            "dt": (0.03, 0.06),
            "n_bin": None,
            "graph_decoder": graph_decoder,
            "batch_key": batch_key,
            "ref_batch": ref_batch,
            "epsilon_ball": 0.1,  # normalized epsilon ball radius

            # Training Parameters
            "batch_size": 128,
            "n_epochs": 1000,
            "n_epochs_post": 500,
            "n_refine": 20,
            "learning_rate": None,
            "learning_rate_ode": None,
            "learning_rate_refine": None,
            "lambda": 1e-3,
            "lambda_rho": 1e-3,
            "kl_t": 1.0,
            "kl_z": 1.0,
            "kl_w": 0.01,
            "reg_v": 0.0,
            "sigma_pos": None,
            "max_rate": 1e4,
            "test_iter": None,
            "save_epoch": 100,
            "n_warmup": 5,
            "early_stop": 50,
            "early_stop_refine": 10,
            "early_stop_thred": early_stop_thred,
            "train_test_split": train_test_split,
            "neg_slope": 0.0,
            "train_ton": (init_method != 'tprior'),
            "enable_edge_weight": True,
            "weight_sample": False,
            "vel_continuity_loss": False,
            "normalize_pos": False,

            # hyperparameters for full vb
            "kl_param": 1.0,

            # Normalization Configurations
            "scale_gene_encoder": True,
            "scale_cell_encoder": False,
            "log1p": False,

            # Plotting
            "sparsify": 1
        }

        self.set_device(device)
        self.encode_batch(adata)
        if slice_key is None:
            slice_key = batch_key
        self.discrete_dim = discrete_dim
        self.encode_slice(adata, slice_key)

        self.dim_z = dim_z
        self.enable_cvae = dim_cond > 0
        self.spatial_decoder = spatial_decoder

        seed_everything(random_state)
        self.split_train_validation_test(adata.n_obs, test_samples)
        self.decoder = Decoder(
            adata,
            tmax,
            self.train_idx,
            dim_z,
            N1=hidden_size[1],
            N2=hidden_size[2],
            spatial_hidden_size=spatial_hidden_size,
            full_vb=full_vb,
            discrete=discrete,
            graph_decoder=graph_decoder,
            attention=attention,
            n_head=n_head,
            dim_cond=self.n_batch,
            dim_coord=dim_coord,
            dim_edge=dim_edge,
            batch_idx=self.batch_,
            ref_batch=self.ref_batch,
            init_ton_zero=init_ton_zero,
            filter_gene=filter_gene,
            min_sigma_u=min_sigma_u,
            min_sigma_s=min_sigma_s,
            xavier_gain=xavier_gain,
            device=self.device,
            init_method=init_method,
            init_key=init_key,
            checkpoint=checkpoints[1],
            **kwargs
        ).float().to(self.device)

        try:
            G = adata.n_vars
            self.encoder = Encoder(2*G,
                                   dim_z,
                                   dim_cond,
                                   dim_edge,
                                   hidden_size[0],
                                   attention=attention,
                                   n_head=n_head,
                                   xavier_gain=xavier_gain,
                                   checkpoint=checkpoints[0]).to(self.device)
        except IndexError:
            logger.error('Please provide two dimensions!')

        self.tmax = tmax
        self.get_prior(adata, tmax, tprior)

        self._pick_loss_func(adata, count_distribution)

        self.p_z = torch.stack([torch.zeros(adata.shape[0], dim_z, device=self.device),
                                torch.ones(adata.shape[0], dim_z, device=self.device)*self.config["std_z_prior"]])\
                        .float()
        # Prior of Decoder Parameters
        self.p_log_alpha = torch.tensor([[rate_prior['alpha'][0]], [rate_prior['alpha'][1]]], device=self.device)
        self.p_log_beta = torch.tensor([[rate_prior['beta'][0]], [rate_prior['beta'][1]]], device=self.device)
        self.p_log_gamma = torch.tensor([[rate_prior['gamma'][0]], [rate_prior['gamma'][1]]], device=self.device)

        self.alpha_w = torch.tensor(find_dirichlet_param(0.5, 0.05), device=self.device).float()

        self.use_knn = False
        self.u0 = None
        self.s0 = None
        self.t0 = None
        self.x0_index = None
        self.u1 = None
        self.s1 = None
        self.t1 = None
        self.x1_index = None

        self.lu_scale = (
            torch.tensor(np.log(adata.obs['library_scale_u'].to_numpy()), device=self.device).unsqueeze(-1).float()
            if self.is_discrete else torch.zeros(adata.n_obs, 1, device=self.device).float()
            )
        self.ls_scale = (
            torch.tensor(np.log(adata.obs['library_scale_s'].to_numpy()), device=self.device).unsqueeze(-1).float()
            if self.is_discrete else torch.zeros(adata.n_obs, 1, device=self.device).float()
            )

        # Class attributes for training
        self.loss_train, self.loss_test = [], []
        self.counter = 0  # Count the number of iterations
        self.n_drop = 0  # Count the number of consecutive iterations with little decrease in loss
        self.train_stage = 1

        logger.setLevel(verbose_level)

        self.timer = time.time()-t_start

    def split_train_validation_test(self, N, test_samples=None):
        """Split training, validation and test samples.

        Args:
            N (int): Number of samples.
            test_samples (array like, optional): Indices of test samples. Defaults to None.
        """
        # Randomly select indices as training samples.
        if test_samples is None:
            rand_perm = np.random.permutation(N)
            n_train = int(N*self.config["train_test_split"][0])
            n_validation = int(N*self.config["train_test_split"][1])
            self.train_idx = rand_perm[:n_train]
            self.validation_idx = rand_perm[n_train:n_train+n_validation]
            self.test_idx = rand_perm[n_train+n_validation:]
        else:
            logger.info("Test samples provided. Distribute training and validation samples based on their proportions.")
            self.test_idx = test_samples
            train_valid_idx = np.array(list(set(range(N)).difference(set(test_samples))))
            rand_perm = np.random.permutation(train_valid_idx)
            n_train = int((N-len(test_samples))*self.config["train_test_split"][0])
            self.train_idx = rand_perm[:n_train]
            self.validation_idx = rand_perm[n_train:]
        return

    def encode_batch(self, adata):
        """Encode batch information.

        Args:
            adata (:class:`anndata.AnnData`): AnnData object.

        Raises:
            ValueError: If the reference batch is not found in the batch field.
        """
        self.n_batch = 0
        self.batch = None
        self.batch_ = None
        batch_count = None
        self.ref_batch = self.config['ref_batch']
        if self.config['batch_key'] is not None and self.config['batch_key'] in adata.obs:
            logger.info('CVAE enabled. Performing batch effect correction.')
            batch_raw = adata.obs[self.config['batch_key']].to_numpy()
            batch_names_raw, batch_count = np.unique(batch_raw, return_counts=True)
            self.batch_dic, self.batch_dic_rev = encode_type(batch_names_raw)
            self.n_batch = len(batch_names_raw)
            self.batch_ = np.array([self.batch_dic[x] for x in batch_raw])
            self.batch = torch.tensor(self.batch_, dtype=int, device=self.device)
            self.batch_names = np.array([self.batch_dic[batch_names_raw[i]] for i in range(self.n_batch)])
        if isinstance(self.ref_batch, int):
            if self.ref_batch >= self.n_batch:
                self.ref_batch = self.n_batch - 1
            elif self.ref_batch < -self.n_batch:
                self.ref_batch = 0
            logger.info(f'Reference batch set to {self.ref_batch} ({batch_names_raw[self.ref_batch]}).')
            if np.issubdtype(batch_names_raw.dtype, np.number) and 0 not in batch_names_raw:
                logger.warning('Integer batch names do not start from 0. Reference batch index may not match the actual batch name!')
        elif isinstance(self.ref_batch, str):
            if self.config['ref_batch'] in batch_names_raw:
                self.ref_batch = self.batch_dic[self.config['ref_batch']]
                logger.info(f'Reference batch set to {self.ref_batch} ({batch_names_raw[self.ref_batch]}).')
            else:
                raise ValueError('Reference batch not found in the provided batch field!')
        elif batch_count is not None:
            self.ref_batch = self.batch_names[np.argmax(batch_count)]
            logger.info(f'Reference batch set to {self.ref_batch} ({batch_names_raw[self.ref_batch]}).')
        self.enable_cvae = self.n_batch > 0
        if self.enable_cvae and 2*self.n_batch > self.config['dim_z']:
            logger.warning('Number of batch classes is larger than half of dim_z. Consider increasing dim_z.')
        if self.enable_cvae and 10*self.n_batch < self.config['dim_z']:
            logger.warning('Number of batch classes is smaller than 1/10 of dim_z. Consider decreasing dim_z.')

    def encode_slice(self, adata, slice_key):
        """Encode slice information when data consists of multiple slices.

        Args:
            adata (:class:`anndata.AnnData`): AnnData object.
            slice_key (str): Key in adata.obs that stores the slice information.
        """
        self.slice_label = None
        self.slices = None
        if slice_key in adata.obs:
            self.slice_label = adata.obs[slice_key].to_numpy()
            self.slices = np.unique(self.slice_label)

    def _pick_loss_func(self, adata, count_distribution):
        """Pick the corresponding loss function based on count distribution.

        Args:
            adata (:class:`anndata.AnnData`): AnnData object.
            count_distribution (str): Count distribution. Should be 'Gaussian', 'auto', 'Poisson', or 'NB'.
        """
        if self.is_discrete:
            # Determine Count Distribution
            dispersion_u = adata.var["dispersion_u"].to_numpy()
            dispersion_s = adata.var["dispersion_s"].to_numpy()
            if count_distribution == "auto":
                p_nb = np.sum((dispersion_u > 1) & (dispersion_s > 1))/adata.n_vars
                if p_nb > 0.5:
                    count_distribution = "NB"
                    self.vae_risk = self.vae_risk_nb
                else:
                    count_distribution = "Poisson"
                    self.vae_risk = self.vae_risk_poisson
                logger.debug(f"Mean dispersion: u={dispersion_u.mean():.2f}, s={dispersion_s.mean():.2f}")
                logger.debug(f"Over-Dispersion = {p_nb:.2f} => Using {count_distribution} to model count data.")
            elif count_distribution == "NB":
                self.vae_risk = self.vae_risk_nb
            else:
                self.vae_risk = self.vae_risk_poisson
            mean_u = adata.var["mean_u"].to_numpy()
            mean_s = adata.var["mean_s"].to_numpy()
            dispersion_u[dispersion_u < 1] = 1.001
            dispersion_s[dispersion_s < 1] = 1.001
            self.eta_u = torch.tensor(np.log(dispersion_u-1)-np.log(mean_u), device=self.device).float()
            self.eta_s = torch.tensor(np.log(dispersion_s-1)-np.log(mean_s), device=self.device).float()
        else:
            self.vae_risk = self.vae_risk_gaussian

    def forward(self,
                data_in,
                edge_index,
                batch_sample,
                lu_scale,
                ls_scale,
                edge_weight=None,
                u0=None,
                s0=None,
                t0=None,
                t1=None,
                condition=None):
        """Standard forward pass.

        Args:
            data_in (`torch.tensor`):
                Input count data, (num_cell, 2*num_gene).
            edge_index (`torch.tensor`):
                Row and column indices of non-zero entries of the adjacency matrix, (2, num_edge).
            batch_sample (`torch.tensor`):
                Indices of the sampled cells, (num_cell).
            lu_scale (`torch.tensor`):
                library size scaling_u factor of unspliced counts, (num_gene)
                Effective in the discrete mode and set to 1's in the continuouts model
            ls_scale (`torch.tensor`):
                Similar to lu_scale, but for spliced counts, (num_gene).
            edge_weight (`torch.tensor`, optional):
                Weights of the edges, (num_edge). Effective only in GCN. Defaults to None
            u0 (`torch.tensor`, optional): 
                Initial condition of u, (num_cell, num_gene).
                This is set to None in the first stage when cell time is not fixed.
                It will have some value in the second stage, so the users
                shouldn't worry about feeding the parameter themselves.
            s0 (`torch.tensor`, optional): 
                Initial condition of s, (num_cell, num_gene).
            t0 (`torch.tensor`, optional): 
                Time at the initial condition, (num_cell, 1).
            t1 (`torch.tensor`, optional): 
                Time at the future state.
                Used only when `vel_continuity_loss` is set to True
            condition (`torch.tensor`, optional): 
                Any additional condition to the VAE, (N, dim_cond)

        Returns:
            - mu_t (`torch.tensor`): time mean, (N,1)
            - std_t (`torch.tensor`): time standard deviation, (N,1)
            - mu_z (`torch.tensor`): cell state mean, (N, Cz)
            - std_z (`torch.tensor`): cell state standard deviation, (N, Cz)
            - t (`torch.tensor`): sampled cell time, (N,1)
            - z (`torch.tensor`): sampled cell sate, (N,Cz)
            - uhat (`torch.tensor`): predicted mean u values, (N,G)
            - shat (`torch.tensor`): predicted mean s values, (N,G)
            - optionally outputs predicted u/s at future time, u/s velocity at past and future times
        """
        data_in_scale = data_in
        G = data_in_scale.shape[-1]//2
        # optional data scaling_u
        if self.config["scale_gene_encoder"]:
            scaling_u = self.decoder.get_param_1d('scaling_u', condition, sample=False, detach=False)
            scaling_s = self.decoder.get_param_1d('scaling_s', condition, sample=False, detach=False)
            data_in_scale = torch.cat((data_in_scale[:, :G]/scaling_u,
                                       data_in_scale[:, G:]/scaling_s), 1)
        if self.config["scale_cell_encoder"]:
            data_in_scale = torch.cat((data_in_scale[:, :, :G]/lu_scale,
                                       data_in_scale[:, :, G:]/ls_scale), 1)
        if self.config["log1p"]:
            data_in_scale = torch.log1p(data_in_scale)

        mu_t, std_t, mu_z, std_z = self.encoder.forward(data_in_scale,
                                                        edge_index,
                                                        edge_weight,
                                                        condition)

        t = self.sample(mu_t, std_t)
        z = self.sample(mu_z, std_z)

        return_vel = self.config['reg_v'] or self.config['vel_continuity_loss']
        
        uhat, shat, vu, vs = self.decoder.forward(t,
                                                  z,
                                                  batch_sample,
                                                  edge_index,
                                                  edge_weight,
                                                  u0,
                                                  s0,
                                                  t0,
                                                  condition,
                                                  neg_slope=self.config["neg_slope"])

        if t1 is not None:  # predict the future state when we enable velocity continuity loss
            uhat_fw, shat_fw, vu_fw, vs_fw = self.decoder.forward(t1,
                                                                  z,
                                                                  batch_sample,
                                                                  edge_index,
                                                                  edge_weight,
                                                                  uhat,
                                                                  shat,
                                                                  t,
                                                                  condition,
                                                                  return_vel=return_vel,
                                                                  neg_slope=self.config["neg_slope"])
        else:
            uhat_fw, shat_fw, vu_fw, vs_fw = None, None, None, None

        return (mu_t[batch_sample], std_t[batch_sample],
                mu_z[batch_sample], std_z[batch_sample],
                t[batch_sample], z[batch_sample],
                uhat, shat,
                uhat_fw, shat_fw,
                vu, vs,
                vu_fw, vs_fw)

    def eval_model(self,
                   data_in,
                   edge_index,
                   lu_scale,
                   ls_scale,
                   edge_weight=None,
                   u0=None,
                   s0=None,
                   t0=None,
                   t1=None,
                   condition=None):
        """Standard forward pass in evaluation mode.

        Args:
            data_in (`torch.tensor`):
                Input count data, (num_cell, 2*num_gene).
            edge_index (`torch.tensor`):
                Row and column indices of non-zero entries of the adjacency matrix, (2, num_edge).
            lu_scale (`torch.tensor`):
                library size scaling_u factor of unspliced counts, (num_gene)
                Effective in the discrete mode and set to 1's in the continuouts model
            ls_scale (`torch.tensor`):
                Similar to lu_scale, but for spliced counts, (num_gene).
            edge_weight (`torch.tensor`, optional):
                Weights of the edges, (num_edge). Effective only in GCN. Defaults to None
            u0 (`torch.tensor`, optional): 
                Initial condition of u, (num_cell, num_gene).
                This is set to None in the first stage when cell time is not fixed.
                It will have some value in the second stage, so the users
                shouldn't worry about feeding the parameter themselves.
            s0 (`torch.tensor`, optional): 
                Initial condition of s, (num_cell, num_gene).
            t0 (`torch.tensor`, optional): 
                Time at the initial condition, (num_cell, 1).
            t1 (`torch.tensor`, optional): 
                Time at the future state.
                Used only when `vel_continuity_loss` is set to True
            condition (`torch.tensor`, optional): 
                Any additional condition to the VAE, (N, dim_cond)

        Returns:
            - mu_t (`torch.tensor`): time mean, (N,1)
            - std_t (`torch.tensor`): time standard deviation, (N,1)
            - mu_z (`torch.tensor`): cell state mean, (N, Cz)
            - std_z (`torch.tensor`): cell state standard deviation, (N, Cz)
            - uhat (`torch.tensor`): predicted mean u values, (N,G)
            - shat (`torch.tensor`): predicted mean s values, (N,G)
            - optionally outputs predicted u/s at future time, u/s velocity at past and future times
        """
        data_in_scale = data_in
        G = data_in_scale.shape[-1]//2
        # optional data scaling_u
        if self.config["scale_gene_encoder"]:
            scaling_u = self.decoder.get_param_1d('scaling_u', condition, sample=False, detach=False)
            scaling_s = self.decoder.get_param_1d('scaling_s', condition, sample=False, detach=False)
            data_in_scale = torch.cat((data_in_scale[:, :G]/scaling_u,
                                       data_in_scale[:, G:]/scaling_s), 1)
        if self.config["scale_cell_encoder"]:
            data_in_scale = torch.cat((data_in_scale[:, :, :G]/lu_scale,
                                       data_in_scale[:, :, G:]/ls_scale), 1)
        if self.config["log1p"]:
            data_in_scale = torch.log1p(data_in_scale)

        mu_t, std_t, mu_z, std_z = self.encoder.forward(data_in_scale,
                                                        edge_index,
                                                        edge_weight,
                                                        condition)
        
        return_vel = self.config['reg_v'] or self.config['vel_continuity_loss']
        uhat, shat, vu, vs = self.decoder.forward(mu_t,
                                                  mu_z,
                                                  None,
                                                  edge_index,
                                                  edge_weight,
                                                  u0,
                                                  s0,
                                                  t0,
                                                  condition,
                                                  neg_slope=self.config["neg_slope"])

        if t1 is not None:  # predict the future state when we enable velocity continuity loss
            uhat_fw, shat_fw, vu_fw, vs_fw = self.decoder.forward(t1,
                                                                  mu_z,
                                                                  None,
                                                                  edge_index,
                                                                  edge_weight,
                                                                  uhat,
                                                                  shat,
                                                                  mu_t,
                                                                  condition,
                                                                  return_vel=return_vel,
                                                                  neg_slope=self.config["neg_slope"])
        else:
            uhat_fw, shat_fw, vu_fw, vs_fw = None, None, None, None

        return (mu_t, std_t,
                mu_z, std_z,
                uhat, shat,
                uhat_fw, shat_fw,
                vu, vs,
                vu_fw, vs_fw)

    def pred_xy(self, t, z, t0, xy0, edge_index, edge_weight, condition=None, batch_sample=None, mode='train'):
            """
            Predicts the xy coordinates for a given time step and latent vector.

            Args:
                t (torch.Tensor): The latent cell time.
                z (torch.Tensor): The latent cell state.
                t0 (torch.Tensor): The initial time .
                xy0 (torch.Tensor): The initial xy coordinates.
                edge_index (torch.Tensor): The edge indices.
                edge_weight (torch.Tensor): The edge weights.
                condition (torch.Tensor, optional): The condition tensor. Defaults to None.
                batch_sample (int, optional): The index of the batch sample to return. Defaults to None.
                mode (str, optional): The mode of the model. Defaults to 'train'.

            Returns:
                torch.Tensor: The predicted xy coordinates.
            """
            self.set_mode(mode)
            xy_hat = self.decoder._compute_xy(t, z, t0, xy0, edge_index, edge_weight, condition)
            if batch_sample is not None:
                return xy_hat[batch_sample]
            return xy_hat

    def _ode_by_batch(self,
                      t,
                      z,
                      edge_index,
                      edge_weight,
                      u0,
                      s0,
                      t0,
                      condition,
                      return_vel,
                      eval_mode=True,
                      to_cpu=False):
        batch_size = min(1024, len(t))
        n_iter = t.shape[0] // batch_size
        device = torch.device('cpu') if to_cpu else self.device
        with torch.no_grad():
            if u0 is None or s0 is None or t0 is None:
                out_shape = (t.shape[0], 2, self.decoder.n_gene)
            else:
                out_shape = (t.shape[0], self.decoder.n_gene)
            uhat = torch.zeros(out_shape, device=device, dtype=torch.float32)
            shat = torch.zeros(out_shape, device=device, dtype=torch.float32)
            vu, vs = None, None
            if return_vel:
                vu = torch.zeros(out_shape, device=device, dtype=torch.float32)
                vs = torch.zeros(out_shape, device=device, dtype=torch.float32)
            which_net = 1 if (u0 is None or s0 is None or t0 is None) else 2
            
            rho = self.decoder._compute_rho(z, which_net, edge_index, edge_weight, condition)
            if condition is None:
                alpha = self.decoder.get_param_1d('alpha', condition, sample=False, detach=True)
                beta = self.decoder.get_param_1d('beta', condition, sample=False, detach=True)
                gamma = self.decoder.get_param_1d('gamma', condition, sample=False, detach=True)
                scaling_u = self.decoder.get_param_1d('scaling_u', condition, sample=False, detach=True)
                scaling_s = self.decoder.get_param_1d('scaling_s', condition, sample=False, detach=True)
            for i in range(n_iter+1):
                if i < n_iter:
                    batch_idx = torch.arange(i*batch_size, (i+1)*batch_size-1).to(device).long()
                else:
                    if n_iter * batch_size < len(t):
                        batch_idx = torch.arange(n_iter*batch_size, len(t)-1).to(device).long()
                    else:
                        break
                if condition is not None:
                    alpha = self.decoder.get_param_1d('alpha', condition[batch_idx], sample=False, detach=True)
                    beta = self.decoder.get_param_1d('beta', condition[batch_idx], sample=False, detach=True)
                    gamma = self.decoder.get_param_1d('gamma', condition[batch_idx], sample=False, detach=True)
                    scaling_u = self.decoder.get_param_1d('scaling_u', condition[batch_idx], sample=False, detach=True)
                    scaling_s = self.decoder.get_param_1d('scaling_s', condition[batch_idx], sample=False, detach=True)
                _u0 = None if u0 is None else u0[batch_idx]
                _s0 = None if s0 is None else s0[batch_idx]
                _t0 = None if t0 is None else t0[batch_idx]
                _uhat, _shat, _vu, _vs = self.decoder._solve_ode(t[batch_idx],
                                                                 alpha * rho[batch_idx],
                                                                 beta,
                                                                 gamma,
                                                                 scaling_u,
                                                                 scaling_s,
                                                                 u0=_u0,
                                                                 s0=_s0,
                                                                 t0=_t0,
                                                                 condition=condition,
                                                                 return_vel=return_vel,
                                                                 neg_slope=0.0,
                                                                 detach=True)
                uhat[batch_idx] = _uhat.to(device).float()
                shat[batch_idx] = _shat.to(device).float()
                torch.cuda.empty_cache()
                if return_vel:
                    vu[batch_idx] = _vu.to(device).float()
                    vs[batch_idx] = _vs.to(device).float()
                del _uhat, _shat, _vu, _vs
        return uhat, shat, vu, vs

    def eval_model_batch(self,
                         data_in,
                         edge_index,
                         lu_scale,
                         ls_scale,
                         edge_weight=None,
                         u0=None,
                         s0=None,
                         t0=None,
                         t1=None,
                         condition=None,
                         return_vel=False,
                         to_cpu=False):
        """Evaluate the model on the validation dataset by batch.
        
        Args:
            data_in (torch.Tensor): Input data tensor.
            edge_index (torch.Tensor): Edge index tensor.
            lu_scale (float): Scaling factor for the upstream gene expression.
            ls_scale (float): Scaling factor for the downstream gene expression.
            edge_weight (torch.Tensor, optional): Edge weight tensor. Defaults to None.
            u0 (torch.Tensor, optional): Initial upstream gene expression tensor. Defaults to None.
            s0 (torch.Tensor, optional): Initial downstream gene expression tensor. Defaults to None.
            t0 (torch.Tensor, optional): Initial time tensor. Defaults to None.
            t1 (torch.Tensor, optional): Final time tensor. Defaults to None.
            condition (torch.Tensor, optional): Condition tensor. Defaults to None.
            return_vel (bool, optional): Whether to return velocity tensors. Defaults to False.
            to_cpu (bool, optional): Whether to move tensors to CPU. Defaults to False.
        
        Returns:
            tuple: A tuple containing the following elements:
            - mu_t (torch.Tensor): Mean time tensor.
            - std_t (torch.Tensor): Standard deviation of time tensor.
            - mu_z (torch.Tensor): Mean latent space tensor.
            - std_z (torch.Tensor): Standard deviation of latent space tensor.
            - uhat (torch.Tensor): Predicted upstream gene expression tensor.
            - shat (torch.Tensor): Predicted downstream gene expression tensor.
            - uhat_fw (torch.Tensor, optional): Forward predicted upstream gene expression tensor. Defaults to None.
            - shat_fw (torch.Tensor, optional): Forward predicted downstream gene expression tensor. Defaults to None.
            - vu (torch.Tensor): Velocity of upstream gene expression tensor.
            - vs (torch.Tensor): Velocity of downstream gene expression tensor.
            - vu_fw (torch.Tensor, optional): Forward velocity of upstream gene expression tensor. Defaults to None.
            - vs_fw (torch.Tensor, optional): Forward velocity of downstream gene expression tensor. Defaults to None.
        """
        data_in_scale = data_in
        G = data_in_scale.shape[-1]//2
        # optional data scaling_u
        scaling_u = self.decoder.get_param_1d('scaling_u', condition, sample=False, detach=True)
        scaling_s = self.decoder.get_param_1d('scaling_s', condition, sample=False, detach=True)
        
        if self.config["scale_gene_encoder"]:
            data_in_scale = torch.cat((data_in_scale[:, :G]/scaling_u,
                                       data_in_scale[:, G:]/scaling_s), 1)

        if self.config["scale_cell_encoder"]:
            data_in_scale = torch.cat((data_in_scale[:, :G]/lu_scale,
                                       data_in_scale[:, G:]/ls_scale), 1)
        if self.config["log1p"]:
            data_in_scale = torch.log1p(data_in_scale)
        mu_t, std_t, mu_z, std_z = self.encoder.forward(data_in_scale,
                                                        edge_index,
                                                        edge_weight,
                                                        condition)
        return_vel = self.config['reg_v'] or self.config['vel_continuity_loss'] or return_vel

        # Compute ODE using batches to avoid GPU memory outage
        uhat, shat, vu, vs = self._ode_by_batch(mu_t,
                                                mu_z,
                                                edge_index,
                                                edge_weight,
                                                u0,
                                                s0,
                                                t0,
                                                condition,
                                                return_vel,
                                                eval_mode=True,
                                                to_cpu=to_cpu)
        if t1 is not None:
            uhat_fw, shat_fw, vu_fw, vs_fw = self._ode_by_batch(t1,
                                                                mu_z,
                                                                edge_index,
                                                                edge_weight,
                                                                uhat,
                                                                shat,
                                                                mu_t,
                                                                condition,
                                                                return_vel=return_vel,
                                                                eval_mode=True,
                                                                to_cpu=to_cpu)
        else:
            uhat_fw, shat_fw, vu_fw, vs_fw = None, None, None, None
        if to_cpu:
            mu_t = mu_t.cpu()
            std_t = std_t.cpu()
            mu_z = mu_z.cpu()
            std_z = std_z.cpu()
            if t1 is not None:
                uhat_fw = uhat_fw.cpu()
                shat_fw = shat_fw.cpu()
                vu_fw = vu_fw.cpu()
                vs_fw = vs_fw.cpu()
        return (mu_t, std_t,
                mu_z, std_z,
                uhat, shat,
                uhat_fw, shat_fw,
                vu, vs,
                vu_fw, vs_fw)

    def loss_vel(self, x0, xhat, v):
        """Calculates the velocity loss between the reconstructed input and the target velocity.

        Args:
            x0 (torch.Tensor): The original input tensor.
            xhat (torch.Tensor): The reconstructed input tensor.
            v (torch.Tensor): The target velocity tensor.

        Returns:
            torch.Tensor: The mean cosine similarity loss between the reconstructed input and the target velocity.
        """
        cossim = nn.CosineSimilarity(dim=1)
        return cossim(xhat-x0, v).mean()

    def _compute_kl_term(self, q_tx, p_t, q_zx, p_z):
        ##############################################################
        # Compute all KL-divergence terms
        # Arguments:
        # q_tx, q_zx: `tensor`
        #   conditional distribution of time and cell state given
        #   observation (count vector)
        # p_t, p_z: `tensor`
        #   Prior distribution, usually Gaussian
        ##############################################################
        kldt = self.kl_time(q_tx[0], q_tx[1], p_t[0], p_t[1], tail=self.config["tail"])
        kldz = kl_gaussian(q_zx[0], q_zx[1], p_z[0], p_z[1])
        kld_param = 0
        # In full VB, we treat all rate parameters as random variables
        if self.is_full_vb:
            kld_param = (kl_gaussian(self.decoder.alpha[0].view(1, -1),
                                     self.decoder.alpha[1].exp().view(1, -1),
                                     self.p_log_alpha[0],
                                     self.p_log_alpha[1])
                         + kl_gaussian(self.decoder.beta[0].view(1, -1),
                                       self.decoder.beta[1].exp().view(1, -1),
                                       self.p_log_beta[0],
                                       self.p_log_beta[1])
                         + kl_gaussian(self.decoder.gamma[0].view(1, -1),
                                       self.decoder.gamma[1].exp().view(1, -1),
                                       self.p_log_gamma[0],
                                       self.p_log_gamma[1])) / q_tx[0].shape[0]
        # In stage 1, dynamical mode weights are considered random
        kldw = (
            elbo_collapsed_categorical(self.decoder.logit_pw, self.alpha_w, 2, self.decoder.scaling_u.shape[0])
            if self.train_stage == 1 else 0
            )
        return (self.config["kl_t"]*kldt
                + self.config["kl_z"]*kldz
                + self.config["kl_param"]*kld_param
                + self.config["kl_w"]*kldw)

    def vae_risk_gaussian(self,
                          q_tx, p_t,
                          q_zx, p_z,
                          u, s,
                          uhat, shat,
                          uhat_fw=None, shat_fw=None,
                          u1=None, s1=None,
                          weight=None,
                          to_cpu=False):
        """Training objective function. This is the negative ELBO.

        Args:
            q_tx (tuple of `torch.tensor`):
                Parameters of time posterior. Mean and std are both (N, 1) tensors.
            p_t (tuple of `torch.tensor`):
                Parameters of time prior.
            q_zx (tuple of `torch.tensor`):
                Parameters of cell state posterior. Mean and std are both (N, Dz) tensors.
            p_z (tuple of `torch.tensor`):
                Parameters of cell state prior.
            u, s (`torch.tensor`):
                Input data
            uhat, shat (torch.tensor):
                Prediction by TopoVelo
            weight (`torch.tensor`, optional):
                Sample weight. This feature is not stable. Please consider setting it to None.

        Returns:
            torch.tensor - Negative ELBO
        """
        kl_term = self._compute_kl_term(q_tx, p_t, q_zx, p_z)

        sigma_u = self.decoder.sigma_u.cpu().exp() if to_cpu else self.decoder.sigma_u.exp()
        sigma_s = self.decoder.sigma_s.cpu().exp() if to_cpu else self.decoder.sigma_s.exp()

        # u and sigma_u has the original scale
        clip_fn = nn.Hardtanh(-P_MAX, P_MAX)
        if uhat.ndim == 3:  # stage 1
            logp = -0.5*((u.unsqueeze(1)-uhat)/sigma_u).pow(2)\
                   - 0.5*((s.unsqueeze(1)-shat)/sigma_s).pow(2)\
                   - torch.log(sigma_u)-torch.log(sigma_s*2*np.pi)
            logp = clip_fn(logp)
            pw = (F.softmax(self.decoder.logit_pw.cpu(), dim=1).T if to_cpu
                  else F.softmax(self.decoder.logit_pw, dim=1).T)
            logp = torch.sum(pw*logp, 1)
        else:
            logp = - 0.5*((u-uhat)/sigma_u).pow(2)\
                   - 0.5*((s-shat)/sigma_s).pow(2)\
                   - torch.log(sigma_u)-torch.log(sigma_s*2*np.pi)
            logp = clip_fn(logp)

        if uhat_fw is not None and shat_fw is not None:
            logp = logp - 0.5*((u1-uhat_fw)/sigma_u).pow(2)-0.5*((s1-shat_fw)/sigma_s).pow(2)

        if weight is not None:
            logp = logp*weight

        err_rec = torch.mean(torch.sum(logp, 1))
        
        return [- err_rec, kl_term]

    def _kl_poisson(self, lamb_1, lamb_2):
        return lamb_1 * (torch.log(lamb_1) - torch.log(lamb_2)) + lamb_2 - lamb_1

    def vae_risk_poisson(self,
                         q_tx, p_t,
                         q_zx, p_z,
                         u, s,
                         uhat, shat,
                         uhat_fw=None, shat_fw=None,
                         u1=None, s1=None,
                         weight=None,
                         to_cpu=False,
                         eps=1e-2):
        """
        Calculates the risk for a Variational Autoencoder (VAE) model using the Poisson distribution.

        Args:
            q_tx (Tensor): The tensor representing the mean of the latent variable t given x.
            p_t (Tensor): The tensor representing the prior distribution of the latent variable t.
            q_zx (Tensor): The tensor representing the mean of the latent variable z given x.
            p_z (Tensor): The tensor representing the prior distribution of the latent variable z.
            u (Tensor): The tensor representing the observed variable u.
            s (Tensor): The tensor representing the observed variable s.
            uhat (Tensor): The tensor representing the reconstructed variable uhat.
            shat (Tensor): The tensor representing the reconstructed variable shat.
            uhat_fw (Tensor, optional): The tensor representing the forward reconstructed variable uhat_fw.
                Defaults to None.
            shat_fw (Tensor, optional): The tensor representing the forward reconstructed variable shat_fw.
                Defaults to None.
            u1 (Tensor, optional): The tensor representing the future state of u. Defaults to None.
            s1 (Tensor, optional): The tensor representing the future state of s. Defaults to None.
            weight (Tensor, optional): The tensor representing the weight for each sample. Defaults to None.
            to_cpu (bool, optional): Whether to move the softmax output to the CPU. Defaults to False.
            eps (float, optional): A small value added to the input to avoid numerical instability. Defaults to 1e-2.

        Returns:
            list: A list containing the negative reconstruction error and the KL divergence term.

        """
        kl_term = self._compute_kl_term(q_tx, p_t, q_zx, p_z)

        # poisson
        mask_u = (torch.isnan(uhat) | torch.isinf(uhat)).float()
        uhat = uhat * (1-mask_u)
        mask_s = (torch.isnan(shat) | torch.isinf(shat)).float()
        shat = shat * (1-mask_s)

        poisson_u = Poisson(F.relu(uhat)+eps)
        poisson_s = Poisson(F.relu(shat)+eps)
        if uhat.ndim == 3:  # stage 1
            logp = poisson_u.log_prob(torch.stack([u, u], 1)) + poisson_s.log_prob(torch.stack([s, s], 1))
            pw = (F.softmax(self.decoder.logit_pw.cpu(), dim=1).T if to_cpu
                  else F.softmax(self.decoder.logit_pw, dim=1).T)
            logp = torch.sum(pw*logp, 1)
        else:
            logp = poisson_u.log_prob(u) + poisson_s.log_prob(s)

        # velocity continuity loss
        if uhat_fw is not None and shat_fw is not None:
            logp = logp - self._kl_poisson(u1, uhat_fw) - self._kl_poisson(s1, shat_fw)
        if weight is not None:
            logp = logp*weight

        err_rec = torch.mean(logp.sum(1))

        return [- err_rec, kl_term]

    def _kl_nb(self, m1, m2, p):
        r1 = m1 * (1 - p) / p
        r2 = m2 * (1 - p) / p
        return (r1 - r2)*torch.log(p)

    def vae_risk_nb(self,
                    q_tx, p_t,
                    q_zx, p_z,
                    u, s,
                    uhat, shat,
                    uhat_fw=None, shat_fw=None,
                    u1=None, s1=None,
                    weight=None,
                    to_cpu=False,
                    eps=1e-2):
        """
        Computes the negative log-likelihood loss and KL divergence loss for the Variational Autoencoder (VAE) model with negative binomial distribution.

        Args:
            q_tx (torch.Tensor): The mean of the distribution for t given x.
            p_t (torch.Tensor): The prior distribution for t.
            q_zx (torch.Tensor): The mean of the distribution for z given x.
            p_z (torch.Tensor): The prior distribution for z.
            u (torch.Tensor): The observed values of the variable u.
            s (torch.Tensor): The observed values of the variable s.
            uhat (torch.Tensor): The predicted values of the variable u.
            shat (torch.Tensor): The predicted values of the variable s.
            uhat_fw (torch.Tensor, optional): The predicted values of the variable u in the forward pass. Defaults to None.
            shat_fw (torch.Tensor, optional): The predicted values of the variable s in the forward pass. Defaults to None.
            u1 (torch.Tensor, optional): The observed values of the variable u in the first stage. Defaults to None.
            s1 (torch.Tensor, optional): The observed values of the variable s in the first stage. Defaults to None.
            weight (torch.Tensor, optional): The weight for each sample. Defaults to None.
            to_cpu (bool, optional): Whether to move the softmax logits to the CPU. Defaults to False.
            eps (float, optional): A small value to avoid numerical instability. Defaults to 1e-2.

        Returns:
            list: A list containing the negative log-likelihood loss and KL divergence loss.

        """
        kl_term = self._compute_kl_term(q_tx, p_t, q_zx, p_z)

        # NB
        p_nb_u = torch.sigmoid(self.eta_u+torch.log(F.relu(uhat)+eps))
        p_nb_s = torch.sigmoid(self.eta_s+torch.log(F.relu(shat)+eps))
        nb_u = NegativeBinomial((F.relu(uhat)+1e-10)*(1-p_nb_u)/p_nb_u, probs=p_nb_u)
        nb_s = NegativeBinomial((F.relu(shat)+1e-10)*(1-p_nb_s)/p_nb_s, probs=p_nb_s)
        if uhat.ndim == 3:  # stage 1
            logp = nb_u.log_prob(torch.stack([u, u], 1)) + nb_s.log_prob(torch.stack([s, s], 1))
            pw = (F.softmax(self.decoder.logit_pw.cpu(), dim=1).T if to_cpu
                  else F.softmax(self.decoder.logit_pw, dim=1).T)
            logp = torch.sum(pw*logp, 1)
        else:
            logp = nb_u.log_prob(u) + nb_s.log_prob(s)
        # velocity continuity loss
        if uhat_fw is not None and shat_fw is not None:
            logp = logp - self._kl_nb(uhat_fw, u1, p_nb_u) - self._kl_nb(shat_fw, s1, p_nb_s)
        if weight is not None:
            logp = logp*weight
        err_rec = torch.mean(torch.sum(logp, 1))

        return [- err_rec, kl_term]

    def vae_spatial_loss(self, xy, xy_hat):
        """
        Calculates the spatial loss for the VAE model.

        Args:
            xy (torch.Tensor): The ground truth xy coordinates.
            xy_hat (torch.Tensor): The predicted xy coordinates.

        Returns:
            torch.Tensor: The spatial loss.
        """
        return torch.mean(torch.sum(((xy-xy_hat)/self.config['sigma_pos']).pow(2), 1))

    def train_epoch(self, optimizer, optimizer2=None):
        """
        Train the model for one epoch.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer used to update the model parameters.
            optimizer2 (torch.optim.Optimizer, optional): A second optimizer used for alternating optimization. Default is None.

        Returns:
            bool: Whether to stop training based on the early stopping criterion.
        """
        G = self.graph_data.G
        self.set_mode('train')
        index_loader = DataLoader(Index(len(self.train_idx)),
                                  batch_size=self.config["batch_size"],
                                  shuffle=True,
                                  pin_memory=True)
        for i, index_batch in enumerate(index_loader):
            torch.cuda.empty_cache()

            batch_sample = self.train_idx[index_batch]
            if self.counter == 1 or self.counter % self.config["test_iter"] == 0:
                elbo_test = self.test(None,
                                      self.counter,
                                      True)
                if len(self.loss_test) > 0:  # update the number of epochs with dropping/converging ELBO
                    if np.sum(elbo_test) - np.sum(self.loss_test[-1]) <= self.config["early_stop_thred"]:
                        self.n_drop = self.n_drop+1
                    # else:
                    #     self.n_drop = 0
                self.loss_test.append(elbo_test)
                self.set_mode('train')

                patience = self.config["early_stop"] if self.train_stage == 1 else self.config["early_stop_refine"]
                if self.n_drop >= patience and patience > 0:
                    return True
            
            optimizer.zero_grad()
            if optimizer2 is not None:
                optimizer2.zero_grad()

            if self.graph_data.u0 is not None:
                u0, s0 = self.graph_data.u0[batch_sample], self.graph_data.s0[batch_sample]
                t0 = self.graph_data.t0[batch_sample]
            else:
                u0, s0, t0 = None, None, None
            if self.graph_data.u1 is not None:
                u1, s1 = self.graph_data.u1[batch_sample], self.graph_data.s1[batch_sample]
                t1 = self.graph_data.t1[batch_sample]
            else:
                u1, s1, t1 = None, None, None
            lu_scale = self.lu_scale[batch_sample].exp()
            ls_scale = self.ls_scale[batch_sample].exp()

            condition = F.one_hot(self.graph_data.batch, self.n_batch).float() if self.enable_cvae else None
            (mu_tx, std_tx,
             mu_zx, std_zx,
             t, z,
             uhat, shat,
             uhat_fw, shat_fw,
             vu, vs,
             vu_fw, vs_fw) = self.forward(self.graph_data.data.x,
                                          self.graph_data.data.adj_t,
                                          batch_sample,
                                          lu_scale, ls_scale,
                                          self.graph_data.edge_weight,
                                          u0, s0,
                                          t0, t1,
                                          condition)
            if uhat.ndim == 3:
                lu_scale = lu_scale.unsqueeze(-1)
                ls_scale = ls_scale.unsqueeze(-1)
            if uhat_fw is not None and shat_fw is not None:
                uhat_fw = uhat_fw[batch_sample]*lu_scale[batch_sample]
                shat_fw = uhat_fw[batch_sample]*ls_scale[batch_sample]
                u1 = u1[batch_sample]
                s1 = s1[batch_sample]
            loss_terms = self.vae_risk((mu_tx, std_tx),
                                       self.p_t[:, batch_sample, :],
                                       (mu_zx, std_zx),
                                       self.p_z[:, batch_sample, :],
                                       self.graph_data.data.x[batch_sample][:, :G],
                                       self.graph_data.data.x[batch_sample][:, G:],
                                       uhat*lu_scale,
                                       shat*ls_scale,
                                       uhat_fw, shat_fw,
                                       u1, s1,
                                       None)
            loss = loss_terms[0] + loss_terms[1]

            # Add velocity regularization
            if self.use_knn and self.config["reg_v"] > 0:
                scaling_u = self.decoder.get_param_1d('scaling_u', condition, sample=False, detach=True)
                scaling_s = self.decoder.get_param_1d('scaling_s', condition, sample=False, detach=True)
                loss = loss - self.config["reg_v"] *\
                    (self.loss_vel(u0/scaling_u, uhat/scaling_u*lu_scale, vu)\
                     + self.loss_vel(s0/scaling_s, shat/scaling_s*ls_scale, vs))
                if vu_fw is not None and vs_fw is not None:
                    loss = loss - self.config["reg_v"]\
                        * (self.loss_vel(uhat/scaling_u, uhat_fw/scaling_u, vu_fw)\
                           + self.loss_vel(shat/scaling_s, shat_fw/scaling_s, vs_fw))
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_value_(self.encoder.parameters(), GRAD_MAX)
            torch.nn.utils.clip_grad_value_(self.decoder.parameters(), GRAD_MAX)
            optimizer.step()
            if optimizer2 is not None:
                optimizer2.step()
            del (mu_tx, std_tx,
                 mu_zx, std_zx,
                 t, z,
                 uhat, shat,
                 uhat_fw, shat_fw,
                 vu, vs,
                 vu_fw, vs_fw)
            self.loss_train.append([loss_terms[i].detach().cpu().item() for i in range(len(loss_terms))])
            self.counter = self.counter + 1

        return False

    def train_spatial_epoch(self, optimizer):
        """
        Trains the spatial epoch of the VAE model.

        Args:
            optimizer: The optimizer used for training.

        Returns:
            bool: True if early stopping criteria is met, False otherwise.
        """
        self.set_mode('train')
        index_loader = DataLoader(Index(len(self.train_idx)),
                                  batch_size=self.config["batch_size"],
                                  shuffle=True,
                                  pin_memory=True)
        for i, index_batch in enumerate(index_loader):
            torch.cuda.empty_cache()
            batch_sample = self.train_idx[index_batch]
            if self.counter == 1 or self.counter % self.config["test_iter"] == 0:
                mse_test = self.test_spatial(self.counter, True)
                if len(self.loss_test_sp) > 0:  # update the number of epochs with dropping/converging ELBO
                    if self.loss_test_sp[-1] - mse_test <= self.config["early_stop_thred"]:
                        self.n_drop = self.n_drop+1
                    # else:
                    #     self.n_drop = 0
                self.loss_test_sp.append(mse_test)
                self.set_mode('train')

                patience = self.config["early_stop"] if self.train_stage == 1 else self.config["early_stop_refine"]
                if self.n_drop >= patience and patience > 0:
                    return True

            optimizer.zero_grad()

            condition = F.one_hot(self.graph_data.batch, self.n_batch).float() if self.enable_cvae else None
            xy_hat = self.pred_xy(self.graph_data.t,
                                  self.graph_data.z,
                                  self.graph_data.t0,
                                  self.graph_data.xy0,
                                  self.graph_data.data.adj_t,
                                  self.graph_data.edge_weight,
                                  condition,
                                  batch_sample)

            loss = self.vae_spatial_loss(self.graph_data.xy[batch_sample], xy_hat)
            loss.backward()

            optimizer.step()
            self.loss_train_sp.append(loss.detach().cpu().item())
            self.counter = self.counter + 1

        return False

    def update_x0(self):
        """
        Updates the initial conditions of the VAE model.

        This method sets the mode to 'eval' and performs the following steps:
        1. Predicts the output and computes the ELBO (Evidence Lower Bound) using the cell labels.
        2. Clips the time values to avoid outliers.
        3. Computes the initial conditions of cells without a valid pool of neighbors.
        4. Computes the knn indices for x0 and x1.
        5. Retrieves the initial conditions for x0 and x1.
        6. Updates the graph data with the computed values.

        Args:
            None

        Returns:
            None
        """
        self.set_mode('eval')
        G = self.graph_data.G
        out, elbo = self.pred_all(
            self.cell_labels,
            "both",
            ["uhat", "shat", "t", "z"],
            np.array(range(G))
        )
        t, z = out["t"], out["z"]

        # Clip the time to avoid outliers
        t = np.clip(t, 0, np.quantile(t, 0.99))
        dt = (self.config["dt"][0]*(t.max()-t.min()), self.config["dt"][1]*(t.max()-t.min()))
        u1, s1, t1 = None, None, None
        # Compute initial conditions of cells without a valid pool of neighbors
        with torch.no_grad():
            init_mask = (t <= np.quantile(t, 0.01))
            u0_init = np.mean(self.graph_data.data.x[init_mask][:, :G].detach().cpu().numpy(), 0)
            s0_init = np.mean(self.graph_data.data.x[init_mask][:, G:].detach().cpu().numpy(), 0)
        xy = self.graph_data.xy.detach().cpu().numpy()
        # Take out the discretized dimension (for 3D multi-slice data)
        d = xy.shape[1]
        if self.discrete_dim is None:
            knn_dim = np.array(range(d))
        else:
            if self.discrete_dim == 0:
                knn_dim = np.array(range(1, d))
            else:
                knn_dim = np.array(range(self.discrete_dim))
                if self.discrete_dim < d-1:
                    knn_dim = np.concatenate((knn_dim, np.array(range(self.discrete_dim+1, d))))
        if self.x0_index is None:
            if self.slice_label is None:
                self.x0_index = knnx0_index_parallel(
                    t[self.train_idx],
                    z[self.train_idx],
                    xy[self.train_idx][:, knn_dim],
                    t,
                    z,
                    xy[:, knn_dim],
                    dt,
                    self.config["n_neighbors"],
                    self.epsilon_ball,
                    hist_eq=True
                )
            else:
                self.x0_index = knnx0_index_batch(
                    t[self.train_idx],
                    z[self.train_idx],
                    xy[self.train_idx][:, knn_dim],
                    self.slice_label[self.train_idx],
                    t,
                    z,
                    xy[:, knn_dim],
                    self.slice_label,
                    dt,
                    self.config["n_neighbors"],
                    self.epsilon_ball,
                    hist_eq=True
                )
        u0, s0, xy0, t0 = get_x0(
            out["uhat"][self.train_idx],
            out["shat"][self.train_idx],
            xy[self.train_idx],
            t[self.train_idx],
            dt,
            self.x0_index,
            u0_init,
            s0_init
        )
        if self.config["vel_continuity_loss"] and self.x1_index is None:
            if self.slice_label is None:
                self.x1_index = knnx0_index_parallel(
                    t[self.train_idx],
                    z[self.train_idx],
                    xy[self.train_idx][:, knn_dim],
                    t,
                    z,
                    xy[:, knn_dim],
                    dt,
                    self.config["n_neighbors"],
                    self.epsilon_ball,
                    forward=True,
                    hist_eq=True,
                )
            else:
                self.x1_index = knnx0_index_batch(
                    t[self.train_idx],
                    z[self.train_idx],
                    xy[self.train_idx][:, knn_dim],
                    self.slice_label[self.train_idx],
                    t,
                    z,
                    xy[:, knn_dim],
                    self.slice_label,
                    dt,
                    self.config["n_neighbors"],
                    self.epsilon_ball,
                    forward=True,
                    hist_eq=True,
                    connect_adjacent=isinstance(self.discrete_dim, int)
                )
            u1, s1, xy1, t1 = get_x0(
                out["uhat"][self.train_idx],
                out["shat"][self.train_idx],
                xy[self.train_idx],
                t[self.train_idx],
                dt,
                self.x1_index,
                None,
                None,
                forward=True
            )

        self.graph_data.t = torch.tensor(t.reshape(-1, 1), dtype=torch.float32, device=self.device, requires_grad=False)
        self.graph_data.z = torch.tensor(z, dtype=torch.float32, device=self.device, requires_grad=False)
        self.graph_data.u0 = torch.tensor(u0, dtype=torch.float32, device=self.device, requires_grad=False)
        self.graph_data.s0 = torch.tensor(s0, dtype=torch.float32, device=self.device, requires_grad=False)
        self.graph_data.xy0 = torch.tensor(xy0, dtype=torch.float32, device=self.device, requires_grad=True)
        self.graph_data.t0 = torch.tensor(t0.reshape(-1, 1), dtype=torch.float32, device=self.device, requires_grad=False)

        self.u0 = u0
        self.s0 = s0
        self.t0 = t0.reshape(-1, 1)
        if self.config['vel_continuity_loss']:
            self.graph_data.u1 = torch.tensor(u1, dtype=torch.float32, device=self.device, requires_grad=False)
            self.graph_data.s1 = torch.tensor(s1, dtype=torch.float32, device=self.device, requires_grad=False)
            self.graph_data.xy1 = torch.tensor(xy1, dtype=torch.float32, device=self.device, requires_grad=False)
            self.graph_data.t1 = torch.tensor(t1.reshape(-1, 1), dtype=torch.float32, device=self.device, requires_grad=False)
            self.u1 = u1
            self.s1 = s1
            self.t1 = t1.reshape(-1, 1)

    def _set_lr(self, p):
        if self.is_discrete:
            self.config["learning_rate"] = 10**(-8.3*p-2.25)
            self.config["learning_rate_refine"] = self.config["learning_rate"]
            self.config["learning_rate_ode"] = 5*self.config["learning_rate"]
        else:
            self.config["learning_rate"] = 10**(-4*p-3)
            self.config["learning_rate_refine"] = self.config["learning_rate"]
            self.config["learning_rate_ode"] = 10*self.config["learning_rate"]

    def _set_sigma_pos(self, X_spatial):
        if self.config['sigma_pos'] is None:
            if self.config["normalize_pos"]:
                self.config['sigma_pos'] = 0.1 / np.sqrt(len(X_spatial))
            else:
                self.config['sigma_pos'] = 0.1 * np.min((X_spatial.max(0)-X_spatial.min(0)) / np.sqrt(len(X_spatial)))

    def _set_epsilon_ball(self, X_spatial):
        if not self.config["normalize_pos"]:
            #if self.enable_cvae:
            #    self.epsilon_ball = []
            #    for i in range(self.n_batch):
            #        xmax = X_spatial[self.batch_ == i].max(0)
            #        xmin = X_spatial[self.batch_ == i].min(0)
            #        self.epsilon_ball.append(self.config["epsilon_ball"] * np.min(xmax - xmin))
            #else:
            self.epsilon_ball = self.config["epsilon_ball"] * np.min(X_spatial.max(0)-X_spatial.min(0))

    def set_mode(self, mode='train'):
        VanillaVAE.set_mode(self, mode)
        if self.train_stage > 2:
            self.decoder.eval()
            if mode == 'train':
                self.decoder.net_coord.train()

    def _set_config(self, adata, config, spatial_key):
        """Set the configuration of the model.

        Args:
            adata (AnnData): Annotated data matrix.
            config (dict): Configuration dictionary.
            spatial_key (str): Key in adata.obsm storing the spatial coordinates.
        """
        self.load_config(config)
        if self.config["learning_rate"] is None:
            try:
                p = (np.sum(adata.layers["unspliced"].A > 0)
                    + (np.sum(adata.layers["spliced"].A > 0)))/adata.n_obs/adata.n_vars/2
            except AttributeError:  # dense matrix/array
                p = (np.sum(adata.layers["unspliced"] > 0)
                    + (np.sum(adata.layers["spliced"] > 0)))/adata.n_obs/adata.n_vars/2
            self._set_lr(p)
            logger.debug(f'Learning Rate based on Data Sparsity: {self.config["learning_rate"]:.4f}')
        self._set_sigma_pos(adata.obsm[spatial_key])
        self._set_epsilon_ball(adata.obsm[spatial_key])
        # Automatically set test iteration if not given
        if self.config["test_iter"] is None:
            self.config["test_iter"] = len(self.train_idx)//self.config["batch_size"]*2

    def _set_cell_type_label(self, adata, cluster_key):
        """Set cell type labels.

        Args:
            adata (AnnData): Annotated data matrix.
            cluster_key (str): Key in adata.obs storing the cell type annotation.
        """
        cell_labels_raw = (adata.obs[cluster_key].to_numpy() if cluster_key in adata.obs
                           else np.array(['Unknown' for i in range(adata.n_obs)]))
        # Encode the labels
        cell_types_raw = np.unique(cell_labels_raw)
        self.label_dic, self.label_dic_rev = encode_type(cell_types_raw)

        self.n_type = len(cell_types_raw)
        self.cell_labels = np.array([self.label_dic[x] for x in cell_labels_raw])
        self.cell_types = np.array([self.label_dic[cell_types_raw[i]] for i in range(self.n_type)])

    def _load_data(self, adata, graph, spatial_key, us_keys, embed, edge_attr):
        """Retreive data from adata and build training/validation datasets.

        Args:
            adata (AnnData): Annotated data matrix.
            spatial_key (str): Key in adata.obsm storing the spatial coordinates.
            us_keys (list): Keys in adata.layers storing the unspliced and spliced count matrices.
            embed (str): Low dimensional embedding in adata.obsm.
                The actual key storing the embedding should be f'X_{embed}'

        Returns:
            bool : Whether the data is successfully loaded.
        """
        if self.is_discrete:
            U, S = np.array(adata.layers['unspliced'].todense()), np.array(adata.layers['spliced'].todense())
            X = np.concatenate((U, S), 1).astype(int)
        elif us_keys is not None:
            U, S = adata.layers[us_keys[0]], adata.layers[us_keys[1]]
            if isinstance(U, csr_matrix) or isinstance(U, csc_matrix) or isinstance(U, coo_matrix):
                U, S = U.A, S.A
            X = np.concatenate((U, S), 1)
        else:
            X = np.concatenate((adata.layers['Mu'], adata.layers['Ms']), 1).astype(float)
        try:
            self.x_embed = adata.obsm[f"X_{embed}"]
        except KeyError:
            logger.warning("Embedding not found! Set to None.")
            self.x_embed = np.nan*np.ones((adata.n_obs, 2))
        
        print("*********               Creating a Graph Dataset              *********")
        self.graph_data = SCGraphData(
            X,
            self.cell_labels,
            graph,
            adata.obsm[spatial_key],
            self.train_idx,
            self.validation_idx,
            self.test_idx,
            self.device,
            edge_attr,
            self.batch_,
            self.config['enable_edge_weight'],
            self.config['normalize_pos']
        )
        print("*********                      Finished.                      *********")

    def train(self,
              adata,
              graph,
              spatial_key,
              edge_attr=None,
              config={},
              plot=False,
              gene_plot=[],
              cluster_key="clusters",
              us_keys=None,
              figure_path="figures",
              embed="umap",
              random_state=2022):
        """The high-level API for training.

        Args:
            adata (:class:`anndata.AnnData`):
                Annotated data matrix.
            graph (:class:`torch_geometric.data.Data`):
                Graph data object.
            spatial_key (str):
                Key in adata.obsm storing the spatial coordinates.
            edge_attr (:class:`numpy.ndarray`, optional):
                Edge attributes. Has a shape of (n_edge, dim_feature).
                The order should follow the edge_index in graph.
                Defaults to None.
            config (dictionary, optional):
                Contains all hyper-parameters.
            plot (bool, optional):
                Whether to plot some sample genes during training. Used for debugging.
            gene_plot (string list, optional):
                List of gene names to plot. Used only if plot==True
            cluster_key (str, optional):
                Key in adata.obs storing the cell type annotation.
            us_keys (list, optional):
                Keys in adata.layers storing the unspliced and spliced count matrices.
            figure_path (str, optional):
                Path to the folder for saving plots.
            embed (str, optional):
                Low dimensional embedding in adata.obsm. The actual key storing the embedding should be f'X_{embed}'
            random_state (int, optional):
                Random seed for reproducibility.
        """
        seed_everything(random_state)
        self._set_config(adata, config, spatial_key)
        self._set_cell_type_label(adata, cluster_key)
        self._load_data(adata, graph, spatial_key, us_keys, embed, edge_attr)

        print("--------------------------- Train a TopoVelo ---------------------------")
        gind, gene_plot = get_gene_index(adata.var_names, gene_plot)
        os.makedirs(figure_path, exist_ok=True)

        # define optimizer
        print("*********                 Creating optimizers                 *********")
        param_nn = list(self.encoder.parameters())\
            + list(self.decoder.net_rho.parameters())
        param_ode = [self.decoder.alpha,
                     self.decoder.beta,
                     self.decoder.gamma,
                     self.decoder.u0,
                     self.decoder.s0,
                     self.decoder.logit_pw]
        if self.config['train_ton']:
            param_ode.append(self.decoder.ton)
        if self.config['enable_edge_weight']:
            param_nn.append(self.graph_data.edge_weight)

        optimizer = torch.optim.Adam(param_nn, lr=self.config["learning_rate"], weight_decay=self.config["lambda"])
        optimizer_ode = torch.optim.Adam(param_ode, lr=self.config["learning_rate_ode"])
        print("*********                      Finished.                      *********")

        # Main Training Process
        print("*********                    Start training                   *********")
        print("*********                      Stage  1                       *********")
        n_epochs = self.config["n_epochs"]
        start = time.time()
        for epoch in range(n_epochs):
            # Train the encoder
            if epoch >= self.config["n_warmup"]:
                stop_training = self.train_epoch(optimizer_ode, optimizer)
            else:
                stop_training = self.train_epoch(optimizer, None)

            if plot and (epoch == 0 or (epoch+1) % self.config["save_epoch"] == 0):
                elbo_train = self.test(self.x_embed[self.train_idx],
                                       f"train{epoch+1}",
                                       False,
                                       gind,
                                       gene_plot,
                                       plot,
                                       figure_path)
                self.set_mode('train')
                elbo_test = self.loss_test[-1] if len(self.loss_test) > 0 else [-np.inf]
                logger.debug(f"Epoch {epoch+1}: Train ELBO = {np.sum(elbo_train):.3f},\t"
                             f"Test ELBO = {np.sum(elbo_test):.3f},\t"
                             f"Total Time = {convert_time(time.time()-start)}")

            if stop_training:
                print(f"*********       Stage 1: Early Stop Triggered at epoch {epoch+1}.       *********")
                print(f"Summary: \n"
                      f"Train ELBO = {-np.sum(self.loss_train[-1]):.3f}\n"
                      f"Test ELBO = {np.sum(self.loss_test[-1]):.3f}\n"
                      f"Total Time = {convert_time(time.time()-start)}\n")
                break

        count_epoch = epoch+1
        n_test1 = len(self.loss_test)

        print("*********                      Stage  2                       *********")
        self.encoder.eval()
        self.use_knn = True
        self.train_stage = 2
        self.num_test = len(self.loss_test)
        self.decoder.logit_pw.requires_grad = False
        if not self.is_discrete:
            sigma_u_prev = self.decoder.sigma_u.detach().cpu().numpy()
            sigma_s_prev = self.decoder.sigma_s.detach().cpu().numpy()
            u0_prev, s0_prev = None, None
            noise_change = np.inf
        x0_change = np.inf
        x0_change_prev = np.inf
        param_post = list(self.decoder.net_rho2.parameters())
        optimizer_post = torch.optim.Adam(param_post,
                                          lr=self.config["learning_rate_refine"],
                                          weight_decay=self.config["lambda_rho"])
        param_ode = [self.decoder.alpha,
                     self.decoder.beta,
                     self.decoder.gamma]
        optimizer_ode = torch.optim.Adam(param_ode, lr=self.config["learning_rate_ode"])
        for r in range(self.config['n_refine']):
            logger.info(f"*********             Velocity Refinement Round {r+1}              *********")
            self.config['early_stop_thred'] *= 0.95
            stop_training = (x0_change - x0_change_prev >= -0.01 and r > 1) or (x0_change < 0.01)
            if (not self.is_discrete) and (noise_change > 0.001) and (r < self.config['n_refine']-1):
                self.update_std_noise()
                stop_training = False

            if stop_training:
                print(f"Stage 2: Early Stop Triggered at round {r}.")
                break
            self.update_x0()
            self.n_drop = 0

            for epoch in range(self.config["n_epochs_post"]):
                if epoch >= self.config["n_warmup"]:
                    stop_training = self.train_epoch(optimizer_post, optimizer_ode)
                else:
                    stop_training = self.train_epoch(optimizer_post, None)

                if plot and (epoch == 0 or (epoch+count_epoch+1) % self.config["save_epoch"] == 0):
                    elbo_train = self.test(self.x_embed[self.train_idx],
                                           f"train{epoch+count_epoch+1}",
                                           False,
                                           gind,
                                           gene_plot,
                                           plot,
                                           figure_path)
                    self.decoder.train()
                    elbo_test = self.loss_test[-1] if len(self.loss_test) > n_test1 else [-np.inf]
                    logger.debug(f"Epoch {epoch+count_epoch+1}: Train ELBO = {np.sum(elbo_train):.3f},\t"
                                 f"Test ELBO = {np.sum(elbo_test):.3f},\t"
                                 f"Total Time = {convert_time(time.time()-start)}")

                if stop_training:
                    logger.debug(f"Summary: \n"
                                 f"Train ELBO = {-np.sum(self.loss_train[-1]):.3f}\n"
                                 f"Test ELBO = {np.sum(self.loss_test[-1]):.3f}\n"
                                 f"Total Time = {convert_time(time.time()-start)}\n")
                    logger.info(f"*********       "
                                f"Round {r+1}: Early Stop Triggered at epoch {epoch+count_epoch+1}."
                                f"       *********")
                    break
            count_epoch += (epoch+1)
            if not self.is_discrete:
                sigma_u = self.decoder.sigma_u.detach().cpu().numpy()
                sigma_s = self.decoder.sigma_s.detach().cpu().numpy()
                norm_delta_sigma = np.sum((sigma_u-sigma_u_prev)**2 + (sigma_s-sigma_s_prev)**2)
                norm_sigma = np.sum(sigma_u_prev**2 + sigma_s_prev**2)
                sigma_u_prev = self.decoder.sigma_u.detach().cpu().numpy()
                sigma_s_prev = self.decoder.sigma_s.detach().cpu().numpy()
                noise_change = norm_delta_sigma/norm_sigma
                logger.debug(f"Change in noise variance: {noise_change:.4f}")
            with torch.no_grad():
                if r > 0:
                    x0_change_prev = x0_change
                    norm_delta_x0 = np.sqrt(((self.u0 - u0_prev)**2 + (self.s0 - s0_prev)**2).sum(1).mean())
                    std_x = np.sqrt((self.u0.var(0) + self.s0.var(0)).sum())
                    x0_change = norm_delta_x0/std_x
                    logger.debug(f"Change in x0: {x0_change:.4f}")
                u0_prev = self.u0
                s0_prev = self.s0
        
        if self.spatial_decoder:
            print("*********                      Stage  3                       *********")
            count_epoch += epoch+1
            del param_nn, param_ode, param_post, optimizer, optimizer_ode
            param_sp = list(self.decoder.net_coord.parameters())
            optimizer = torch.optim.Adam(param_sp, lr=1e-3, weight_decay=self.config["lambda"])
            self.train_stage = 3
            self.n_drop = 0
            self.loss_train_sp = []
            self.loss_test_sp = []
            self.config["early_stop_thred"] = 0
            for epoch in range(n_epochs):
                stop_training = self.train_spatial_epoch(optimizer)

                if plot and (epoch == 0 or (epoch+1) % self.config["save_epoch"] == 0):
                    mse_train = self.test_spatial(f"train{epoch+1}", False, plot, figure_path)
                    self.set_mode('train')
                    mse_test = self.loss_test_sp[-1] if len(self.loss_test_sp) > 0 else [-np.inf]
                    logging.debug(f"Epoch {epoch+1}: Train MSE = {np.sum(mse_train):.3f},\t"
                                  f"Test MSE = {np.sum(mse_test):.3f},\t"
                                  f"Total Time = {convert_time(time.time()-start)}")

                if stop_training:
                    print(f"*********       Stage 3: Early Stop Triggered at epoch {count_epoch+epoch+1}.       *********")
                    logger.debug(f"Summary: \n"
                                 f"Train MSE = {np.sum(self.loss_train_sp[-1]):.3f}\n"
                                 f"Test MSE = {np.sum(self.loss_test_sp[-1]):.3f}\n"
                                 f"Total Time = {convert_time(time.time()-start)}\n")
                    break

            mse_train = self.test_spatial("final-train", False, plot, figure_path)
            mse_test = self.test_spatial("final-test", True, plot, figure_path)
            self.loss_train_sp.append(mse_train)
            self.loss_test_sp.append(mse_test)
            # Plot final results
            if plot:
                self.loss_train = np.stack(self.loss_train)
                plot_train_loss(self.loss_train[:, 0],
                                range(1, len(self.loss_train)+1),
                                save=f'{figure_path}/likelihood_train.png')
                plot_train_loss(self.loss_train[:, 1],
                                range(1, len(self.loss_train)+1),
                                save=f'{figure_path}/kl_train.png')
                plot_train_loss(self.loss_train_sp,
                                range(1, len(self.loss_train_sp)+1),
                                save=f'{figure_path}/xyloss_train.png')
                if self.config["test_iter"] > 0:
                    self.loss_test = np.stack(self.loss_test)
                    plot_test_loss(self.loss_test[:, 0],
                                   [i*self.config["test_iter"] for i in range(1, len(self.loss_test)+1)],
                                   save=f'{figure_path}/likelihood_validation.png')
                    plot_test_loss(self.loss_test[:, 1],
                                   [i*self.config["test_iter"] for i in range(1, len(self.loss_test)+1)],
                                   save=f'{figure_path}/kl_validation.png')
                    plot_test_loss(self.loss_test_sp,
                                   [i*self.config["test_iter"] for i in range(1, len(self.loss_test_sp)+1)],
                                   save=f'{figure_path}/xyloss_test.png')
            
        else:
            logger.info("Skipping the spatial decoder training.")
        
        # Final test
        elbo_train = self.test(self.x_embed[self.train_idx],
                               "final-train",
                               False,
                               gind,
                               gene_plot,
                               plot,
                               figure_path)
        elbo_test = self.test(self.x_embed[self.validation_idx],
                              "final-test",
                              True,
                              gind,
                              gene_plot,
                              plot,
                              figure_path)
        print(f"Final: Train ELBO = {np.sum(elbo_train):.3f},\tTest ELBO = {np.sum(elbo_test):.3f}")
        self.timer = self.timer + (time.time()-start)
        print(f"*********              Finished. Total Time = {convert_time(self.timer)}             *********")
        
        return

    def pred_all(self, cell_labels, mode='test', output=["uhat", "shat", "t", "z"], gene_idx=None):
        """
        Predicts the values of various variables for all cells.

        Args:
            cell_labels (array-like): The labels of the cells.
            mode (str, optional): The mode of prediction, choose from {'train', 'test', 'both'}. Defaults to 'test'.
            output (list, optional): The variables to be predicted. Defaults to ["uhat", "shat", "t", "z"].
            gene_idx (array-like, optional): The indices of the genes to be predicted. Defaults to None.

        Returns:
            dict: A dictionary containing the predicted values of the specified variables.

        Notes:
            - The predicted values are based on the trained model.
            - If gene_idx is not specified, all genes will be predicted.
            - The output dictionary will contain the predicted values for each specified variable.
        """
        N, G = self.graph_data.N, self.graph_data.G
        if gene_idx is None:
            gene_idx = np.array(range(G))
        save_uhat_fw = "uhat_fw" in output and self.use_knn and self.config["vel_continuity_loss"]
        save_shat_fw = "shat_fw" in output and self.use_knn and self.config["vel_continuity_loss"]

        with torch.no_grad():
            w_hard = F.one_hot(torch.argmax(self.decoder.logit_pw.cpu(), 1), num_classes=2).T
            if mode == "test":
                sample_idx = self.validation_idx
            elif mode == "train":
                sample_idx = self.train_idx
            else:
                sample_idx = np.array(range(self.graph_data.N))
            u0 = self.graph_data.u0
            s0 = self.graph_data.s0
            t0 = self.graph_data.t0
            u1 = self.graph_data.u1
            s1 = self.graph_data.s1
            t1 = self.graph_data.t1
            lu_scale = self.lu_scale.exp()
            ls_scale = self.ls_scale.exp()
            y_onehot = (F.one_hot(self.graph_data.batch, self.n_batch).float()
                        if self.enable_cvae else None)

            p_t = self.p_t[:, sample_idx, :].cpu()
            p_z = self.p_z[:, sample_idx, :].cpu()
            (mu_tx, std_tx,
             mu_zx, std_zx,
             uhat, shat,
             uhat_fw, shat_fw,
             vu, vs,
             vu_fw, vs_fw) = self.eval_model_batch(self.graph_data.data.x,
                                                   self.graph_data.data.adj_t,
                                                   lu_scale,
                                                   ls_scale,
                                                   self.graph_data.edge_weight,
                                                   u0,
                                                   s0,
                                                   t0,
                                                   t1,
                                                   y_onehot,
                                                   return_vel=("vs" in output or "vu" in output),
                                                   to_cpu=True)
            lu_scale = lu_scale.cpu()
            ls_scale = ls_scale.cpu()
            if uhat.ndim == 3:
                lu_scale = lu_scale.unsqueeze(-1)
                ls_scale = ls_scale.unsqueeze(-1)
            if uhat_fw is not None and shat_fw is not None:
                uhat_fw = uhat_fw[sample_idx] * lu_scale[sample_idx]
                shat_fw = uhat_fw[sample_idx] * ls_scale[sample_idx]
                u1 = u1[sample_idx].cpu()
                s1 = s1[sample_idx].cpu()

            loss_terms = self.vae_risk((mu_tx[sample_idx], std_tx[sample_idx]), p_t,
                                       (mu_zx[sample_idx], std_zx[sample_idx]), p_z,
                                       self.graph_data.data.x[sample_idx, :G].cpu(),
                                       self.graph_data.data.x[sample_idx, G:].cpu(),
                                       uhat[sample_idx].cpu()*lu_scale[sample_idx],
                                       shat[sample_idx].cpu()*ls_scale[sample_idx],
                                       uhat_fw, shat_fw,
                                       u1, s1,
                                       None,
                                       to_cpu=True)

        out = {}
        if "uhat" in output:
            if uhat.ndim == 3:
                uhat = torch.sum(uhat*w_hard, 1)
            out["uhat"] = uhat[sample_idx][:, gene_idx].numpy()
        if "shat" in output:
            if shat.ndim == 3:
                shat = torch.sum(shat*w_hard, 1)
            out["shat"] = shat[sample_idx][:, gene_idx].numpy()
        if "t" in output:
            out["t"] = mu_tx[sample_idx].detach().cpu().squeeze().numpy()
            out["std_t"] = std_tx[sample_idx].detach().cpu().squeeze().numpy()
        if "z" in output:
            out["z"] = mu_zx[sample_idx].detach().cpu().numpy()
            out["std_z"] = std_zx[sample_idx].detach().cpu().numpy()
        if "vs" in output:
            out["vs"] = vs[sample_idx][:, gene_idx].detach().cpu().numpy()
        if "vu" in output:
            out["vu"] = vu[sample_idx][:, gene_idx].detach().cpu().numpy()
        if save_uhat_fw:
            out["uhat_fw"] = uhat_fw[sample_idx][:, gene_idx].numpy()
        if save_shat_fw:
            out["shat_fw"] = shat_fw[sample_idx][:, gene_idx].numpy()
        del (mu_tx, std_tx,
             mu_zx, std_zx,
             uhat, shat,
             uhat_fw, shat_fw,
             vu, vs,
             vu_fw, vs_fw)

        return out, [-loss_terms[i].cpu().item() for i in range(len(loss_terms))]

    def reload_training(self,
                        adata,
                        graph,
                        spatial_key,
                        key,
                        edge_attr=None,
                        config={},
                        cluster_key="clusters",
                        us_keys=None,
                        embed="umap",
                        random_state=2022):
        """Regenerate the training scene without actually training the model.

        Args:
            adata (:class:`anndata.AnnData`): AnnData object.
            graph (:class:`numpy.ndarray`): Spatial graph.
            spatial_key (str): Key for extracting spatial coordinates.
            key (str): Key for storing pretrained model parameters in AnnData
            edge_attr (:class:`numpy.ndarray`, optional):
                Edge attributes. Has a shape of (n_edge, dim_feature).
                The order should follow the edge_index in graph.
                Defaults to None.
            config (dictionary, optional):
                Contains all hyper-parameters.
            cluster_key (str, optional):
                Key in adata.obs storing the cell type annotation.
            us_keys (list, optional):
                Keys in adata.layers storing the unspliced and spliced count matrices.
            embed (str, optional):
                Low dimensional embedding in adata.obsm. The actual key storing the embedding should be f'X_{embed}'
            random_state (int, optional):
                Random seed for reproducibility.
        """
        seed_everything(random_state)
        print("--------------------------- Reloading a TopoVelo ---------------------------")
        self._set_config(adata, config, spatial_key)
        self._set_cell_type_label(adata, cluster_key)
        self._load_data(adata, graph, spatial_key, us_keys, embed, edge_attr)

        self.use_knn = True
        self.train_stage = 2

        self.graph_data.t = torch.tensor(adata.obs[f'{key}_time'].to_numpy().reshape(-1, 1),
                                         dtype=torch.float32,
                                         device=self.device,
                                         requires_grad=False)
        self.graph_data.z = torch.tensor(adata.obsm[f'{key}_z'],
                                         dtype=torch.float32,
                                         device=self.device,
                                         requires_grad=False)
        self.graph_data.u0 = torch.tensor(adata.layers[f'{key}_u0'],
                                          dtype=torch.float32,
                                          device=self.device,
                                          requires_grad=False)
        self.graph_data.s0 = torch.tensor(adata.layers[f'{key}_s0'],
                                          dtype=torch.float32,
                                          device=self.device,
                                          requires_grad=False)
        self.graph_data.xy0 = torch.tensor(adata.obsm[f'X_{key}_xy0'],
                                           dtype=torch.float32,
                                           device=self.device,
                                           requires_grad=True)
        self.graph_data.t0 = torch.tensor(adata.obs[f'{key}_t0'].to_numpy().reshape(-1, 1),
                                          dtype=torch.float32,
                                          device=self.device,
                                          requires_grad=True)

        return
    
    def resume_train_stage_3(self,
                             adata,
                             lr=1e-3,
                             plot=False,
                             gene_plot=[],
                             figure_path="figures",
                             embed="umap",
                             random_state=2022):
        """The high-level API for resuming training the spatial decoder.

        Args:
            adata (:class:`anndata.AnnData`): Annotated data matrix.
            lr (float, optional): Learning rate for the spatial decoder. Defaults to 1e-3.
            plot (bool, optional): Whether to plot some sample genes during training. Used for debugging.
            gene_plot (string list, optional): List of gene names to plot. Used only if plot==True
            figure_path (str, optional): Path to the folder for saving plots.
            embed (str, optional): Low dimensional embedding in adata.obsm.
                The actual key storing the embedding should be f'X_{embed}'
            random_state (int, optional): Random seed for reproducibility.
        """
        start = time.time()
        seed_everything(random_state)
        print("*********                  Resuming Stage  3                  *********")
        gind, gene_plot = get_gene_index(adata.var_names, gene_plot)
        param_sp = list(self.decoder.net_coord.parameters())
        optimizer = torch.optim.Adam(param_sp, lr=lr, weight_decay=self.config["lambda"])
        self.train_stage = 3
        self.n_drop = 0
        self.loss_train_sp = []
        self.loss_test_sp = []
        self.config["early_stop_thred"] = 0
        
        if "X_embed" not in adata.obsm:
            logger.warning("Embedding not found! Skip plotting.")
            plot = False
        for epoch in range(self.config['n_epochs']):
            stop_training = self.train_spatial_epoch(optimizer)

            if plot and (epoch == 0 or (epoch+1) % self.config["save_epoch"] == 0):
                mse_train = self.test_spatial(f"train{epoch+1}", False, plot, figure_path)
                self.set_mode('train')
                mse_test = self.loss_test_sp[-1] if len(self.loss_test_sp) > 0 else [-np.inf]
                logger.debug(f"Epoch {epoch+1}: Train MSE = {np.sum(mse_train):.3f},\t"
                             f"Test MSE = {np.sum(mse_test):.3f},\t"
                             f"Total Time = {convert_time(time.time()-start)}")

            if stop_training:
                print(f"*********       Stage 3: Early Stop Triggered at epoch {epoch+1}.       *********")
                print(f"Summary: \n"
                      f"Train MSE = {np.sum(self.loss_train_sp[-1]):.3f}\n"
                      f"Test MSE = {np.sum(self.loss_test_sp[-1]):.3f}\n"
                      f"Total Time = {convert_time(time.time()-start)}\n")
                break

        mse_train = self.test_spatial("final-train", False, plot, figure_path)
        mse_test = self.test_spatial("final-test", True, plot, figure_path)
        self.loss_train_sp.append(mse_train)
        self.loss_test_sp.append(mse_test)
        # Plot final results
        if plot:
            self.loss_train_sp = np.stack(self.loss_train_sp)
            plot_train_loss(self.loss_train_sp,
                            range(1, len(self.loss_train_sp)+1),
                            save=f'{figure_path}/xyloss_train.png')
            if self.config["test_iter"] > 0:
                plot_test_loss(self.loss_test_sp,
                               [i*self.config["test_iter"] for i in range(1, len(self.loss_test_sp)+1)],
                               save=f'{figure_path}/xyloss_test.png')

        self.timer = self.timer + (time.time()-start)
        print(f"*********              Finished. Total Time = {convert_time(self.timer)}             *********")
        print(f"\tTrain MSE = {mse_train:.3f},\tTestMSE = {mse_test:.3f}")
        return

    def test(self,
             x_embed,
             testid=0,
             test_mode=True,
             gind=[],
             gene_plot=[],
             plot=False,
             path='figures',
             **kwargs):
        """Evaluate the model upon training/test dataset.

        Args:
            x_embed (`numpy array`):
                Low-dimensional embedding for plotting
            testid (str or int, optional):
                Used to name the figures.
            test_mode (bool, optional):
                Whether dataset is training or validation dataset. This is used when retreiving certain class variable,
                e.g. cell-specific initial condition.
            gind (`numpy array`, optional):
                Index of genes in adata.var_names. Used for plotting.
            gene_plot (`numpy array`, optional):
                Gene names.
            plot (bool, optional):
                Whether to generate plots.
            path (str):
                Saving path.

        Returns:
            list[float] : elbo_terms
        """
        self.set_mode('eval')
        mode = "test" if test_mode else "train"
        
        out_type = ["uhat", "shat", "uhat_fw", "shat_fw", "t"]
        if self.train_stage == 2:
            out_type.append("v")
        out, elbo_terms = self.pred_all(self.cell_labels, mode, out_type, gind)
        Uhat, Shat, t = out["uhat"], out["shat"], out["t"]
        G = self.graph_data.G

        if plot:
            # Plot Time
            plot_time(t, x_embed, save=f"{path}/time-{testid}-TopoVelo.png")
            cell_labels = np.array([self.label_dic_rev[x] for x in self.cell_labels])
            cell_labels = cell_labels[self.validation_idx] if test_mode else cell_labels[self.train_idx]
            # Plot u/s-t and phase portrait for each gene
            cell_idx = np.array(range(self.graph_data.N))
            if mode == "train":
                cell_idx = self.train_idx
            elif mode == "test":
                cell_idx = self.validation_idx
            for i in range(len(gind)):
                idx = gind[i]
                u = self.graph_data.data.x[cell_idx, idx].detach().cpu().numpy()
                s = self.graph_data.data.x[cell_idx, idx+G].detach().cpu().numpy()
                plot_sig(t.squeeze(),
                         u,
                         s,
                         Uhat[:, i], Shat[:, i],
                         cell_labels,
                         gene_plot[i],
                         save=f"{path}/sig-{gene_plot[i]}-{testid}.png",
                         sparsify=self.config['sparsify'])
                if self.config['vel_continuity_loss'] and self.train_stage == 2:
                    plot_sig(t.squeeze(),
                             u,
                             s,
                             out["uhat_fw"][:, i], out["shat_fw"][:, i],
                             cell_labels,
                             gene_plot[i],
                             save=f"{path}/sig-{gene_plot[i]}-{testid}-bw.png",
                             sparsify=self.config['sparsify'])
            plt.close('all')

        return elbo_terms

    def test_spatial(self,
                     testid,
                     mode,
                     plot=False,
                     path='figures'):
            """Evaluate the spatial decoder model upon training/test dataset.

            Args:
                testid (int): The identifier for the test.
                mode (str): Which part of the data to evaluate, should be one of {'train', 'test', 'both'}.
                plot (bool, optional): Flag to indicate if the results should be plotted. Defaults to False.
                path (str, optional): The path to save the figures. Defaults to 'figures'.

            Returns:
                float: The loss value.

            Notes:
                This method evaluates the spatial decoder model using the training/test dataset. It calculates the loss
                between the predicted and actual coordinates and returns the loss value. If the `plot` flag is set to True,
                it also generates and saves the plots of the predicted coordinates.
            """
            if mode == "test":
                sample_idx = self.validation_idx
            elif mode == "train":
                sample_idx = self.train_idx
            else:
                sample_idx = np.array(range(self.graph_data.N))

            with torch.no_grad():
                y_onehot = (F.one_hot(self.graph_data.batch, self.n_batch).float()
                            if self.enable_cvae else None)
                xy_hat = self.pred_xy(self.graph_data.t,
                                      self.graph_data.z,
                                      self.graph_data.t0,
                                      self.graph_data.xy0,
                                      self.graph_data.data.adj_t,
                                      self.graph_data.edge_weight,
                                      y_onehot,
                                      None,
                                      mode='eval')
                loss = self.vae_spatial_loss(self.graph_data.xy[sample_idx], xy_hat[sample_idx]) 
                loss *= self.config['sigma_pos']
                loss *= self.config['sigma_pos']
            if plot:
                cell_labels = np.array([self.label_dic_rev[x] for x in self.cell_labels])
                plot_cluster(xy_hat.cpu().numpy()[self.train_idx],
                             cell_labels[self.train_idx],
                             embed='Predicted Coordinates',
                             real_aspect_ratio=True,
                             save=f"{path}/xy-{testid}-train.png")
                plot_cluster(xy_hat.cpu().numpy()[self.test_idx],
                             cell_labels[self.test_idx],
                             embed='Predicted Coordinates',
                             real_aspect_ratio=True,
                             save=f"{path}/xy-{testid}-test.png")
            return loss.cpu().item()
    
    def pred_inductive(self, output=["uhat", "shat", "t", "z"], gene_idx=None):
        N, G = self.unseen_data.N, self.unseen_data.G
        if gene_idx is None:
            gene_idx = np.array(range(G))
        save_uhat_fw = "uhat_fw" in output and self.use_knn and self.config["vel_continuity_loss"]
        save_shat_fw = "shat_fw" in output and self.use_knn and self.config["vel_continuity_loss"]

        with torch.no_grad():
            w_hard = F.one_hot(torch.argmax(self.decoder.logit_pw.cpu(), 1), num_classes=2).T
            u0 = self.unseen_data.u0
            s0 = self.unseen_data.s0
            t0 = self.unseen_data.t0
            u1 = self.unseen_data.u1
            s1 = self.unseen_data.s1
            t1 = self.unseen_data.t1
            lu_scale = self.lu_scale_unseen.exp()
            ls_scale = self.ls_scale_unseen.exp()
            y_onehot = (torch.zeros(N, self.n_batch).float() if self.enable_cvae else None)

            p_t = self.p_t_unseen.cpu()
            p_z = torch.stack([torch.zeros(N, self.dim_z),
                               torch.ones(N, self.dim_z)*self.config["std_z_prior"]]).float()
            (mu_tx, std_tx,
             mu_zx, std_zx,
             uhat, shat,
             uhat_fw, shat_fw,
             vu, vs,
             vu_fw, vs_fw) = self.eval_model_batch(self.unseen_data.data.x,
                                                   self.unseen_data.data.adj_t,
                                                   lu_scale,
                                                   ls_scale,
                                                   self.unseen_data.edge_weight,
                                                   u0,
                                                   s0,
                                                   t0,
                                                   t1,
                                                   y_onehot,
                                                   return_vel=("vs" in output or "vu" in output),
                                                   to_cpu=True)
            lu_scale = lu_scale.cpu()
            ls_scale = ls_scale.cpu()
            if uhat.ndim == 3:
                lu_scale = lu_scale.unsqueeze(-1)
                ls_scale = ls_scale.unsqueeze(-1)
            if uhat_fw is not None and shat_fw is not None:
                uhat_fw = uhat_fw*lu_scale
                shat_fw = uhat_fw*ls_scale
                u1 = u1.cpu()
                s1 = s1.cpu()

            loss_terms = self.vae_risk((mu_tx, std_tx), p_t,
                                       (mu_zx, std_zx), p_z,
                                       self.unseen_data.data.x[:, :G].cpu(),
                                       self.unseen_data.data.x[:, G:].cpu(),
                                       uhat.cpu()*lu_scale,
                                       shat.cpu()*ls_scale,
                                       uhat_fw, shat_fw,
                                       u1, s1,
                                       None,
                                       to_cpu=True)

        out = {}
        if "uhat" in output:
            if uhat.ndim == 3:
                uhat = torch.sum(uhat*w_hard, 1)
            out["uhat"] = uhat[:, gene_idx].numpy()
        if "shat" in output:
            if shat.ndim == 3:
                shat = torch.sum(shat*w_hard, 1)
            out["shat"] = shat[:, gene_idx].numpy()
        if "t" in output:
            out["t"] = mu_tx.detach().cpu().squeeze().numpy()
            out["std_t"] = std_tx.detach().cpu().squeeze().numpy()
        if "z" in output:
            out["z"] = mu_zx.detach().cpu().numpy()
            out["std_z"] = std_zx.detach().cpu().numpy()
        if "vs" in output:
            out["vs"] = vs[:, gene_idx].detach().cpu().numpy()
        if "vu" in output:
            out["vu"] = vu[:, gene_idx].detach().cpu().numpy()
        if save_uhat_fw:
            out["uhat_fw"] = uhat_fw[:, gene_idx].detach().cpu().numpy()
        if save_shat_fw:
            out["shat_fw"] = shat_fw[:, gene_idx].detach().cpu().numpy()
        del (mu_tx, std_tx,
             mu_zx, std_zx,
             uhat, shat,
             uhat_fw, shat_fw,
             vu, vs,
             vu_fw, vs_fw)

        return out, [-loss_terms[i].cpu().item() for i in range(len(loss_terms))]

    def _update_x0_inductive(self):
        logger.info("*********           Estimating initial conditions          *********")
        G = self.unseen_data.G
        out, _ = self.pred_all(["uhat", "shat", "t", "z"], mode="all")
        out_query, _ = self.pred_inductive(["uhat", "shat", "t", "z"])
        t = np.concatenate([out["t"], out_query["t"]])
        z = np.concatenate([out["z"], out_query["z"]], 0)
        uhat = np.concatenate([out["uhat"], out_query["uhat"]], 0)
        shat = np.concatenate([out["shat"], out_query["shat"]], 0)
        t_query = out_query["t"]
        z_query = out_query["z"]
        del out, out_query
        
        # Clip the time to avoid outliers
        t = np.clip(t, 0, np.quantile(t, 0.99))
        dt = (self.config["dt"][0]*(t.max()-t.min()), self.config["dt"][1]*(t.max()-t.min()))
        u1, s1, t1 = None, None, None
        # Compute initial conditions of cells without a valid pool of neighbors
        with torch.no_grad():
            init_mask = (t_query <= np.quantile(t_query, 0.01))
            u0_init = np.mean(self.unseen_data.data.x[init_mask][:, :G].detach().cpu().numpy(), 0)
            s0_init = np.mean(self.unseen_data.data.x[init_mask][:, G:].detach().cpu().numpy(), 0)
        xy = self.graph_data.xy.detach().cpu().numpy()
        xy_query = self.unseen_data.xy.detach().cpu().numpy()
        xy = np.concatenate([xy, xy_query], 0)

        x0_index = knnx0_index_parallel(
            t,
            z,
            xy,
            t_query,
            z_query,
            xy_query,
            dt,
            self.config["n_neighbors"],
            self.epsilon_ball,
            hist_eq=True
        )
        u0, s0, xy0, t0 = get_x0(
            uhat,
            shat,
            xy,
            t,
            dt,
            x0_index,
            u0_init,
            s0_init
        )
        if self.config["vel_continuity_loss"]:
            x1_index = knnx0_index_parallel(
                t,
                z,
                xy,
                t_query,
                z_query,
                xy_query,
                dt,
                self.config["n_neighbors"],
                self.epsilon_ball,
                forward=True,
                hist_eq=True
            )
            u1, s1, xy1, t1 = get_x0(
                uhat,
                shat,
                xy,
                t,
                dt,
                x1_index,
                None,
                None,
                forward=True
            )

        self.unseen_data.t = torch.tensor(t_query.reshape(-1, 1), dtype=torch.float32, device=self.device, requires_grad=False)
        self.unseen_data.z = torch.tensor(z_query, dtype=torch.float32, device=self.device, requires_grad=False)
        self.unseen_data.u0 = torch.tensor(u0, dtype=torch.float32, device=self.device, requires_grad=False)
        self.unseen_data.s0 = torch.tensor(s0, dtype=torch.float32, device=self.device, requires_grad=False)
        self.unseen_data.xy0 = torch.tensor(xy0, dtype=torch.float32, device=self.device, requires_grad=True)
        self.unseen_data.t0 = torch.tensor(t0.reshape(-1, 1), dtype=torch.float32, device=self.device, requires_grad=False)

        if self.config['vel_continuity_loss']:
            self.unseen_data.u1 = torch.tensor(u1, dtype=torch.float32, device=self.device, requires_grad=False)
            self.unseen_data.s1 = torch.tensor(s1, dtype=torch.float32, device=self.device, requires_grad=False)
            self.unseen_data.xy1 = torch.tensor(xy1, dtype=torch.float32, device=self.device, requires_grad=False)
            self.unseen_data.t1 = torch.tensor(t1.reshape(-1, 1), dtype=torch.float32, device=self.device, requires_grad=False)

    def _load_unseen_data(self, adata, graph, spatial_key, cluster_key, us_keys=None, edge_attr=None):
        """Retreive data from an unseen adata for inductive evaluation.

        Args:
            adata (AnnData): Annotated data matrix.
            spatial_key (str): Key in adata.obsm storing the spatial coordinates.
            us_keys (list): Keys in adata.layers storing the unspliced and spliced count matrices.
            embed (str): Low dimensional embedding in adata.obsm.
                The actual key storing the embedding should be f'X_{embed}'

        Returns:
            bool : Whether the data is successfully loaded.
        """
        if self.is_discrete:
            U, S = np.array(adata.layers['unspliced'].todense()), np.array(adata.layers['spliced'].todense())
            X = np.concatenate((U, S), 1).astype(int)
        elif us_keys is not None:
            U, S = adata.layers[us_keys[0]], adata.layers[us_keys[1]]
            if isinstance(U, csr_matrix) or isinstance(U, csc_matrix) or isinstance(U, coo_matrix):
                U, S = U.A, S.A
            X = np.concatenate((U, S), 1)
        else:
            X = np.concatenate((adata.layers['Mu'], adata.layers['Ms']), 1).astype(float)
        
        print("*********           Creating an unseen Graph Dataset          *********")
        cell_labels_raw = (adata.obs[cluster_key].to_numpy() if cluster_key in adata.obs
                           else np.array(['Unknown' for i in range(adata.n_obs)]))
        # Encode the labels
        cell_types_raw = np.unique(cell_labels_raw)
        label_dic, label_dic_rev = encode_type(cell_types_raw)
        n_type = len(cell_types_raw)
        cell_labels = np.array([label_dic[x] for x in cell_labels_raw])

        self.unseen_data = SCGraphData(X,
                                       cell_labels,
                                       graph,
                                       adata.obsm[spatial_key],
                                       None,
                                       None,
                                       None,
                                       self.device,
                                       edge_attr,
                                       None,
                                       self.config['enable_edge_weight'],
                                       self.config['normalize_pos'])
        print("*********                      Finished.                      *********")
        self.lu_scale_unseen = (
            torch.tensor(np.log(adata.obs['library_scale_u'].to_numpy()), device=self.device).unsqueeze(-1).float()
            if self.is_discrete else torch.zeros(adata.n_obs, 1, device=self.device).float()
            )
        self.ls_scale_unseen = (
            torch.tensor(np.log(adata.obs['library_scale_s'].to_numpy()), device=self.device).unsqueeze(-1).float()
            if self.is_discrete else torch.zeros(adata.n_obs, 1, device=self.device).float()
            )

    def _get_time_prior_unseen(self, adata, tprior=None):
        print("*********           Setting time prior for test data          *********")
        _p_t = self.p_t
        self.get_prior(adata, self.tmax, tprior)
        temp = self.p_t
        self.p_t = _p_t
        self.p_t_unseen = temp

    def _copy_var(self, adata, adata_unseen, key):
        if self.enable_cvae:
            if self.is_full_vb:
                adata_unseen.varm[f"{key}_logmu_alpha"] = adata.varm[f"{key}_logmu_alpha"]
                adata_unseen.varm[f"{key}_logmu_beta"] = adata.varm[f"{key}_logmu_beta"]
                adata_unseen.varm[f"{key}_logmu_gamma"] = adata.varm[f"{key}_logmu_gamma"]
                adata_unseen.varm[f"{key}_logstd_alpha"] = adata.varm[f"{key}_logstd_alpha"]
                adata_unseen.varm[f"{key}_logstd_beta"] = adata.varm[f"{key}_logstd_beta"]
                adata_unseen.varm[f"{key}_logstd_gamma"] = adata.varm[f"{key}_logstd_gamma"]
            else:
                adata_unseen.varm[f"{key}_alpha"] = adata.varm[f"{key}_alpha"]
                adata_unseen.varm[f"{key}_beta"] = adata.varm[f"{key}_beta"]
                adata_unseen.varm[f"{key}_gamma"] = adata.varm[f"{key}_gamma"]
            adata_unseen.varm[f"{key}_ton"] = adata.varm[f"{key}_ton"]
            adata_unseen.varm[f"{key}_scaling_u"] = adata.varm[f"{key}_scaling_u"]
            adata_unseen.varm[f"{key}_scaling_s"] = adata.varm[f"{key}_scaling_s"]
            adata_unseen.var[f"{key}_sigma_u"] = adata.var[f"{key}_sigma_u"].to_numpy()
            adata_unseen.var[f"{key}_sigma_s"] = adata.var[f"{key}_sigma_s"].to_numpy()
        else:
            if self.is_full_vb:
                adata_unseen.var[f"{key}_logmu_alpha"] = adata.var[f"{key}_logmu_alpha"].to_numpy()
                adata_unseen.var[f"{key}_logmu_beta"] = adata.var[f"{key}_logmu_beta"].to_numpy()
                adata_unseen.var[f"{key}_logmu_gamma"] = adata.var[f"{key}_logmu_gamma"].to_numpy()
                adata_unseen.var[f"{key}_logstd_alpha"] = adata.var[f"{key}_logstd_alpha"].to_numpy()
                adata_unseen.var[f"{key}_logstd_beta"] = adata.var[f"{key}_logstd_beta"].to_numpy()
                adata_unseen.var[f"{key}_logstd_gamma"] = adata.var[f"{key}_logstd_gamma"].to_numpy()
            else:
                adata_unseen.var[f"{key}_alpha"] = adata.var[f"{key}_alpha"].to_numpy()
                adata_unseen.var[f"{key}_beta"] = adata.var[f"{key}_beta"].to_numpy()
                adata_unseen.var[f"{key}_gamma"] = adata.var[f"{key}_gamma"].to_numpy()

            adata_unseen.var[f"{key}_ton"] = adata.var[f"{key}_ton"].to_numpy()
            adata_unseen.var[f"{key}_scaling_u"] = adata.var[f"{key}_scaling_u"].to_numpy()
            adata_unseen.var[f"{key}_scaling_s"] = adata.var[f"{key}_scaling_s"].to_numpy()
            adata_unseen.var[f"{key}_sigma_u"] = adata.var[f"{key}_sigma_u"].to_numpy()
            adata_unseen.var[f"{key}_sigma_s"] = adata.var[f"{key}_sigma_s"].to_numpy()
        adata_unseen.varm[f"{key}_mode"] = F.softmax(self.decoder.logit_pw, 1).detach().cpu().numpy()

    def test_inductive(self,
                       adata,
                       adata_unseen,
                       key,
                       graph_key,
                       spatial_key,
                       cluster_key,
                       tprior=None,
                       us_keys=None,
                       edge_attr=None,
                       batch_key=None):
        """Evaluate the model inductively on unseen graph data.

        Warning:
            This function is considered unstable and may change in future releases.
        
        Args:
            adata (:class:`anndata.AnnData`): AnnData object.
            adata_unseen (:class:`anndata.AnnData`): AnnData object for unseen data.
            key (str): Key for storing pretrained model parameters in AnnData
            graph_key (str): Key for extracting spatial graph.
            spatial_key (str): Key for extracting spatial coordinates.
            cluster_key (str): Key for cell type labels.
            tprior (list): Time prior.
            us_keys (list): Keys for unspliced and spliced count matrices.
            edge_attr (str): Key for edge attributes.
            batch_key (str): Key for batch information.
        """
        self._load_unseen_data(adata_unseen,
                               adata_unseen.obsp[graph_key],
                               spatial_key,
                               cluster_key,
                               us_keys,
                               edge_attr)
        self._get_time_prior_unseen(adata_unseen, tprior)
        self._update_x0_inductive()
        out, _ = self.pred_inductive(["uhat", "shat", "t", "z", "std_t", "std_z"])
        uhat, shat, t, std_t, z, std_z = out["uhat"], out["shat"], out["t"], out["std_t"], out["z"], out["std_z"]
        with torch.no_grad():
            condition = (torch.zeros(adata_unseen.n_obs, self.n_batch).float()
                         if self.enable_cvae else None)
            if self.spatial_decoder:
                xy_hat = self.pred_xy(self.unseen_data.t,
                                      self.unseen_data.z,
                                      self.unseen_data.t0,
                                      self.unseen_data.xy0,
                                      self.unseen_data.data.adj_t,
                                      self.unseen_data.edge_weight,
                                      condition,
                                      None,
                                      mode='eval')
            z_in = self.unseen_data.z
            if self.enable_cvae:
                z_in = torch.cat((z_in, condition), 1)
            rho = (self.decoder.net_rho2(z_in, self.unseen_data.data.adj_t, None)
                   if isinstance(self.decoder.net_rho2, GraphDecoder) else
                   self.decoder.net_rho2(torch.tensor(z_in, device=self.device).float()))
        self._copy_var(adata, adata_unseen, key)
        adata_unseen.layers[f"{key}_rho"] = rho.cpu().numpy()
        adata_unseen.obs[f"{key}_time"] = t
        adata_unseen.obs[f"{key}_std_t"] = std_t
        adata_unseen.obsm[f"{key}_z"] = z
        adata_unseen.obsm[f"{key}_std_z"] = std_z
        if self.spatial_decoder:
            adata_unseen.obsm[f"X_{key}_xy"] = xy_hat.detach().cpu().numpy()
        adata_unseen.obsm[f"X_{key}_xy0"] = self.unseen_data.xy0.detach().cpu().numpy()
        adata_unseen.layers[f"{key}_uhat"] = uhat
        adata_unseen.layers[f"{key}_shat"] = shat
        
        # Attention score
        enc_att = self.get_enc_att(self.unseen_data.data.x, self.unseen_data.data.adj_t)
        z_ts = torch.tensor(z, device=self.device)
        dec_att = self.get_dec_att(z_ts, self.unseen_data.data.adj_t)
        
        if enc_att is not None:
            for i in range(len(enc_att)):
                adata_unseen.obsp[f"{key}_enc_att_{i}"] = enc_att[i]
        if dec_att is not None:
            for i in range(len(dec_att)):
                adata_unseen.obsp[f"{key}_dec_att_{i}"] = dec_att[i]
        del z_ts, enc_att, dec_att

        adata_unseen.obs[f"{key}_t0"] = self.unseen_data.t0.detach().cpu().squeeze().numpy()
        adata_unseen.layers[f"{key}_u0"] = self.unseen_data.u0.detach().cpu().numpy()
        adata_unseen.layers[f"{key}_s0"] = self.unseen_data.s0.detach().cpu().numpy()
        if self.config["vel_continuity_loss"]:
            adata_unseen.obs[f"{key}_t1"] = self.t1.detach().cpu().squeeze().numpy()
            adata_unseen.layers[f"{key}_u1"] = self.unseen_data.u1.detach().cpu().numpy()
            adata_unseen.layers[f"{key}_s1"] = self.unseen_data.s1.detach().cpu().numpy()
        if self.spatial_decoder:
            adata_unseen.obsm[f"{key}_velocity_{key}_xy"] = self.xy_velocity(self.unseen_data, condition)
        del condition, rho, out

        rna_velocity_vae(adata_unseen,
                         key,
                         batch_key=batch_key,
                         use_raw=False,
                         use_scv_genes=False,
                         full_vb=self.is_full_vb)

    def update_std_noise(self):
        """Update the standard deviation of Gaussian noise."""
        G = self.graph_data.G
        out, _ = self.pred_all(self.cell_labels,
                               mode='train',
                               output=["uhat", "shat", "xy"],
                               gene_idx=np.array(range(G)))
        std_u = (out["uhat"]-self.graph_data.data.x[:, :G].detach().cpu().numpy()[self.train_idx]).std(0)
        std_s = (out["shat"]-self.graph_data.data.x[:, G:].detach().cpu().numpy()[self.train_idx]).std(0)
        self.decoder.sigma_u = nn.Parameter(torch.tensor(np.log(std_u+1e-16),
                                            dtype=torch.float,
                                            device=self.device))
        self.decoder.sigma_s = nn.Parameter(torch.tensor(np.log(std_s+1e-16),
                                            dtype=torch.float,
                                            device=self.device))
        
        return

    def _scale_data_for_enc(self, data_in):
        data_in_scale = data_in
        G = self.decoder.n_gene
        condition = (F.one_hot(self.graph_data.batch, self.n_batch).float()
                     if self.enable_cvae else None)
        # optional data scaling_u
        if self.config["scale_gene_encoder"]:
            scaling_u = self.decoder.get_param_1d('scaling_u', condition, sample=False, detach=False)
            scaling_s = self.decoder.get_param_1d('scaling_s', condition, sample=False, detach=False)
            data_in_scale = torch.cat((data_in_scale[:, :G]/scaling_u,
                                       data_in_scale[:, G:]/scaling_s), 1)
        if self.config["scale_cell_encoder"]:
            lu_scale = self.lu_scale.exp()
            ls_scale = self.ls_scale.exp()
            data_in_scale = torch.cat((data_in_scale[:, :, :G]/lu_scale,
                                       data_in_scale[:, :, G:]/ls_scale), 1)
        if self.config["log1p"]:
            data_in_scale = torch.log1p(data_in_scale)
        return data_in_scale

    def get_enc_att(self, data_in, edge_index):
        """Retreive the encoder attention score.
        
        Args:
            data_in (torch.Tensor): Input data (cell by gene, with unspliced and spliced concatenated at dim=1).
            edge_index (torch.Tensor): Edge index.
        
        Returns:
            numpy.array : Attention score of shape (num_cell, num_cell, num_att_head).
        """
        self.set_mode('eval')
        data_in_scale = self._scale_data_for_enc(data_in)
        
        gatconv = self.encoder.conv1
        if not isinstance(gatconv, GATConv):
            logger.info("Skipping encoder attention score computation.")
            return None
        with torch.no_grad():
            _, att = gatconv(data_in_scale, edge_index, return_attention_weights=True)
        try:
            cum_num_col, row, val = att.cpu().csr()
            cum_num_col = cum_num_col.detach().numpy()
            row = row.detach().numpy()
            val = val.detach().numpy()
        except AttributeError:
            # Notice that the row and column are flipped in our convention
            cum_num_col = att[0].cpu().crow_indices().numpy()
            row = att[0].cpu().col_indices().numpy()
            val = att[0].cpu().values().numpy()

        return dge2array(cum_num_col, row, val)

    def get_dec_att(self, data_in, edge_index):
        """Retreive the decoder attention score.
        
        Args:
            data_in (torch.Tensor): Input data (cell by gene, with unspliced and spliced concatenated at dim=1).
            edge_index (torch.Tensor): Edge index.
            edge_weight (torch.Tensor): Edge weight.

        Returns:
            numpy.array : Attention score of shape (num_cell, num_cell, num_att_head).
        """
        self.set_mode('eval')
        gatconv = self.decoder.net_rho2.conv1
        if not isinstance(gatconv, GATConv):
            logger.info("Skipping decoder attention score computation.")
            return None
        with torch.no_grad():
            if self.enable_cvae:
                condition = F.one_hot(self.graph_data.batch, self.n_batch).float()
                _, att = gatconv(torch.cat([data_in, condition], 1), edge_index, return_attention_weights=True)
            else:
                _, att = gatconv(data_in, edge_index, return_attention_weights=True)
        try:
            cum_num_col, row, val = att.cpu().csr()
            cum_num_col = cum_num_col.detach().numpy()
            row = row.detach().numpy()
            val = val.detach().numpy()
        except AttributeError:
            # Notice that the row and column are flipped in our convention
            cum_num_col = att[0].cpu().crow_indices().numpy()
            row = att[0].cpu().col_indices().numpy()
            val = att[0].cpu().values().numpy()

        return dge2array(cum_num_col, row, val)
    
    def get_enc_att_all_pairs(self, data_in, n_nodes, batch_size=64):
        """Retreive the encoder attention score for all pairs of nodes.
        
        Args:
            data_in (torch.Tensor): Input data (cell by gene, with unspliced and spliced concatenated at dim=1).
            n_nodes (int): Number of nodes.
            batch_size (int): Batch size.
        
        Returns:
            numpy.array : Attention score of shape (num_cell, num_cell, num_att_head).
                The first dimension corresponds to source. Second dimension is target.
        """
        self.set_mode('eval')
        data_in_scale = self._scale_data_for_enc(data_in)

        gatconv = self.encoder.conv1
        if not isinstance(gatconv, GATConv):
            logger.info("Skipping encoder attention score computation.")
            return None
        att_score = []
        n_batch = n_nodes // batch_size

        with torch.no_grad():
            # Each batch is the target (receiver) of the attention
            for i in range(n_batch):
                edge_index = torch.cartesian_prod(torch.range(start=0, end=n_nodes-1, dtype=int),
                                                  torch.range(start=i*batch_size, end=(i+1)*batch_size-1, dtype=int)).T.to(self.device)
                _, att = gatconv(data_in_scale,
                                 edge_index, None, return_attention_weights=True)
                col_idx = att[0][1].cpu().numpy()
                mask = (col_idx >= i*batch_size) & (col_idx < (i+1)*batch_size)
                n_head = att[1].shape[1]
                att_score.append(att[1].cpu().numpy()[mask].reshape(n_nodes, batch_size, n_head))
            if n_batch * batch_size < n_nodes:
                edge_index = torch.cartesian_prod(torch.range(start=0, end=n_nodes-1, dtype=int),
                                                  torch.range(start=n_batch*batch_size, end=n_nodes-1, dtype=int)).T.to(self.device)
                _, att = gatconv(data_in_scale,
                                 edge_index, None, return_attention_weights=True)
                col_idx = att[0][1].cpu().numpy()
                mask = col_idx >= n_batch*batch_size
                n_head = att[1].shape[1]
                att_score.append(att[1].cpu().numpy()[mask].reshape(n_nodes, n_nodes-n_batch*batch_size, n_head))
        return np.concatenate(att_score, 1)
    
    def get_dec_att_all_pairs(self, data_in, n_nodes, batch_size=64):
        """Retreive the decoder attention score for all pairs of nodes.
        
        Args:
            data_in (torch.Tensor): Input data (cell by gene, with unspliced and spliced concatenated at dim=1).
            n_nodes (int): Number of nodes.
            batch_size (int): Batch size.
        
        Returns:
            numpy.array : Attention score of shape (num_cell, num_cell, num_att_head).
                The first dimension corresponds to source. Second dimension is target.
        """
        self.set_mode('eval')
        gatconv = self.decoder.net_rho2.conv1
        if not isinstance(gatconv, GATConv):
            logger.info("Skipping decoder attention score computation.")
            return None
        att_score = []
        n_batch = n_nodes // batch_size
        with torch.no_grad():
            if self.enable_cvae:
                for i in range(n_batch):
                    edge_index = torch.cartesian_prod(torch.range(start=0, end=n_nodes-1, dtype=int),
                                                      torch.range(start=i*batch_size, end=(i+1)*batch_size-1, dtype=int)).T.to(self.device)
                    condition = F.one_hot(self.graph_data.batch, self.n_batch).float()
                    _, att = gatconv(torch.cat([data_in, condition], 1),
                                     edge_index,
                                     None,
                                     return_attention_weights=True)
                    col_idx = att[0][1].cpu().numpy()
                    mask = (col_idx >= i*batch_size) & (col_idx < (i+1)*batch_size)
                    n_head = att[1].shape[1]
                    att_score.append(att[1].cpu().numpy()[mask].reshape(n_nodes, batch_size, n_head))
                if n_batch*batch_size < n_nodes:
                    edge_index = torch.cartesian_prod(torch.range(start=0, end=n_nodes-1, dtype=int),
                                                      torch.range(start=n_batch*batch_size, end=n_nodes-1, dtype=int)).T.to(self.device)
                    _, att = gatconv(torch.cat([data_in, condition], 1),
                                     edge_index,
                                     None,
                                     return_attention_weights=True)
                    col_idx = att[0][1].cpu().numpy()
                    mask = col_idx >= n_batch*batch_size
                    n_head = att[1].shape[1]
                    att_score.append(att[1].cpu().numpy()[mask].reshape(n_nodes, n_nodes-n_batch*batch_size, n_head))
            else:
                for i in range(n_batch):
                    edge_index = torch.cartesian_prod(torch.range(start=0, end=n_nodes-1, dtype=int),
                                                      torch.range(start=i*batch_size, end=(i+1)*batch_size-1, dtype=int)).T.to(self.device)
                    _, att = gatconv(data_in,
                                     edge_index,
                                     None,
                                     return_attention_weights=True)
                    col_idx = att[0][1].cpu().numpy()
                    mask = (col_idx >= i*batch_size) & (col_idx < (i+1)*batch_size)
                    n_head = att[1].shape[1]
                    att_score.append(att[1].cpu().numpy()[mask].reshape(n_nodes, batch_size, n_head))
                if n_batch*batch_size < n_nodes:
                    edge_index = torch.cartesian_prod(torch.range(start=0, end=n_nodes-1, dtype=int),
                                                      torch.range(start=n_batch*batch_size, end=n_nodes-1, dtype=int)).T.to(self.device)
                    _, att = gatconv(data_in,
                                     edge_index,
                                     None,
                                     return_attention_weights=True)
                    col_idx = att[0][1].cpu().numpy()
                    mask = col_idx >= n_batch*batch_size
                    n_head = att[1].shape[1]
                    att_score.append(att[1].cpu().numpy()[mask].reshape(n_nodes, n_nodes-n_batch*batch_size, n_head))
        return np.concatenate(att_score, 1)

    def _forward_to_rho(self):
        self.set_mode('eval')
        G = self.decoder.n_gene
        self.graph_data.data.x.requires_grad = True
        condition = (F.one_hot(self.graph_data.batch, self.n_batch).float()
                     if self.enable_cvae else None)
        scaling_u = self.decoder.get_param_1d('scaling_u', condition, sample=False, detach=False)
        scaling_s = self.decoder.get_param_1d('scaling_s', condition, sample=False, detach=False)
        data_in_scale = torch.cat((self.graph_data.data.x[:, :G]/scaling_u,
                                  self.graph_data.data.x[:, G:]/scaling_s), 1)

        mu_t, std_t, mu_z, std_z = self.encoder.forward(data_in_scale,
                                                        self.graph_data.data.adj_t,
                                                        self.graph_data.edge_weight,
                                                        condition)

        t = self.sample(mu_t, std_t)
        z = self.sample(mu_z, std_z)
        graph_conv = self.decoder.net_rho2.conv1
        if isinstance(graph_conv, GCNConv):
            h = graph_conv(data_in_scale, self.graph_data.data.adj_t, self.graph_data.edge_weight)
        else:
            h = graph_conv(data_in_scale, self.graph_data.data.adj_t)
        h = self.decoder.net_rho2.act(h)
        if condition is not None:
            h = torch.cat((h, condition), 1)
        if not self.decoder.net_rho2.enable_sigmoid:
            rho = self.decoder.net_rho2.fc_out(h)
        else:
            rho = self.decoder.net_rho2.sigm(self.decoder.net_rho2.fc_out(h))
        return rho

    def get_all_gradients(self, adata):
        rho = self._forward_to_rho()
        grad = np.zeros((2*adata.n_vars, adata.n_vars))
        for i in range(adata.n_vars):
            _gd = torch.autograd.grad(rho[:, i],
                                      self.graph_data.data.x,
                                      grad_outputs=torch.ones_like(rho[:, i]),
                                      retain_graph=True,
                                      allow_unused=True)[0]
            grad[:, i] = _gd.detach().cpu().numpy().sum(0)
        return grad[:adata.n_vars], grad[adata.n_vars:]

    def get_gradients(self, adata, source_genes, query_cells, spatial_graph_key, target_genes=None):
        """Computes the gradient of gene transcription factors within some query cells
        with respect to query genes within their neighboring cells.

        Warning:
            This function is considered unstable and may change in future releases.

        Args:
            adata (:class:`anndata.AnnData`): AnnData object.
            source_genes (list[str]): list of gene names.
            query_cells (:class:`numpy.ndarray`): cell indices.
            spatial_graph_key (str): Key for obtaining the spatial graph in adata.obsp.
            target_genes (list[str], optional): list of genes affected by the source genes.
                If not provided, all genes are considered.
                Defaults to None.

        Returns:
            :class:`numpy.ndarray`: query gene by query cell by gene array
        """
        rho = self._forward_to_rho()
        gd = []
        source_gene_idx = np.array([np.where(adata.var_names == x)[0][0] for x in source_genes])
        target_gene_idx = (np.array([np.where(adata.var_names == x)[0][0] for x in target_genes])
                           if target_genes is not None else
                           np.array(range(adata.n_vars)))
        for idx in target_gene_idx:
            gd_gene = []  # query cell by source gene
            # gd_s_gene = []  # query cell by source gene
            for cell in query_cells:
                nbs = np.where(adata.obsp[spatial_graph_key][cell].A.squeeze() > 0)[0]
                _gd = torch.autograd.grad(rho[cell, idx],
                                          self.graph_data.data.x,
                                          retain_graph=True,
                                          allow_unused=True)[0]
                _gd_u = _gd[nbs][:, source_gene_idx].detach().cpu().numpy()
                _gd_s = _gd[nbs][:, source_gene_idx+adata.n_vars].detach().cpu().numpy()
                _gd_us = np.nanmean(_gd_u*(_gd_s/(_gd_u+_gd_s+1e-20)), 0)
                gd_gene.append(_gd_us)
                # gd_s_gene.append(np.quantile(_gd[nbs][:, source_gene_idx+adata.n_vars].detach().cpu().numpy(), 0.5, 0))
            gd.append(np.stack(gd_gene))
            # gd_s.append(np.stack(gd_s_gene))

        return np.stack(gd)

    def get_gradients_lr(self, adata, ligand_genes, receptor_genes, query_cells, spatial_graph_key, target_genes=None):
        """Computes the gradient of gene transcription factors within some query cells
        with respect to ligand-receptor interaction pairs.

        Warning:
            This function is considered unstable and may change in future releases.

        Args:
            adata (:class:`anndata.AnnData`): AnnData object.
            source_genes (list[str]): list of gene names.
            query_cells (:class:`numpy.ndarray`): cell indices.
            spatial_graph_key (str): Key for obtaining the spatial graph in adata.obsp.
            target_genes (list[str], optional): list of genes affected by the source genes.
                If not provided, all genes are considered.
                Defaults to None.

        Returns:
            :class:`numpy.ndarray`: query gene by query cell by gene array
        """
        rho = self._forward_to_rho()
        gd_u, gd_s = [], []
        ligand_gene_idx = np.array([np.where(adata.var_names == x)[0][0] for x in ligand_genes])
        receptor_gene_idx = np.array([np.where(adata.var_names == x)[0][0] for x in receptor_genes])
        target_gene_idx = (np.array([np.where(adata.var_names == x)[0][0] for x in target_genes])
                           if target_genes is not None else
                           np.array(range(adata.n_vars)))
        for idx in target_gene_idx:
            gd_u_gene = []  # query cell by source gene
            gd_s_gene = []  # query cell by source gene
            for cell in query_cells:
                nbs = np.where(adata.obsp[spatial_graph_key][cell].A.squeeze() > 0)[0]
                _gd = torch.autograd.grad(rho[cell, idx],
                                          self.graph_data.data.x,
                                          retain_graph=True,
                                          allow_unused=True)[0]
                gd_u_gene.append(np.quantile(_gd[nbs][:, source_gene_idx].detach().cpu().numpy(), 0.5, 0))
                gd_s_gene.append(np.quantile(_gd[nbs][:, source_gene_idx+adata.n_vars].detach().cpu().numpy(), 0.5, 0))
            gd_u.append(np.stack(gd_u_gene))
            gd_s.append(np.stack(gd_s_gene))

        return np.stack(gd_u), np.stack(gd_s)

    def xy_velocity(self,
                    graph_data,
                    condition=None,
                    delta_t=0.05):
        """Compute the velocity of spatial coordinates using the spatial decoder.
        
        Args:
            graph_data (:class:`SCGraphData`): Graph data.
            condition (:class:`torch.Tensor`, optional): Condition tensor.
            delta_t (float, optional): Time interval.
        
        Returns:
            :class:`numpy.ndarray`: Velocity of spatial coordinates.
        """
        self.set_mode('eval')
        edge_index = graph_data.data.adj_t
        edge_weight = graph_data.edge_weight
        xy_hat = self.decoder._compute_xy(graph_data.t,
                                          graph_data.z,
                                          graph_data.t0,
                                          graph_data.xy0,
                                          edge_index,
                                          edge_weight,
                                          condition)
        # tau = torch.ones((len(xy_hat), 1), dtype=torch.float32, device=self.device)*delta_t
        xy_hat_1 = self.decoder._compute_xy(graph_data.t + delta_t,
                                            graph_data.z,
                                            graph_data.t,
                                            xy_hat,
                                            edge_index,
                                            edge_weight,
                                            condition).detach().cpu().numpy()
        xy_hat = xy_hat.detach().cpu().numpy()
        return (xy_hat_1 - xy_hat) / delta_t

    def spatial_extrapolation(self,
                              adata,
                              cluster_key,
                              time_interval,
                              figure_path=None):
        """Predict the future spatial coordinates of cells.
        
        Warning:
            This function is considered unstable and may change in future releases.

        Args:
            adata (:class:`anndata.AnnData`): AnnData object.
            cluster_key (str): Key for cell type labels.
            time_interval (list[float]): Time intervals. 
            figure_path (str, optional): Path for saving figures.
        """
        cell_labels = adata.obs[cluster_key].to_numpy()
        condition = (F.one_hot(self.graph_data.batch, self.n_batch).float()
                     if self.enable_cvae else None)
        xy_hat = self.pred_xy(self.graph_data.t,
                              self.graph_data.z,
                              self.graph_data.t0,
                              self.graph_data.xy0,
                              self.graph_data.data.adj_t,
                              self.graph_data.edge_weight,
                              condition,
                              None,
                              mode='eval')
        xy_future_multitime = []
        for i, delta_t in enumerate(time_interval):
            xy_future = self.pred_xy(self.graph_data.t + delta_t,
                                     self.graph_data.z,
                                     self.graph_data.t,
                                     xy_hat,
                                     self.graph_data.data.adj_t,
                                     self.graph_data.edge_weight,
                                     condition,
                                     None,
                                     mode='eval').detach().cpu().numpy()
            xy_future_multitime.append(xy_future)
            # Plot the predicted future positions
            save = None
            if figure_path is not None:
                save = f'{figure_path}/xy_ext_{i}.png'
            plot_spatial_extrapolation(xy_hat.detach().cpu().numpy(),
                                       xy_future,
                                       cell_labels,
                                       save=save)
        adata.uns['delta_t'] = time_interval
        adata.obsm['X_x_ext'] = np.stack([xy[:, 0] for xy in xy_future_multitime], axis=1)
        adata.obsm['X_y_ext'] = np.stack([xy[:, 1] for xy in xy_future_multitime], axis=1)
        return

    def save_anndata(self, adata, key, file_path, file_name=None):
        """Save the ODE parameters and cell time to the anndata object and write it to disk.

        Args:
            adata (:class:`anndata.AnnData`):
            key (str):
                Used to store all parameters of the model.
            file_path (str):
                Saving path.
            file_name (str, optional):
                If set to a string ending with .h5ad, the updated anndata object will be written to disk.
        """
        self.set_mode('eval')
        os.makedirs(file_path, exist_ok=True)
        if self.enable_cvae:
            if self.is_full_vb:
                adata.varm[f"{key}_logmu_alpha"] = self.decoder.alpha[:, 0, :].detach().cpu().numpy().T
                adata.varm[f"{key}_logmu_beta"] = self.decoder.beta[:, 0, :].detach().cpu().numpy().T
                adata.varm[f"{key}_logmu_gamma"] = self.decoder.gamma[:, 0, :].detach().cpu().numpy().T
                adata.varm[f"{key}_logstd_alpha"] = self.decoder.alpha[:, 1, :].detach().cpu().exp().numpy().T
                adata.varm[f"{key}_logstd_beta"] = self.decoder.beta[:, 1, :].detach().cpu().exp().numpy().T
                adata.varm[f"{key}_logstd_gamma"] = self.decoder.gamma[:, 1, :].detach().cpu().exp().numpy().T
            else:
                adata.varm[f"{key}_alpha"] = np.exp(self.decoder.alpha.detach().cpu().numpy()).T
                adata.varm[f"{key}_beta"] = np.exp(self.decoder.beta.detach().cpu().numpy()).T
                adata.varm[f"{key}_gamma"] = np.exp(self.decoder.gamma.detach().cpu().numpy()).T
            adata.varm[f"{key}_ton"] = self.decoder.ton.exp().detach().cpu().numpy().T
            adata.varm[f"{key}_scaling_u"] = np.exp(self.decoder.scaling_u.detach().cpu().numpy()).T
            adata.varm[f"{key}_scaling_s"] = np.exp(self.decoder.scaling_s.detach().cpu().numpy()).T
            adata.var[f"{key}_sigma_u"] = np.exp(self.decoder.sigma_u.detach().cpu().numpy()).T
            adata.var[f"{key}_sigma_s"] = np.exp(self.decoder.sigma_s.detach().cpu().numpy()).T
        else:
            if self.is_full_vb:
                adata.var[f"{key}_logmu_alpha"] = self.decoder.alpha[0].detach().cpu().numpy()
                adata.var[f"{key}_logmu_beta"] = self.decoder.beta[0].detach().cpu().numpy()
                adata.var[f"{key}_logmu_gamma"] = self.decoder.gamma[0].detach().cpu().numpy()
                adata.var[f"{key}_logstd_alpha"] = self.decoder.alpha[1].detach().cpu().exp().numpy()
                adata.var[f"{key}_logstd_beta"] = self.decoder.beta[1].detach().cpu().exp().numpy()
                adata.var[f"{key}_logstd_gamma"] = self.decoder.gamma[1].detach().cpu().exp().numpy()
            else:
                adata.var[f"{key}_alpha"] = np.exp(self.decoder.alpha.detach().cpu().numpy())
                adata.var[f"{key}_beta"] = np.exp(self.decoder.beta.detach().cpu().numpy())
                adata.var[f"{key}_gamma"] = np.exp(self.decoder.gamma.detach().cpu().numpy())

            adata.var[f"{key}_ton"] = self.decoder.ton.exp().detach().cpu().numpy()
            adata.var[f"{key}_scaling_u"] = np.exp(self.decoder.scaling_u.detach().cpu().numpy())
            adata.var[f"{key}_scaling_s"] = np.exp(self.decoder.scaling_s.detach().cpu().numpy())
            adata.var[f"{key}_sigma_u"] = np.exp(self.decoder.sigma_u.detach().cpu().numpy())
            adata.var[f"{key}_sigma_s"] = np.exp(self.decoder.sigma_s.detach().cpu().numpy())
        adata.varm[f"{key}_mode"] = F.softmax(self.decoder.logit_pw, 1).detach().cpu().numpy()

        out, elbo = self.pred_all(self.cell_labels,
                                  "both",
                                  gene_idx=np.array(range(adata.n_vars)))
        Uhat, Shat, t, std_t, z, std_z = out["uhat"], out["shat"], out["t"], out["std_t"], out["z"], out["std_z"]
        with torch.no_grad():
            condition = (F.one_hot(self.graph_data.batch, self.n_batch).float()
                         if self.enable_cvae else None)
            if self.spatial_decoder:
                xy_hat = self.pred_xy(self.graph_data.t,
                                      self.graph_data.z,
                                      self.graph_data.t0,
                                      self.graph_data.xy0,
                                      self.graph_data.data.adj_t,
                                      self.graph_data.edge_weight,
                                      condition,
                                      None,
                                      mode='eval')

            z_in = self.graph_data.z
            if self.enable_cvae:
                z_in = torch.cat((z_in, condition), 1)
            rho = (self.decoder.net_rho2(z_in,
                                         self.graph_data.data.adj_t,
                                         None)
                   if isinstance(self.decoder.net_rho2, GraphDecoder) else
                   self.decoder.net_rho2(torch.tensor(z_in, device=self.device).float()))
        adata.layers[f"{key}_rho"] = rho.cpu().numpy()
        adata.obs[f"{key}_time"] = t
        adata.obs[f"{key}_std_t"] = std_t
        adata.obsm[f"{key}_z"] = z
        adata.obsm[f"{key}_std_z"] = std_z
        adata.layers[f"{key}_uhat"] = Uhat
        adata.layers[f"{key}_shat"] = Shat
        
        # Attention score
        enc_att = self.get_enc_att(self.graph_data.data.x, self.graph_data.data.adj_t)
        z_ts = torch.tensor(z, device=self.device)
        dec_att = self.get_dec_att(z_ts, self.graph_data.data.adj_t)
        
        if enc_att is not None:
            for i in range(len(enc_att)):
                adata.obsp[f"{key}_enc_att_{i}"] = enc_att[i]
        if dec_att is not None:
            for i in range(len(dec_att)):
                adata.obsp[f"{key}_dec_att_{i}"] = dec_att[i]
        del z_ts, enc_att, dec_att

        adata.obs[f"{key}_t0"] = self.graph_data.t0.detach().cpu().squeeze().numpy()
        adata.layers[f"{key}_u0"] = self.graph_data.u0.detach().cpu().numpy()
        adata.layers[f"{key}_s0"] = self.graph_data.s0.detach().cpu().numpy()
        if self.config["vel_continuity_loss"]:
            adata.obs[f"{key}_t1"] = self.t1.detach().cpu().squeeze().numpy()
            adata.layers[f"{key}_u1"] = self.graph_data.u1.detach().cpu().numpy()
            adata.layers[f"{key}_s1"] = self.graph_data.s1.detach().cpu().numpy()

        adata.uns[f"{key}_train_idx"] = self.train_idx
        adata.uns[f"{key}_validation_idx"] = self.validation_idx
        if self.batch_ is not None:
            adata.obs['batch_int'] = self.batch_
        if self.test_idx is not None:
            adata.uns[f"{key}_test_idx"] = self.test_idx
        adata.uns[f"{key}_run_time"] = self.timer

        if self.spatial_decoder:
            adata.obsm[f"X_{key}_xy0"] = self.graph_data.xy0.detach().cpu().numpy()
            adata.obsm[f"X_{key}_xy"] = xy_hat.detach().cpu().numpy()
            adata.obsm[f"{key}_velocity_{key}_xy"] = self.xy_velocity(self.graph_data, condition)
        del condition, rho, out

        rna_velocity_vae(adata,
                         key,
                         batch_key='batch_int' if self.enable_cvae else None,
                         use_raw=False,
                         use_scv_genes=False,
                         full_vb=self.is_full_vb)
        if file_name is not None:
            adata.write_h5ad(f"{file_path}/{file_name}")
