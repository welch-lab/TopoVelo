import numpy as np
import scipy.stats as stats
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions.negative_binomial import NegativeBinomial
from torch.distributions.poisson import Poisson

from torch_geometric.nn import GCNConv

import time
from ..plotting import plot_sig, plot_sig_, plot_time, plot_vel
from ..plotting import plot_train_loss, plot_test_loss

from .model_util import hist_equal, init_params, get_ts_global, reinit_params
from .model_util import convert_time, get_gene_index
from .model_util import pred_su, ode_numpy, knnx0, knnx0_bin
from .model_util import elbo_collapsed_categorical
from .model_util import assign_gene_mode, find_dirichlet_param
from .model_util import get_cell_scale, get_dispersion
from .model_util import get_neigh_index

from .transition_graph import encode_type
from .training_data import SCGraphData
from .vanilla_vae import VanillaVAE, kl_gaussian
from .velocity import rna_velocity_vae
P_MAX = 1e30
GRAD_MAX = 1e7
##############################################################
# VAE
##############################################################


class encoder(nn.Module):
    """Encoder class for the graph VAE model
    """
    def __init__(self,
                 Cin,
                 dim_z,
                 dim_cond=0,
                 N1=500,
                 N2=250,
                 checkpoint=None):
        super(encoder, self).__init__()
        self.conv1 = GCNConv(Cin, N1)
        self.conv2 = GCNConv(N1, N2)

        self.fc_mu_t = nn.Linear(N2+dim_cond, 1).float()
        self.spt1 = nn.Softplus().float()
        self.fc_std_t = nn.Linear(N2+dim_cond, 1).float()
        self.spt2 = nn.Softplus().float()

        self.fc_mu_z = nn.Linear(N2+dim_cond, dim_z).float()
        self.fc_std_z = nn.Linear(N2+dim_cond, dim_z).float()
        self.spt3 = nn.Softplus().float()

        if checkpoint is not None:
            self.load_state_dict(torch.load(checkpoint, map_location=device))
        else:
            self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv1.lin.weight)
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.xavier_uniform_(self.conv2.lin.weight)
        nn.init.constant_(self.conv2.bias, 0)
        for m in [self.fc_mu_t,
                  self.fc_std_t,
                  self.fc_mu_z,
                  self.fc_std_z]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, data_in, edge_index, edge_weight=None, condition=None):

        h = self.conv1(data_in, edge_index, edge_weight)
        h = self.conv2(h, edge_index, edge_weight)
        if condition is not None:
            h = torch.cat((h, condition), 1)
        mu_tx, std_tx = self.spt1(self.fc_mu_t(h)), self.spt2(self.fc_std_t(h))
        mu_zx, std_zx = self.fc_mu_z(h), self.spt3(self.fc_std_z(h))
        return mu_tx, std_tx, mu_zx, std_zx


class decoder(nn.Module):
    def __init__(self,
                 adata,
                 tmax,
                 train_idx,
                 dim_z,
                 full_vb=False,
                 discrete=False,
                 dim_cond=0,
                 N1=250,
                 N2=500,
                 p=98,
                 init_ton_zero=False,
                 filter_gene=False,
                 device=torch.device('cpu'),
                 init_method="steady",
                 init_key=None,
                 checkpoint=None,
                 **kwargs):
        super(decoder, self).__init__()
        G = adata.n_vars
        self.tmax = tmax
        self.is_full_vb = full_vb
        self.is_discrete = discrete

        if checkpoint is None:
            # Get dispersion and library size factor for the discrete model
            if discrete:
                U, S = adata.layers['unspliced'].A.astype(float), adata.layers['spliced'].A.astype(float)
                # Dispersion
                mean_u, mean_s, dispersion_u, dispersion_s = get_dispersion(U[train_idx], S[train_idx])
                adata.var["mean_u"] = mean_u
                adata.var["mean_s"] = mean_s
                adata.var["dispersion_u"] = dispersion_u
                adata.var["dispersion_s"] = dispersion_s
                lu, ls = get_cell_scale(U, S, train_idx, True, 50)
                adata.obs["library_scale_u"] = lu
                adata.obs["library_scale_s"] = ls
                scaling_discrete = np.std(U[train_idx], 0) / (np.std(S[train_idx, 0]) + 1e-16)
            # Get the ODE parameters
            U, S = adata.layers['Mu'][train_idx], adata.layers['Ms'][train_idx]
            X = np.concatenate((U, S), 1)
            # Dynamical Model Parameters
            (alpha, beta, gamma,
             scaling,
             toff,
             u0, s0,
             sigma_u, sigma_s,
             T,
             gene_score) = init_params(X, p, fit_scaling=True)
            gene_mask = (gene_score == 1.0)
            if discrete:
                scaling = np.clip(scaling_discrete, 1e-6, None)
            if filter_gene:
                adata._inplace_subset_var(gene_mask)
                U, S = U[:, gene_mask], S[:, gene_mask]
                G = adata.n_vars
                alpha = alpha[gene_mask]
                beta = beta[gene_mask]
                gamma = gamma[gene_mask]
                scaling = scaling[gene_mask]
                toff = toff[gene_mask]
                u0 = u0[gene_mask]
                s0 = s0[gene_mask]
                sigma_u = sigma_u[gene_mask]
                sigma_s = sigma_s[gene_mask]
                T = T[:, gene_mask]
            if init_method == "random":
                print("Random Initialization.")
                self.alpha = nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float())
                self.beta = nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float())
                self.gamma = nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float())
                self.ton = torch.nn.Parameter(torch.ones(G, device=device).float()*(-10))
                self.t_init = None
            elif init_method == "tprior":
                print("Initialization using prior time.")
                t_prior = adata.obs[init_key].to_numpy()
                t_prior = t_prior[train_idx]
                std_t = (np.std(t_prior)+1e-3)*0.05
                self.t_init = np.random.uniform(t_prior-std_t, t_prior+std_t)
                self.t_init -= self.t_init.min()
                self.t_init = self.t_init
                self.t_init = self.t_init/self.t_init.max()*tmax
                self.toff_init = get_ts_global(self.t_init, U/scaling, S, 95)
                self.alpha_init, self.beta_init, self.gamma_init, self.ton_init = reinit_params(U/scaling,
                                                                                                S,
                                                                                                self.t_init,
                                                                                                self.toff_init)

                self.alpha = nn.Parameter(torch.tensor(np.log(self.alpha_init), device=device).float())
                self.beta = nn.Parameter(torch.tensor(np.log(self.beta_init), device=device).float())
                self.gamma = nn.Parameter(torch.tensor(np.log(self.gamma_init), device=device).float())
                self.ton = (nn.Parameter((torch.ones(G, device=device)*(-10)).float())
                            if init_ton_zero else
                            nn.Parameter(torch.tensor(np.log(self.ton_init+1e-10), device=device).float()))
            else:
                print("Initialization using the steady-state and dynamical models.")
                if init_key is not None:
                    self.t_init = adata.obs[init_key].to_numpy()[train_idx]
                else:
                    T = T+np.random.rand(T.shape[0], T.shape[1]) * 1e-3
                    T_eq = np.zeros(T.shape)
                    n_bin = T.shape[0]//50+1
                    for i in range(T.shape[1]):
                        T_eq[:, i] = hist_equal(T[:, i], tmax, 0.9, n_bin)
                    if "init_t_quant" in kwargs:
                        self.t_init = np.quantile(T_eq, kwargs["init_t_quant"], 1)
                    else:
                        self.t_init = np.quantile(T_eq, 0.5, 1)
                self.toff_init = get_ts_global(self.t_init, U/scaling, S, 95)
                self.alpha_init, self.beta_init, self.gamma_init, self.ton_init = reinit_params(U/scaling,
                                                                                                S,
                                                                                                self.t_init,
                                                                                                self.toff_init)

                self.alpha = nn.Parameter(torch.tensor(np.log(self.alpha_init), device=device).float())
                self.beta = nn.Parameter(torch.tensor(np.log(self.beta_init), device=device).float())
                self.gamma = nn.Parameter(torch.tensor(np.log(self.gamma_init), device=device).float())
                self.ton = (nn.Parameter((torch.ones(adata.n_vars, device=device)*(-10)).float())
                            if init_ton_zero else
                            nn.Parameter(torch.tensor(np.log(self.ton_init+1e-10), device=device).float()))
            self.register_buffer('scaling', torch.tensor(np.log(scaling), device=device).float())
            # Add Gaussian noise in case of continuous model
            if not discrete:
                self.register_buffer('sigma_u', torch.tensor(np.log(sigma_u), device=device).float())
                self.register_buffer('sigma_s', torch.tensor(np.log(sigma_s), device=device).float())

            # Add variance in case of full vb
            if full_vb:
                sigma_param = np.log(0.05) * torch.ones(adata.n_vars, device=device)
                self.alpha = nn.Parameter(torch.stack([self.alpha, sigma_param]).float())
                self.beta = nn.Parameter(torch.stack([self.beta, sigma_param]).float())
                self.gamma = nn.Parameter(torch.stack([self.gamma, sigma_param]).float())

            self.u0 = nn.Parameter(torch.tensor(np.log(u0+1e-10), device=device).float())
            self.s0 = nn.Parameter(torch.tensor(np.log(s0+1e-10), device=device).float())

        # Gene dyncamical mode initialization
        if init_method == 'tprior':
            w = assign_gene_mode_tprior(adata, init_key, train_idx)
        else:
            dyn_mask = (T > tmax*0.01) & (np.abs(T-toff) > tmax*0.01)
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
        print(f"Initial induction: {np.sum(w >= 0.5)}, repression: {np.sum(w < 0.5)}/{G}")
        adata.var["w_init"] = w
        logit_pw = 0.5*(np.log(w+1e-10) - np.log(1-w+1e-10))
        logit_pw = np.stack([logit_pw, -logit_pw], 1)
        self.logit_pw = nn.Parameter(torch.tensor(logit_pw, device=device).float())

        self.fc1 = nn.Linear(dim_z+dim_cond, N1).to(device)
        self.bn1 = nn.BatchNorm1d(num_features=N1).to(device)
        self.dpt1 = nn.Dropout(p=0.2).to(device)
        self.fc2 = nn.Linear(N1, N2).to(device)
        self.bn2 = nn.BatchNorm1d(num_features=N2).to(device)
        self.dpt2 = nn.Dropout(p=0.2).to(device)

        self.fc_out1 = nn.Linear(N2, G).to(device)

        self.net_rho = nn.Sequential(self.fc1, self.bn1, nn.LeakyReLU(), self.dpt1,
                                     self.fc2, self.bn2, nn.LeakyReLU(), self.dpt2)

        self.fc3 = nn.Linear(dim_z+dim_cond, N1).to(device)
        self.bn3 = nn.BatchNorm1d(num_features=N1).to(device)
        self.dpt3 = nn.Dropout(p=0.2).to(device)
        self.fc4 = nn.Linear(N1, N2).to(device)
        self.bn4 = nn.BatchNorm1d(num_features=N2).to(device)
        self.dpt4 = nn.Dropout(p=0.2).to(device)

        self.fc_out2 = nn.Linear(N2, G).to(device)

        self.net_rho2 = nn.Sequential(self.fc3, self.bn3, nn.LeakyReLU(), self.dpt3,
                                      self.fc4, self.bn4, nn.LeakyReLU(), self.dpt4)

        if checkpoint is not None:
            self.alpha = nn.Parameter(torch.empty(G, device=device).float())
            self.beta = nn.Parameter(torch.empty(G, device=device).float())
            self.gamma = nn.Parameter(torch.empty(G, device=device).float())
            self.scaling = nn.Parameter(torch.empty(G, device=device).float())
            self.ton = nn.Parameter(torch.empty(G, device=device).float())
            if not discrete:
                self.sigma_u = nn.Parameter(torch.empty(G, device=device).float())
                self.sigma_s = nn.Parameter(torch.empty(G, device=device).float())

            self.load_state_dict(torch.load(checkpoint, map_location=device))
        else:
            self.init_weights()

    def init_weights(self, net_id=None):
        if net_id == 1 or net_id is None:
            for m in self.net_rho.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            nn.init.xavier_normal_(self.fc_out1.weight)
            nn.init.constant_(self.fc_out1.bias, 0.0)
        if net_id == 2 or net_id is None:
            for m in self.net_rho2.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            nn.init.xavier_normal_(self.fc_out2.weight)
            nn.init.constant_(self.fc_out2.bias, 0.0)

    def _sample_ode_param(self, random=True):
        ####################################################
        # Sample rate parameters for full vb or
        # output fixed rate parameters when
        # (1) random is set to False
        # (2) full vb is not enabled
        ####################################################
        if self.is_full_vb:
            if random:
                eps = torch.normal(mean=torch.zeros((3, self.alpha.shape[1]), device=self.alpha.device),
                                   std=torch.ones((3, self.alpha.shape[1]), device=self.alpha.device))
                alpha = torch.exp(self.alpha[0] + eps[0]*(self.alpha[1].exp()))
                beta = torch.exp(self.beta[0] + eps[1]*(self.beta[1].exp()))
                gamma = torch.exp(self.gamma[0] + eps[2]*(self.gamma[1].exp()))
                # self._eps = eps
            else:
                alpha = self.alpha[0].exp()
                beta = self.beta[0].exp()
                gamma = self.gamma[0].exp()
        else:
            alpha = self.alpha.exp()
            beta = self.beta.exp()
            gamma = self.gamma.exp()
        return alpha, beta, gamma

    def _clip_rate(self, rate, max_val):
        clip_fn = nn.Hardtanh(-16, np.log(max_val))
        return clip_fn(rate)

    def forward_basis(self, t, z, condition=None, eval_mode=False, neg_slope=0.0):
        ####################################################
        # Outputs a (n sample, n basis, n gene) tensor
        ####################################################
        if condition is None:
            rho = torch.sigmoid(self.fc_out1(self.net_rho(z)))
        else:
            rho = torch.sigmoid(self.fc_out1(self.net_rho(torch.cat((z, condition), 1))))
        zero_mtx = torch.zeros(rho.shape, device=rho.device, dtype=float)
        zero_vec = torch.zeros(self.u0.shape, device=rho.device, dtype=float)
        alpha, beta, gamma = self._sample_ode_param(random=not eval_mode)
        # tensor shape (n_cell, 2, n_gene)
        alpha = torch.stack([alpha*rho, zero_mtx], 1)
        u0 = torch.stack([zero_vec, self.u0.exp()])
        s0 = torch.stack([zero_vec, self.s0.exp()])
        tau = torch.stack([F.leaky_relu(t - self.ton.exp(), neg_slope) for i in range(2)], 1)
        Uhat, Shat = pred_su(tau,
                             u0,
                             s0,
                             alpha,
                             beta,
                             gamma)
        Uhat = Uhat * torch.exp(self.scaling)

        Uhat = F.relu(Uhat)
        Shat = F.relu(Shat)
        vu = alpha - beta * Uhat / torch.exp(self.scaling)
        vs = beta * Uhat / torch.exp(self.scaling) - gamma * Shat
        return Uhat, Shat, vu, vs

    def forward(self, t, z, u0=None, s0=None, t0=None, condition=None, eval_mode=False, neg_slope=0.0):
        ####################################################
        # top-level forward function for the decoder class
        ####################################################
        if u0 is None or s0 is None or t0 is None:
            return self.forward_basis(t, z, condition, eval_mode, neg_slope)
        else:
            if condition is None:
                rho = torch.sigmoid(self.fc_out2(self.net_rho2(z)))
            else:
                rho = torch.sigmoid(self.fc_out2(self.net_rho2(torch.cat((z, condition), 1))))
            alpha, beta, gamma = self._sample_ode_param(random=not eval_mode)
            alpha = alpha*rho
            Uhat, Shat = pred_su(F.leaky_relu(t-t0, neg_slope),
                                 u0/self.scaling.exp(),
                                 s0,
                                 alpha,
                                 beta,
                                 gamma)
            Uhat = Uhat * torch.exp(self.scaling)

            Uhat = F.relu(Uhat)
            Shat = F.relu(Shat)
            Vu = alpha - beta * Uhat / torch.exp(self.scaling)
            Vs = beta * Uhat / torch.exp(self.scaling) - gamma * Shat
        return Uhat, Shat, Vu, Vs


class VAE(VanillaVAE):
    """VeloVAE Model
    """
    def __init__(self,
                 adata,
                 tmax,
                 dim_z,
                 dim_cond=0,
                 device='cpu',
                 hidden_size=(500, 250, 250, 500),
                 full_vb=False,
                 discrete=False,
                 init_method="steady",
                 init_key=None,
                 tprior=None,
                 init_ton_zero=True,
                 filter_gene=False,
                 count_distribution="Poisson",
                 std_z_prior=0.01,
                 checkpoints=[None, None],
                 rate_prior={
                     'alpha': (0.0, 1.0),
                     'beta': (0.0, 0.5),
                     'gamma': (0.0, 0.5)
                 },
                 **kwargs):
        """VeloVAE Model

        Arguments
        ---------

        adata : :class:`anndata.AnnData`
            AnnData object containing all relevant data and meta-data
        tmax : float
            Time range.
            This is used to restrict the cell time within certain range. In the
            case of a Gaussian model without capture time, tmax/2 will be the mean of prior.
            If capture time is provided, then they are scaled to match a range of tmax.
            In the case of a uniform model, tmax is strictly the maximum time.
        dim_z : int
            Dimension of the latent cell state
        dim_cond : int, optional
            Dimension of additional information for the conditional VAE.
            Set to zero by default, equivalent to a VAE.
            This feature is not stable now.
        device : {'gpu','cpu'}, optional
            Training device
        hidden_size : tuple of int, optional
            Width of the hidden layers. Should be a tuple of the form
            (encoder layer 1, encoder layer 2, decoder layer 1, decoder layer 2)
        full_vb : bool, optional
            Enable the full variational Bayes
        discrete : bool, optional
            Enable the discrete count model
        init_method : {'random', 'tprior', 'steady}, optional
            Initialization method.
            Should choose from
            (1) random: random initialization
            (2) tprior: use the capture time to estimate rate parameters. Cell time will be
                        randomly sampled with the capture time as the mean. The variance can
                        be controlled by changing 'time_overlap' in config.
            (3) steady: use the steady-state model to estimate gamma, alpha and assume beta = 1.
                        After this, a global cell time is estimated by taking the quantile over
                        all local times. Finally, rate parameters are reinitialized using the
                        global cell time.
        init_key : str, optional
            column in the AnnData object containing the capture time
        tprior : str, optional
            key in adata.obs that stores the capture time.
            Used for informative time prior
        init_ton_zero : bool, optional
            Whether to add a non-zero switch-on time for each gene.
            It's set to True if there's no capture time.
        filter_gene : bool, optional
            Whether to remove non-velocity genes
        count_distriution : {'auto', 'Poisson', 'NB'}, optional
            Count distribution, effective only when discrete=True
            The current version only assumes Poisson or negative binomial distributions.
            When set to 'auto', the program determines a proper one based on over dispersion
        std_z_prior : float, optional
            Standard deviation of the prior (isotropical Gaussian) of cell state.
        checkpoints : list of 2 strings, optional
            Contains the path to saved encoder and decoder models.
            Should be a .pt file.
        rate_prior : dict, optional
            Prior distribution of rate parameters.
            Keys are always `alpha',`beta',`gamma'
            Values are length-2 tuples (mu, sigma), representing the mean and standard deviation
            of log rates.
        """
        t_start = time.time()
        self.timer = 0
        self.is_discrete = discrete
        self.is_full_vb = full_vb

        # Training Configuration
        self.config = {
            # Model Parameters
            "dim_z": dim_z,
            "hidden_size": hidden_size,
            "tmax": tmax,
            "init_method": init_method,
            "init_key": init_key,
            "tprior": tprior,
            "std_z_prior": std_z_prior,
            "tail": 0.01,
            "time_overlap": 0.05,
            "n_neighbors": 10,
            "dt": (0.03, 0.06),
            "n_bin": None,

            # Training Parameters
            "n_epochs": 1000,
            "n_epochs_post": 1000,
            "n_refine": 20,
            "learning_rate": None,
            "learning_rate_ode": None,
            "learning_rate_post": None,
            "lambda": 1e-3,
            "lambda_rho": 1e-3,
            "kl_t": 1.0,
            "kl_z": 1.0,
            "kl_w": 0.01,
            "reg_v": 0.0,
            "max_rate": 1e4,
            "test_epoch": 10,
            "save_epoch": 100,
            "n_warmup": 5,
            "early_stop": 3,
            "early_stop_thred": adata.n_vars*1e-3,
            "train_test_split": 0.7,
            "neg_slope": 0.0,
            "train_std": False,
            "train_ton": (init_method != 'tprior'),
            "weight_sample": False,
            "vel_continuity_loss": False,

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
        self.split_train_test(adata.n_obs)

        self.dim_z = dim_z
        self.enable_cvae = dim_cond > 0

        self.decoder = decoder(adata,
                               tmax,
                               self.train_idx,
                               dim_z,
                               N1=hidden_size[2],
                               N2=hidden_size[3],
                               full_vb=full_vb,
                               discrete=discrete,
                               init_ton_zero=init_ton_zero,
                               filter_gene=filter_gene,
                               device=self.device,
                               init_method=init_method,
                               init_key=init_key,
                               checkpoint=checkpoints[1],
                               **kwargs).float()

        try:
            G = adata.n_vars
            self.encoder = encoder(2*G,
                                   dim_z,
                                   dim_cond,
                                   hidden_size[0],
                                   hidden_size[1],
                                   checkpoint=checkpoints[0]).to(device)
        except IndexError:
            print('Please provide two dimensions!')

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

        self.timer = time.time()-t_start

    def _pick_loss_func(self, adata, count_distribution):
        ##############################################################
        # Pick the corresponding loss function (mainly the generative
        # likelihood function) based on count distribution
        ##############################################################
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
                print(f"Mean dispersion: u={dispersion_u.mean():.2f}, s={dispersion_s.mean():.2f}")
                print(f"Over-Dispersion = {p_nb:.2f} => Using {count_distribution} to model count data.")
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
                lu_scale,
                ls_scale,
                edge_weight=None,
                u0=None,
                s0=None,
                t0=None,
                t1=None,
                condition=None):
        """Standard forward pass.

        Arguments
        ---------

        data_in : `torch.tensor`
            input count data, (N, 2G)
        lu_scale : `torch.tensor`
            library size scaling factor of unspliced counts, (G)
            Effective in the discrete mode and set to 1's in the
            continuouts model
        ls_scale : `torch.tensor`
            Similar to lu_scale, but for spliced counts, (G)
        u0 : `torch.tensor`, optional
            Initial condition of u, (N, G)
            This is set to None in the first stage when cell time is
            not fixed. It will have some value in the second stage, so the users
            shouldn't worry about feeding the parameter themselves.
        s0 : `torch.tensor`, optional
            Initial condition of s, (N,G)
        t0 : `torch.tensor`, optional
            time at the initial condition, (N,1)
        t1 : `torch.tensor`, optional
            time at the future state.
            Used only when `vel_continuity_loss` is set to True
        condition : `torch.tensor`, optional
            Any additional condition to the VAE

        Returns
        -------

        mu_t : `torch.tensor`, optional
            time mean, (N,1)
        std_t : `torch.tensor`, optional
            time standard deviation, (N,1)
        mu_z : `torch.tensor`, optional
            cell state mean, (N, Cz)
        std_z : `torch.tensor`, optional
            cell state standard deviation, (N, Cz)
        t : `torch.tensor`, optional
            sampled cell time, (N,1)
        z : `torch.tensor`, optional
            sampled cell sate, (N,Cz)
        uhat : `torch.tensor`, optional
            predicted mean u values, (N,G)
        shat : `torch.tensor`, optional
            predicted mean s values, (N,G)
        """
        data_in_scale = data_in
        G = data_in_scale.shape[-1]//2
        # optional data scaling
        if self.config["scale_gene_encoder"]:
            data_in_scale = torch.cat((data_in_scale[:, :G]/self.decoder.scaling.exp(),
                                       data_in_scale[:, G:]), 1)
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

        uhat, shat, vu, vs = self.decoder.forward(t, z, u0, s0, t0, condition, neg_slope=self.config["neg_slope"])

        if t1 is not None:  # predict the future state when we enable velocity continuity loss
            uhat_fw, shat_fw, vu_fw, vs_fw = self.decoder.forward(t1,
                                                                  z,
                                                                  uhat,
                                                                  shat,
                                                                  t,
                                                                  condition,
                                                                  neg_slope=self.config["neg_slope"])
        else:
            uhat_fw, shat_fw, vu_fw, vs_fw = None, None, None, None

        return mu_t, std_t, mu_z, std_z, t, z, uhat, shat, uhat_fw, shat_fw, vu, vs, vu_fw, vs_fw

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
        """Evaluate the model on the validation dataset.
        The major difference from forward pass is that we use the mean time and
        cell state instead of random sampling. The input arguments are the same as 'forward'.
        """
        data_in_scale = data_in
        G = data_in_scale.shape[-1]//2
        # optional data scaling
        if self.config["scale_gene_encoder"]:
            data_in_scale = torch.cat((data_in_scale[:, :G]/self.decoder.scaling.exp(),
                                       data_in_scale[:, G:]), 1)
        if self.config["scale_cell_encoder"]:
            data_in_scale = torch.cat((data_in_scale[:, :G]/lu_scale,
                                       data_in_scale[:, G:]/ls_scale), 1)
        if self.config["log1p"]:
            data_in_scale = torch.log1p(data_in_scale)
        mu_t, std_t, mu_z, std_z = self.encoder.forward(data_in_scale,
                                                        edge_index,
                                                        edge_weight,
                                                        condition)

        uhat, shat, vu, vs = self.decoder.forward(mu_t,
                                                  mu_z,
                                                  u0=u0,
                                                  s0=s0,
                                                  t0=t0,
                                                  condition=condition,
                                                  eval_mode=True)
        if t1 is not None:
            uhat_fw, shat_fw, vu_fw, vs_fw = self.decoder.forward(t1,
                                                                  mu_z,
                                                                  uhat,
                                                                  shat,
                                                                  mu_t,
                                                                  condition,
                                                                  eval_mode=True)
        else:
            uhat_fw, shat_fw, vu_fw, vs_fw = None, None, None, None

        return mu_t, std_t, mu_z, std_z, uhat, shat, uhat_fw, shat_fw, vu, vs, vu_fw, vs_fw

    def loss_vel(self, x0, xhat, v):
        ##############################################################
        # Velocity correlation loss, optionally
        # added to the total loss function
        ##############################################################
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
            elbo_collapsed_categorical(self.decoder.logit_pw, self.alpha_w, 2, self.decoder.scaling.shape[0])
            if self.train_stage == 1 else 0
            )
        return (self.config["kl_t"]*kldt
                + self.config["kl_z"]*kldz
                + self.config["kl_param"]*kld_param
                + self.config["kl_w"]*kldw)

    def vae_risk_gaussian(self,
                          q_tx, p_t,
                          q_zx, p_z,
                          u, s, uhat, shat,
                          uhat_fw=None, shat_fw=None,
                          u1=None, s1=None,
                          weight=None):
        """Training objective function. This is the negative ELBO.

        Arguments
        ---------

        q_tx : tuple of `torch.tensor`
            Parameters of time posterior. Mean and std are both (N, 1) tensors.
        p_t : tuple of `torch.tensor`
            Parameters of time prior.
        q_zx : tuple of `torch.tensor`
            Parameters of cell state posterior. Mean and std are both (N, Dz) tensors.
        p_z  : tuple of `torch.tensor`
            Parameters of cell state prior.
        u, s : `torch.tensor`
            Input data
        uhat, shat : torch.tensor
            Prediction by VeloVAE
        weight : `torch.tensor`, optional
            Sample weight. This feature is not stable. Please consider setting it to None.

        Returns
        -------
        Negative ELBO : torch.tensor, scalar
        """
        kl_term = self._compute_kl_term(q_tx, p_t, q_zx, p_z)

        sigma_u = self.decoder.sigma_u.exp()
        sigma_s = self.decoder.sigma_s.exp()

        # u and sigma_u has the original scale
        clip_fn = nn.Hardtanh(-P_MAX, P_MAX)
        if uhat.ndim == 3:  # stage 1
            logp = -0.5*((u.unsqueeze(1)-uhat)/sigma_u).pow(2)\
                   - 0.5*((s.unsqueeze(1)-shat)/sigma_s).pow(2)\
                   - torch.log(sigma_u)-torch.log(sigma_s*2*np.pi)
            logp = clip_fn(logp)
            pw = F.softmax(self.decoder.logit_pw, dim=1).T
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

        return - err_rec + kl_term

    def _kl_poisson(self, lamb_1, lamb_2):
        return lamb_1 * (torch.log(lamb_1) - torch.log(lamb_2)) + lamb_2 - lamb_1

    def vae_risk_poisson(self,
                         q_tx, p_t,
                         q_zx, p_z,
                         u, s, uhat, shat,
                         uhat_fw=None, shat_fw=None,
                         u1=None, s1=None,
                         weight=None,
                         eps=1e-2):
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
            pw = F.softmax(self.decoder.logit_pw, dim=1).T
            logp = torch.sum(pw*logp, 1)
        else:
            logp = poisson_u.log_prob(u) + poisson_s.log_prob(s)

        # velocity continuity loss
        if uhat_fw is not None and shat_fw is not None:
            logp = logp - self._kl_poisson(u1, uhat_fw) - self._kl_poisson(s1, shat_fw)
        if weight is not None:
            logp = logp*weight

        err_rec = torch.mean(logp.sum(1))

        return - err_rec + kl_term

    def _kl_nb(self, m1, m2, p):
        r1 = m1 * (1 - p) / p
        r2 = m2 * (1 - p) / p
        return (r1 - r2)*torch.log(p)

    def vae_risk_nb(self,
                    q_tx, p_t,
                    q_zx, p_z,
                    u, s, uhat, shat,
                    uhat_fw=None, shat_fw=None,
                    u1=None, s1=None,
                    weight=None,
                    eps=1e-2):
        kl_term = self._compute_kl_term(q_tx, p_t, q_zx, p_z)

        # NB
        p_nb_u = torch.sigmoid(self.eta_u+torch.log(F.relu(uhat)+eps))
        p_nb_s = torch.sigmoid(self.eta_s+torch.log(F.relu(shat)+eps))
        nb_u = NegativeBinomial((F.relu(uhat)+1e-10)*(1-p_nb_u)/p_nb_u, probs=p_nb_u)
        nb_s = NegativeBinomial((F.relu(shat)+1e-10)*(1-p_nb_s)/p_nb_s, probs=p_nb_s)
        if uhat.ndim == 3:  # stage 1
            logp = nb_u.log_prob(torch.stack([u, u], 1)) + nb_s.log_prob(torch.stack([s, s], 1))
            pw = F.softmax(self.decoder.logit_pw, dim=1).T
            logp = torch.sum(pw*logp, 1)
        else:
            logp = nb_u.log_prob(u) + nb_s.log_prob(s)
        # velocity continuity loss
        if uhat_fw is not None and shat_fw is not None:
            logp = logp - self._kl_nb(uhat_fw, u1, p_nb_u) - self._kl_nb(shat_fw, s1, p_nb_s)
        if weight is not None:
            logp = logp*weight
        err_rec = torch.mean(torch.sum(logp, 1))

        return - err_rec + kl_term

    def train_epoch(self, optimizer, optimizer2=None, validation=False):
        ##########################################################################
        # Training in each epoch.
        # Early stopping if enforced by default.
        # Arguments
        # ---------
        # 1.  train_loader [torch.utils.data.DataLoader]
        #     Data loader of the input data.
        # 2.  test_set [torch.utils.data.Dataset]
        #     Validation dataset
        # 3.  optimizer  [optimizer from torch.optim]
        # 4.  optimizer2 [optimizer from torch.optim
        #     (Optional) A second optimizer.
        #     This is used when we optimize NN and ODE simultaneously in one epoch.
        #     By default, VeloVAE performs alternating optimization in each epoch.
        #     The argument will be set to proper value automatically.
        # 5.  K [int]
        #     Alternating update period.
        #     For every K updates of optimizer, there's one update for optimizer2.
        #     If set to 0, optimizer2 will be ignored and only optimizer will be
        #     updated. Users can set it to 0 if they want to update sorely NN in one
        #     epoch and ODE in the next epoch.
        # Returns
        # 1.  stop_training [bool]
        #     Whether to stop training based on the early stopping criterium.
        ##########################################################################
        G = self.graph_data.G
        self.set_mode('train')
        if validation:
            elbo_test = self.test(None,
                                  self.counter,
                                  True)

            if len(self.loss_test) > 0:  # update the number of epochs with dropping/converging ELBO
                if elbo_test - self.loss_test[-1] <= self.config["early_stop_thred"]:
                    self.n_drop = self.n_drop+1
                else:
                    self.n_drop = 0
            self.loss_test.append(elbo_test)
            self.set_mode('train')

            if self.n_drop >= self.config["early_stop"] and self.config["early_stop"] > 0:
                return True

        optimizer.zero_grad()
        if optimizer2 is not None:
            optimizer2.zero_grad()
        u0 = self.graph_data.u0
        s0 = self.graph_data.s0
        t0 = self.graph_data.t0
        u1 = self.graph_data.u1
        s1 = self.graph_data.s1
        t1 = self.graph_data.t1
        lu_scale = self.lu_scale.exp()
        ls_scale = self.ls_scale.exp()

        condition = F.one_hot(self.graph_data.data.y, self.n_type).float() if self.enable_cvae else None
        (mu_tx, std_tx,
         mu_zx, std_zx,
         t, z,
         uhat, shat,
         uhat_fw, shat_fw,
         vu, vs,
         vu_fw, vs_fw) = self.forward(self.graph_data.data.x,
                                      self.graph_data.data.edge_index,
                                      lu_scale, ls_scale,
                                      self.graph_data.edge_weight,
                                      u0, s0,
                                      t0, t1,
                                      condition)
        if uhat.ndim == 3:
            lu_scale = lu_scale.unsqueeze(-1)
            ls_scale = ls_scale.unsqueeze(-1)
        if uhat_fw is not None and shat_fw is not None:
            uhat_fw = uhat_fw[self.train_idx]
            shat_fw = shat_fw[self.train_idx]
            u1 = u1[self.train_idx]
            s1 = s1[self.train_idx]
        loss = self.vae_risk((mu_tx[self.train_idx], std_tx[self.train_idx]),
                             self.p_t[:, self.train_idx, :],
                             (mu_zx[self.train_idx], std_zx[self.train_idx]),
                             self.p_z[:, self.train_idx, :],
                             self.graph_data.data.x[self.train_idx][:, :G],
                             self.graph_data.data.x[self.train_idx][:, G:],
                             uhat[self.train_idx]*lu_scale[self.train_idx],
                             shat[self.train_idx]*ls_scale[self.train_idx],
                             uhat_fw, shat_fw,
                             u1, s1,
                             None)

        # Add velocity regularization
        if self.use_knn and self.config["reg_v"] > 0:
                scaling = self.decoder.scaling.exp()
                loss = loss - self.config["reg_v"] *\
                    (self.loss_vel(u0/scaling, uhat/scaling*lu_scale, vu) + self.loss_vel(s0, shat*ls_scale, vs))
                if vu_fw is not None and vs_fw is not None:
                    loss = loss - self.config["reg_v"]\
                        * (self.loss_vel(uhat/scaling, uhat_fw/scaling, vu_fw) + self.loss_vel(shat, shat_fw, vs_fw))
        loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_value_(self.encoder.parameters(), GRAD_MAX)
        torch.nn.utils.clip_grad_value_(self.decoder.parameters(), GRAD_MAX)
        optimizer.step()
        if optimizer2 is not None:
            optimizer2.step()

        self.loss_train.append(loss.detach().cpu().item())
        self.counter = self.counter + 1

        if validation:
            with torch.no_grad():
                loss_val = self.vae_risk((mu_tx[self.test_idx], std_tx[self.test_idx]),
                                        self.p_t[:, self.test_idx, :],
                                        (mu_zx[self.test_idx], std_zx[self.test_idx]),
                                        self.p_z[:, self.test_idx, :],
                                        self.graph_data.data.x[self.test_idx][:, :G],
                                        self.graph_data.data.x[self.test_idx][:, G:],
                                        uhat[self.test_idx]*lu_scale[self.test_idx],
                                        shat[self.test_idx]*ls_scale[self.test_idx],
                                        uhat_fw, shat_fw,
                                        u1, s1,
                                        None)
            if len(self.loss_test) > 0:  # update the number of epochs with dropping/converging ELBO
                if loss_val - self.loss_test[-1] <= self.config["early_stop_thred"]:
                    self.n_drop = self.n_drop+1
                else:
                    self.n_drop = 0
            self.loss_test.append(loss_val)
            if self.n_drop >= self.config["early_stop"] and self.config["early_stop"] > 0:
                return True
        return False

    def update_x0(self, n_bin=None):
        ##########################################################################
        # Estimate the initial conditions using KNN
        # < Input Arguments >
        # 1.  U [tensor (N,G)]
        #     Input unspliced count matrix
        # 2   S [tensor (N,G)]
        #     Input spliced count matrix
        # 3.  n_bin [int]
        #     (Optional) Set to a positive integer if binned KNN is used.
        # < Output >
        # 1.  u0 [tensor (N,G)]
        #     Initial condition for u
        # 2.  s0 [tensor (N,G)]
        #     Initial condition for s
        # 3. t0 [tensor (N,1)]
        #    Initial time
        ##########################################################################
        start = time.time()
        self.set_mode('eval')
        G = self.graph_data.G
        out, elbo = self.pred_all(self.cell_labels,
                                  "both",
                                  ["uhat", "shat", "t", "z"],
                                  np.array(range(G)))
        t, z = out["t"], out["z"]
        # Clip the time to avoid outliers
        t = np.clip(t, 0, np.quantile(t, 0.99))
        dt = (self.config["dt"][0]*(t.max()-t.min()), self.config["dt"][1]*(t.max()-t.min()))
        u1, s1, t1 = None, None, None
        # Compute initial conditions of cells without a valid pool of neighbors
        with torch.no_grad():
            init_mask = (t <= np.quantile(t, 0.01))
            u0_init = np.mean(self.graph_data.data.x[init_mask, :G].detach().cpu().numpy(), 0)
            s0_init = np.mean(self.graph_data.data.x[init_mask, G:].detach().cpu().numpy(), 0)
        if n_bin is None:
            print("Cell-wise KNN Estimation.")
            u0, s0, t0 = knnx0(out["uhat"][self.train_idx],
                               out["shat"][self.train_idx],
                               t[self.train_idx],
                               z[self.train_idx],
                               t,
                               z,
                               dt,
                               self.config["n_neighbors"],
                               u0_init,
                               s0_init,
                               hist_eq=True)
            if self.config["vel_continuity_loss"]:
                u1, s1, t1 = knnx0(out["uhat"][self.train_idx],
                                   out["shat"][self.train_idx],
                                   t[self.train_idx],
                                   z[self.train_idx],
                                   t,
                                   z,
                                   dt,
                                   self.config["n_neighbors"],
                                   u0_init,
                                   s0_init,
                                   forward=True,
                                   hist_eq=True)
                t1 = t1.reshape(-1, 1)
        else:
            print(f"Fast KNN Estimation with {n_bin} time bins.")
            u0, s0, t0 = knnx0_bin(out["uhat"][self.train_idx],
                                   out["shat"][self.train_idx],
                                   t[self.train_idx],
                                   z[self.train_idx],
                                   t,
                                   z,
                                   dt,
                                   self.config["n_neighbors"])
            if self.config["vel_continuity_loss"]:
                u1, s1, t1 = knnx0_bin(out["uhat"][self.train_idx],
                                       out["shat"][self.train_idx],
                                       t[self.train_idx],
                                       z[self.train_idx],
                                       t,
                                       z,
                                       dt,
                                       self.config["n_neighbors"],
                                       forward=True)
        print(f"Finished. Actual Time: {convert_time(time.time()-start)}")
        # return u0, s0, t0.reshape(-1,1), u1, s1, t1
        self.graph_data.u0 = torch.tensor(u0, dtype=torch.float32, device=self.device)
        self.graph_data.s0 = torch.tensor(s0, dtype=torch.float32, device=self.device)
        self.graph_data.t0 = torch.tensor(t0.reshape(-1, 1), dtype=torch.float32, device=self.device)
        if self.config['vel_continuity_loss']:
            self.u1 = torch.tensor(u1, dtype=torch.float32, device=self.device)
            self.s1 = torch.tensor(s1, dtype=torch.float32, device=self.device)
            self.t1 = torch.tensor(t1.reshape(-1, 1), dtype=torch.float32, device=self.device)

    def _set_lr(self, p):
        if self.is_discrete:
            self.config["learning_rate"] = 10**(-8.3*p-2.25)
            self.config["learning_rate_post"] = self.config["learning_rate"]
            self.config["learning_rate_ode"] = 5*self.config["learning_rate"]
        else:
            self.config["learning_rate"] = 10**(-4*p-3)
            self.config["learning_rate_post"] = self.config["learning_rate"]
            self.config["learning_rate_ode"] = 8*self.config["learning_rate"]

    def train(self,
              adata,
              graph,
              config={},
              plot=False,
              gene_plot=[],
              cluster_key="clusters",
              figure_path="figures",
              embed="umap",
              random_state=2022):
        """The high-level API for training.

        Arguments
        ---------

        adata : :class:`anndata.AnnData`
        config : dictionary, optional
            Contains all hyper-parameters.
        plot : bool, optional
            Whether to plot some sample genes during training. Used for debugging.
        gene_plot : string list, optional
            List of gene names to plot. Used only if plot==True
        cluster_key : str, optional
            Key in adata.obs storing the cell type annotation.
        figure_path : str, optional
            Path to the folder for saving plots.
        embed : str, optional
            Low dimensional embedding in adata.obsm. The actual key storing the embedding should be f'X_{embed}'
        """
        self.load_config(config)
        if self.config["learning_rate"] is None:
            p = (np.sum(adata.layers["unspliced"].A > 0)
                 + (np.sum(adata.layers["spliced"].A > 0)))/adata.n_obs/adata.n_vars/2
            self._set_lr(p)
            print(f'Learning Rate based on Data Sparsity: {self.config["learning_rate"]:.4f}')
        print("--------------------------- Train a VeloVAE ---------------------------")
        # Get data loader
        if self.is_discrete:
            U, S = np.array(adata.layers['unspliced'].todense()), np.array(adata.layers['spliced'].todense())
            X = np.concatenate((U, S), 1).astype(int)
        else:
            X = np.concatenate((adata.layers['Mu'], adata.layers['Ms']), 1).astype(float)
        try:
            Xembed = adata.obsm[f"X_{embed}"]
        except KeyError:
            print("Embedding not found! Set to None.")
            Xembed = np.nan*np.ones((adata.n_obs, 2))
            plot = False

        cell_labels_raw = (adata.obs[cluster_key].to_numpy() if cluster_key in adata.obs
                           else np.array(['Unknown' for i in range(adata.n_obs)]))
        # Encode the labels
        cell_types_raw = np.unique(cell_labels_raw)
        self.label_dic, self.label_dic_rev = encode_type(cell_types_raw)

        self.n_type = len(cell_types_raw)
        self.cell_labels = np.array([self.label_dic[x] for x in cell_labels_raw])
        self.cell_types = np.array([self.label_dic[cell_types_raw[i]] for i in range(self.n_type)])

        print("*********               Creating a Graph Dataset              *********")
        n_train = int(adata.n_obs * self.config['train_test_split'])
        self.graph_data = SCGraphData(X,
                                      self.cell_labels,
                                      graph,
                                      n_train,
                                      self.device,
                                      random_state)
        self.train_idx = self.graph_data.train_idx
        self.test_idx = self.graph_data.test_idx
        print("*********                      Finished.                      *********")

        gind, gene_plot = get_gene_index(adata.var_names, gene_plot)
        os.makedirs(figure_path, exist_ok=True)

        # define optimizer
        print("*********                 Creating optimizers                 *********")
        param_nn = list(self.encoder.parameters())\
            + list(self.decoder.net_rho.parameters())\
            + list(self.decoder.fc_out1.parameters())
        param_ode = [self.decoder.alpha,
                     self.decoder.beta,
                     self.decoder.gamma,
                     self.decoder.u0,
                     self.decoder.s0,
                     self.decoder.logit_pw]
        if self.config['train_ton']:
            param_ode.append(self.decoder.ton)
        if self.config['train_std']:
            self.decoder.sigma_u.requires_grad = True
            self.decoder.sigma_s.requires_grad = True
            param_ode = param_ode+[self.decoder.sigma_u, self.decoder.sigma_s]

        optimizer = torch.optim.Adam(param_nn, lr=self.config["learning_rate"], weight_decay=self.config["lambda"])
        optimizer_ode = torch.optim.Adam(param_ode, lr=self.config["learning_rate_ode"])
        print("*********                      Finished.                      *********")

        # Main Training Process
        print("*********                    Start training                   *********")
        print("*********                      Stage  1                       *********")
        n_epochs = self.config["n_epochs"]
        start = time.time()
        for epoch in range(n_epochs):
            validation = self.counter == 1 or self.counter % self.config["test_epoch"] == 0
            # Train the encoder
            if epoch >= self.config["n_warmup"]:
                stop_training = self.train_epoch(optimizer_ode, optimizer, validation)
            else:
                stop_training = self.train_epoch(optimizer, None, validation)

            if plot and (epoch == 0 or (epoch+1) % self.config["save_epoch"] == 0):
                elbo_train = self.test(Xembed[self.train_idx],
                                       f"train{epoch+1}",
                                       False,
                                       gind,
                                       gene_plot,
                                       plot,
                                       figure_path)
                self.set_mode('train')
                elbo_test = self.loss_test[-1] if len(self.loss_test) > 0 else -np.inf
                print(f"Epoch {epoch+1}: Train ELBO = {elbo_train:.3f},\t"
                      f"Test ELBO = {elbo_test:.3f},\t"
                      f"Total Time = {convert_time(time.time()-start)}")

            if stop_training:
                print(f"*********       Stage 1: Early Stop Triggered at epoch {epoch+1}.       *********")
                break

        count_epoch = epoch+1
        n_test1 = len(self.loss_test)

        print("*********                      Stage  2                       *********")
        self.encoder.eval()
        self.use_knn = True
        self.train_stage = 2
        self.decoder.logit_pw.requires_grad = False
        if not self.is_discrete:
            sigma_u_prev = self.decoder.sigma_u.detach().cpu().numpy()
            sigma_s_prev = self.decoder.sigma_s.detach().cpu().numpy()
            u0_prev, s0_prev = None, None
            noise_change = np.inf
        x0_change = np.inf
        x0_change_prev = np.inf
        param_post = list(self.decoder.net_rho2.parameters())+list(self.decoder.fc_out2.parameters())
        optimizer_post = torch.optim.Adam(param_post,
                                          lr=self.config["learning_rate_post"],
                                          weight_decay=self.config["lambda_rho"])
        for r in range(self.config['n_refine']):
            print(f"*********             Velocity Refinement Round {r+1}              *********")
            validation = self.counter == 1 or self.counter % self.config["test_epoch"] == 0
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
                    stop_training = self.train_epoch(optimizer_post, optimizer_ode, validation)
                else:
                    stop_training = self.train_epoch(optimizer_post, None, validation)

                if plot and (epoch == 0 or (epoch+count_epoch+1) % self.config["save_epoch"] == 0):
                    elbo_train = self.test(Xembed[self.train_idx],
                                           f"train{epoch+count_epoch+1}",
                                           False,
                                           gind,
                                           gene_plot,
                                           plot,
                                           figure_path)
                    self.decoder.train()
                    elbo_test = self.loss_test[-1] if len(self.loss_test) > n_test1 else -np.inf
                    print(f"Epoch {epoch+count_epoch+1}: Train ELBO = {elbo_train:.3f},\t"
                          f"Test ELBO = {elbo_test:.3f},\t"
                          f"Total Time = {convert_time(time.time()-start)}")

                if stop_training:
                    print(f"*********       "
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
                print(f"Change in noise variance: {noise_change}")
            with torch.no_grad():
                if r > 0:
                    x0_change_prev = x0_change
                    norm_delta_x0 = torch.sqrt(((self.graph_data.u0 - u0_prev).pow(2)
                                                + (self.graph_data.s0 - s0_prev).pow(2)).sum(1).mean())
                    std_x = torch.sqrt((torch.var(u0, 0) + torch.var(s0, 0)).sum())
                    x0_change = (norm_delta_x0/std_x).cpu().item()
                    print(f"Change in x0: {x0_change}")
                u0_prev = self.graph_data.u0
                s0_prev = self.graph_data.s0

        elbo_train = self.test(Xembed[self.train_idx],
                               "final-train",
                               False,
                               gind,
                               gene_plot,
                               plot,
                               figure_path)
        elbo_test = self.test(Xembed[self.test_idx],
                              "final-test",
                              True,
                              gind,
                              gene_plot,
                              plot,
                              figure_path)
        self.loss_train.append(elbo_train)
        self.loss_test.append(elbo_test)
        # Plot final results
        if plot:
            plot_train_loss(self.loss_train,
                            range(1, len(self.loss_train)+1),
                            save=f'{figure_path}/train_loss_velovae.png')
            if self.config["test_iter"] > 0:
                plot_test_loss(self.loss_test,
                               [i*self.config["test_iter"] for i in range(1, len(self.loss_test)+1)],
                               save=f'{figure_path}/test_loss_velovae.png')

        self.timer = self.timer + (time.time()-start)
        print(f"*********              Finished. Total Time = {convert_time(self.timer)}             *********")
        print(f"Final: Train ELBO = {elbo_train:.3f},\tTest ELBO = {elbo_test:.3f}")
        return

    def pred_all(self, cell_labels, mode='test', output=["uhat", "shat", "t", "z"], gene_idx=None):
        G = self.graph_data.G
        if gene_idx is None:
            gene_idx = np.array(range(G))
        elbo = 0
        save_uhat_fw = "uhat_fw" in output and self.use_knn and self.config["vel_continuity_loss"]
        save_shat_fw = "shat_fw" in output and self.use_knn and self.config["vel_continuity_loss"]

        with torch.no_grad():
            w_hard = F.one_hot(torch.argmax(self.decoder.logit_pw, 1), num_classes=2).T
            if mode == "test":
                sample_idx = self.test_idx
            elif mode == "train":
                sample_idx = self.train_idx
            else:
                sample_idx = np.array(range(self.graph_data.G))
            u0 = self.graph_data.u0
            s0 = self.graph_data.s0
            t0 = self.graph_data.t0
            u1 = self.graph_data.u1
            s1 = self.graph_data.s1
            t1 = self.graph_data.t1
            lu_scale = self.lu_scale.exp()
            ls_scale = self.ls_scale.exp()
            y_onehot = (
                F.one_hot(self.graph_data.data.y,
                          self.n_type).float()
                if self.enable_cvae else None)
            p_t = self.p_t[:, sample_idx, :]
            p_z = self.p_z[:, sample_idx, :]
            (mu_tx, std_tx,
             mu_zx, std_zx,
             uhat, shat,
             uhat_fw, shat_fw,
             vu, vs,
             vu_fw, vs_fw) = self.eval_model(self.graph_data.data.x,
                                             self.graph_data.data.edge_index,
                                             lu_scale,
                                             ls_scale,
                                             self.graph_data.data.edge_attr,
                                             u0,
                                             s0,
                                             t0,
                                             t1,
                                             y_onehot)
            if uhat.ndim == 3:
                lu_scale = lu_scale.unsqueeze(-1)
                ls_scale = ls_scale.unsqueeze(-1)
            if uhat_fw is not None and shat_fw is not None:
                uhat_fw = uhat_fw[sample_idx]*lu_scale[sample_idx]
                shat_fw = uhat_fw[sample_idx]*ls_scale[sample_idx]
                u1 = u1[sample_idx]
                s1 = s1[sample_idx]
            loss = self.vae_risk((mu_tx[sample_idx], std_tx[sample_idx]), p_t,
                                 (mu_zx[sample_idx], std_zx[sample_idx]), p_z,
                                 self.graph_data.data.x[sample_idx, :G],
                                 self.graph_data.data.x[sample_idx, G:],
                                 uhat[sample_idx]*lu_scale[sample_idx],
                                 shat[sample_idx]*ls_scale[sample_idx],
                                 uhat_fw, shat_fw,
                                 u1, s1,
                                 None)

        out = {}
        if "uhat" in output:
            if uhat.ndim == 3:
                uhat = torch.sum(uhat*w_hard, 1)
            out["uhat"] = uhat[sample_idx][:, gene_idx].detach().cpu().numpy()
        if "shat" in output:
            if shat.ndim == 3:
                shat = torch.sum(shat*w_hard, 1)
            out["shat"] = shat[sample_idx][:, gene_idx].detach().cpu().numpy()
        if "t" in output:
            out["t"] = mu_tx[sample_idx].detach().cpu().squeeze().numpy()
            out["std_t"] = std_tx[sample_idx].detach().cpu().squeeze().numpy()
        if "z" in output:
            out["z"] = mu_zx[sample_idx].detach().cpu().numpy()
            out["std_z"] = std_zx[sample_idx].detach().cpu().numpy()
        if save_uhat_fw:
            out["uhat_fw"] = uhat_fw[sample_idx][:, gene_idx].detach().cpu().numpy()
        if save_shat_fw:
            out["shat_fw"] = shat_fw[sample_idx][:, gene_idx].detach().cpu().numpy()
        if "v" in output:
            out["vu"] = vu[sample_idx].detach().cpu().numpy()
            out["vs"] = vs[sample_idx].detach().cpu().numpy()

        return out, loss.detach().cpu().item()

    def test(self,
             Xembed,
             testid=0,
             test_mode=True,
             gind=None,
             gene_plot=None,
             plot=False,
             path='figures',
             **kwargs):
        """Evaluate the model upon training/test dataset.

        Arguments
        ---------
        Xembed : `numpy array`
            Low-dimensional embedding for plotting
        testid : str or int, optional
            Used to name the figures.
        test_mode : bool, optional
            Whether dataset is training or validation dataset. This is used when retreiving certain class variable,
            e.g. cell-specific initial condition.
        gind : `numpy array`, optional
            Index of genes in adata.var_names. Used for plotting.
        gene_plot : `numpy array`, optional
            Gene names.
        plot : bool, optional
            Whether to generate plots.
        path : str
            Saving path.

        Returns
        -------
        elbo : float
        """
        self.set_mode('eval')
        mode = "test" if test_mode else "train"
        out_type = ["uhat", "shat", "uhat_fw", "shat_fw", "t"]
        if self.train_stage == 2:
            out_type.append("v")
        out, elbo = self.pred_all(self.cell_labels, mode, out_type, gind)
        Uhat, Shat, t = out["uhat"], out["shat"], out["t"]

        G = self.graph_data.G

        if plot:
            # Plot Time
            plot_time(t, Xembed, save=f"{path}/time-{testid}-velovae.png")
            cell_labels = np.array([self.label_dic_rev[x] for x in self.cell_labels])
            cell_labels = cell_labels[self.test_idx] if test_mode else cell_labels[self.train_idx]
            # Plot u/s-t and phase portrait for each gene
            cell_idx = np.array(range(self.graph_data.N))
            if mode == "train":
                cell_idx = self.train_idx
            elif mode == "test":
                cell_idx = self.test_idx
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
                if self.train_stage == 2:
                    scaling = self.decoder.scaling[i].detach().cpu().exp().numpy()
                    plot_vel(t.squeeze(),
                             Uhat[:, i]/scaling, Shat[:, i],
                             out["vu"][:, i], out["vs"][:, i],
                             self.t0[cell_idx].squeeze(),
                             self.u0[cell_idx, idx]/scaling, self.s0[cell_idx, idx],
                             title=gene_plot[i],
                             save=f"{path}/vel-{gene_plot[i]}-{testid}.png")
                if self.config['vel_continuity_loss'] and self.train_stage == 2:
                    plot_sig(t.squeeze(),
                             u,
                             s,
                             out["uhat_fw"][:, i], out["shat_fw"][:, i],
                             cell_labels,
                             gene_plot[i],
                             save=f"{path}/sig-{gene_plot[i]}-{testid}-bw.png",
                             sparsify=self.config['sparsify'])

        return elbo

    def update_std_noise(self):
        """Update the standard deviation of Gaussian noise.
        .. deprecated:: 1.0
        """
        G = self.graph_data.G
        out, elbo = self.pred_all(self.cell_labels,
                                  mode='train',
                                  output=["uhat", "shat"],
                                  gene_idx=np.array(range(G)))
        std_u = (out["uhat"]-self.graph_data.data.x[:, :G].detach().cpu().numpy()).std(0)
        std_s = (out["shat"]-self.train_data.data.x[:, G:].detach().cpu().numpy()).std(0)
        self.decoder.sigma_u = nn.Parameter(torch.tensor(np.log(std_u),
                                            dtype=torch.float,
                                            device=self.device))
        self.decoder.sigma_s = nn.Parameter(torch.tensor(np.log(std_s),
                                            dtype=torch.float,
                                            device=self.device))
        return

    def save_anndata(self, adata, key, file_path, file_name=None):
        """Save the ODE parameters and cell time to the anndata object and write it to disk.

        Arguments
        ---------
        adata : :class:`anndata.AnnData`
        key : str
            Used to store all parameters of the model.
        file_path : str
            Saving path.
        file_name : str, optional
            If set to a string ending with .h5ad, the updated anndata object will be written to disk.
        """
        self.set_mode('eval')
        os.makedirs(file_path, exist_ok=True)
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
        adata.var[f"{key}_scaling"] = np.exp(self.decoder.scaling.detach().cpu().numpy())
        adata.var[f"{key}_sigma_u"] = np.exp(self.decoder.sigma_u.detach().cpu().numpy())
        adata.var[f"{key}_sigma_s"] = np.exp(self.decoder.sigma_s.detach().cpu().numpy())
        if self.is_discrete:
            U, S = np.array(adata.layers['unspliced'].todense()), np.array(adata.layers['spliced'].todense())
        else:
            U, S = adata.layers['Mu'], adata.layers['Ms']
        adata.varm[f"{key}_mode"] = F.softmax(self.decoder.logit_pw, 1).detach().cpu().numpy()

        out, elbo = self.pred_all(self.cell_labels,
                                  "both",
                                  gene_idx=np.array(range(adata.n_vars)))
        Uhat, Shat, t, std_t, z, std_z = out["uhat"], out["shat"], out["t"], out["std_t"], out["z"], out["std_z"]

        adata.obs[f"{key}_time"] = t
        adata.obs[f"{key}_std_t"] = std_t
        adata.obsm[f"{key}_z"] = z
        adata.obsm[f"{key}_std_z"] = std_z
        adata.layers[f"{key}_uhat"] = Uhat
        adata.layers[f"{key}_shat"] = Shat

        rho = np.zeros(adata.shape)
        with torch.no_grad():
            rho = torch.sigmoid(
                self.decoder.fc_out2(
                    self.decoder.net_rho2(torch.tensor(z, device=self.device).float())
                )
            )
        adata.layers[f"{key}_rho"] = rho.cpu().numpy()

        adata.obs[f"{key}_t0"] = self.graph_data.t0.detach().cpu().squeeze().numpy()
        adata.layers[f"{key}_u0"] = self.graph_data.u0.detach().cpu().numpy()
        adata.layers[f"{key}_s0"] = self.graph_data.s0.detach().cpu().numpy()
        if self.config["vel_continuity_loss"]:
            adata.obs[f"{key}_t1"] = self.t1.detach().cpu().squeeze().numpy()
            adata.layers[f"{key}_u1"] = self.graph_data.u1.detach().cpu().numpy()
            adata.layers[f"{key}_s1"] = self.graph_data.s1.detach().cpu().numpy()

        adata.uns[f"{key}_train_idx"] = self.train_idx
        adata.uns[f"{key}_test_idx"] = self.test_idx
        adata.uns[f"{key}_run_time"] = self.timer

        rna_velocity_vae(adata,
                         key,
                         use_raw=False,
                         use_scv_genes=False,
                         full_vb=self.is_full_vb)

        if file_name is not None:
            adata.write_h5ad(f"{file_path}/{file_name}")
