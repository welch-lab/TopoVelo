import numpy as np
import scipy as sp
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch_geometric.transforms as T
# from torch_geometric.loader import NeighborLoader
from sys import getsizeof


class SCData(Dataset):
    """This is a simple pytorch dataset class for batch training.
    Each sample represents a cell. Each dimension represents a single gene.
    The dataset also contains the cell labels (types).
    """
    def __init__(self, D, labels, u0=None, s0=None, t0=None, weight=None):
        """Class constructor

        Arguments
        ---------

        D : `numpy array`
            Cell by gene data matrix, (N,G)
        labels : `numpy array`
            Cell type information, (N,1)
        u0, s0 : `numpy array`, optional
            Cell-specific initial condition, (N,G)
        t0 : `numpy array`, optional
            Cell-specific initial time, (N,1)
        weight : `numpy array`, optional
            Training weight of each sample.
        """
        self.N, self.G = D.shape[0], D.shape[1]//2
        self.data = D
        self.labels = labels
        self.u0 = u0
        self.s0 = s0
        self.t0 = t0
        self.weight = np.ones((self.N, self.G)) if weight is None else weight

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.u0 is not None and self.s0 is not None and self.t0 is not None:
            return (self.data[idx],
                    self.labels[idx],
                    self.weight[idx],
                    idx,
                    self.u0[idx],
                    self.s0[idx],
                    self.t0[idx])

        return (self.data[idx],
                self.labels[idx],
                self.weight[idx],
                idx)


class SCTimedData(Dataset):
    """
    This class is almost the same as SCData. The only difference is the addition
    of cell time. This is used for training the branching ODE.
    """
    def __init__(self, D, labels, t, u0=None, s0=None, t0=None, weight=None):
        """Class constructor

        Arguments
        ---------

        D : `numpy array`
            Cell by gene data matrix, (N,G)
        labels : `numpy array`
            Cell type information, (N,1)
        t : `numpy array`
            Cell time, (N,1)
        u0, s0 : `numpy array`, optional
            Cell-specific initial condition, (N,G)
        t0 : `numpy array`, optional
            Cell-specific initial time, (N,1)
        weight : `numpy array`, optional
            Training weight of each sample.
        """
        self.N, self.G = D.shape[0], D.shape[1]//2
        self.data = D
        self.labels = labels
        self.time = t.reshape(-1, 1)
        self.u0 = u0
        self.s0 = s0
        self.t0 = t0
        self.weight = np.ones((self.N, self.G)) if weight is None else weight

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.u0 is not None and self.s0 is not None and self.t0 is not None:
            return (self.data[idx],
                    self.labels[idx],
                    self.time[idx],
                    self.weight[idx],
                    idx,
                    self.u0[idx],
                    self.s0[idx],
                    self.t0[idx])

        return (self.data[idx],
                self.labels[idx],
                self.time[idx],
                self.weight[idx],
                idx)


class SCGraphData():
    """
    This class wraps around torch_geometric.data to include graph structured datasets.
    """
    def __init__(self,
                 data,
                 labels,
                 graph,
                 xy,
                 train_idx,
                 validation_idx,
                 test_idx,
                 device,
                 batch=None,
                 enable_edge_weight=False,
                 normalize_xy=False,
                 seed=2022):
        """Constructor

        Args:
            data (:class:`numpy.ndarray`):
                Cell-by-gene count matrix. Unspliced and spliced counts are concatenated in the gene dimension.
            labels (:class:`numpy.ndarray`):
                Cell type annotation encoded in integer.
            graph (:class:`scipy.sparse.csr_matrix`):
                Cell-cell connectivity graph.
            n_train (int):
                Number of training samples.
            device (:class:`torch.device`):
                {'cpu' or 'cuda'}
            seed (int, optional):
                Random seed. Defaults to 2022.
        """
        self.N, self.G = data.shape[0], data.shape[1]//2
        self.graph = graph.A
        self.data = T.ToSparseTensor()(Data(x=torch.tensor(data,
                                                           dtype=torch.float32,
                                                           requires_grad=False),
                                            edge_index=torch.tensor(np.stack(graph.nonzero()),
                                                                    dtype=torch.long,
                                                                    requires_grad=False),
                                            y=torch.tensor(labels,
                                                           dtype=torch.int8,
                                                           requires_grad=False))).to(device)
        # Normalize spatial coordinates
        if normalize_xy:
            xy_norm = (xy - np.min(xy, 0))/(np.max(xy, 0) - np.min(xy, 0))
            self.xy = torch.tensor(xy_norm, dtype=torch.float32, device=device)
        else:
            self.xy = torch.tensor(xy, dtype=torch.float32, device=device)
        self.xy_scale = np.max(xy, 0) - np.min(xy, 0)
        
        # Batch information
        self.batch = torch.tensor(batch, dtype=int, device=device) if batch is not None else None
        if enable_edge_weight:
            self.edge_weight = torch.tensor(graph.data,
                                            dtype=torch.float32,
                                            device=device,
                                            requires_grad=False)
        else:
            self.edge_weight = None
        
        self.train_idx = torch.tensor(train_idx,
                                      dtype=torch.int32,
                                      requires_grad=False,
                                      device=device)
        self.validation_idx = torch.tensor(validation_idx,
                                           dtype=torch.int32,
                                           requires_grad=False,
                                           device=device)
        self.test_idx = torch.tensor(test_idx,
                                     dtype=torch.int32,
                                     requires_grad=False,
                                     device=device)

        self.u0 = None
        self.s0 = None
        self.t0 = None
        self.u1 = None
        self.s1 = None
        self.t1 = None
        self.xy0 = None
        self.t = None
        self.z = None

        return


class SCMultiGraphData():
    """
    This class builds a single graph dataset from multiple slices of graph datasets.
    """
    def __init__(self, data, labels, graphs, n_train, test_slices, device, train_edge_weight, seed=2022):
        """Constructor

        Args:
            data (list[:class:`numpy.ndarray`]):
                Multiple slices of cell-by-gene count matrix.
                Assumes that the gene dimension is the same for all slices.
                Unspliced and spliced counts are concatenated in the gene dimension.
            labels (list[:class:`numpy.ndarray`]):
                Cell type annotation encoded in integer.
            graphs (list[:class:`scipy.sparse.csr_matrix`]):
                Cell-cell connectivity graph.
            n_train (int):
                Number of training samples.
            test_slices (list[int]):
                List of slice indices for testing
            device (:class:`torch.device`):
                {'cpu' or 'cuda'}
            seed (int, optional):
                Random seed. Defaults to 2022.
        """
        graph_combined = self._concat_adj_mtx(graphs)
        self.data = T.ToSparseTensor()(Data(x=torch.tensor(np.stack(data),
                                                           dtype=torch.float32,
                                                           requires_grad=False),
                                            edge_index=torch.tensor(np.stack(graph_combined.nonzero()),
                                                                    dtype=torch.long,
                                                                    requires_grad=False),
                                            y=torch.tensor(np.concatenate(labels),
                                                           dtype=torch.int8,
                                                           requires_grad=False))).to(device)
        if train_edge_weight:
            self.edge_weight = torch.tensor(graph_combined.data,
                                            dtype=torch.float32,
                                            device=device,
                                            requires_grad=False)
        else:
            self.edge_weight = None
        np.random.seed(seed)
        train_valid_idx = []
        test_idx = []
        start = 0
        for i in range(len(data)):
            if i in test_slices:
                test_idx.extend(list(range(start, start+data[i].shape[0])))
            else:
                train_valid_idx.extend(list(range(start, start+data[i].shape[0])))
            start += data[i].shape[0]
        self.N, self.G = len(train_valid_idx), data[0].shape[1]//2
        rand_perm = np.random.permutation(train_valid_idx)
        self.train_idx = torch.tensor(rand_perm[:n_train],
                                      dtype=torch.int32,
                                      requires_grad=False,
                                      device=device)
        self.validation_idx = torch.tensor(rand_perm[n_train:],
                                           dtype=torch.int32,
                                           requires_grad=False,
                                           device=device)
        self.test_idx = torch.tensor(test_idx,
                                     dtype=torch.int32,
                                     requires_grad=False,
                                     device=device)
        self.n_train = n_train
        self.n_test = self.N - self.n_train

        self.u0 = None
        self.s0 = None
        self.t0 = None
        self.u1 = None
        self.s1 = None
        self.t1 = None

        return

    def _concat_adj_mtx(self, graphs):
        return sp.sparse.block_diag(graphs)


class Index(Dataset):
    """This dataset contains only indices. Used for generating
    batch indices.
    """
    def __init__(self, n_samples):
        self.index = np.array(range(n_samples))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        return self.index[idx]