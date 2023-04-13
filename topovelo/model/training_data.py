import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
# from torch_geometric.loader import NeighborLoader


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
    def __init__(self, data, labels, graph, n_train, device, seed=2022):
        self.N, self.G = data.shape[0], data.shape[1]//2
        self.data = Data(x=torch.tensor(data, dtype=torch.float32),
                         edge_index=torch.tensor(np.stack(graph.nonzero()), dtype=torch.long),
                         y=torch.tensor(labels, dtype=torch.long)).to(device)
        self.edge_weight=torch.tensor(graph.data,
                                      dtype=torch.float32,
                                      device=device,
                                      requires_grad=False)
        np.random.seed(seed)
        rand_perm = np.random.permutation(self.N)
        self.train_idx = torch.tensor(rand_perm[:n_train], dtype=torch.long).to(device)
        self.test_idx = torch.tensor(rand_perm[n_train:], dtype=torch.long).to(device)
        self.n_train = n_train
        self.n_test = self.N - self.n_train

        self.u0 = None
        self.s0 = None
        self.t0 = None
        self.u1 = None
        self.s1 = None
        self.t1 = None

        return


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