import numpy as np
from torch.utils.data import Dataset
from .model_util import get_neigh_index

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


class SCGraphData(Dataset):
    """
    This class wraps around SCData to include graph structured datasets.
    In particular, it contains a train-test split of nodes (cells), while
    keeping all the edges.
    """
    def __init__(self, data, labels, graph, n_train, u0=None, s0=None, t0=None, weight=None, seed=2022):
        self.N, self.G = data.shape[0], data.shape[1]//2
        np.random.seed(seed)
        n = data.shape[0]
        self.node_features = data
        rand_perm = np.random.permutation(n)
        self.train_idx = rand_perm[:n_train]
        self.test_idx = rand_perm[n_train:]

        self.neighbor_indices, self.degrees, self.edge_weights = get_neigh_index(graph)
        self.k = self.neighbor_indices.shape[1]
        self.u0 = u0
        self.s0 = s0
        self.t0 = t0
        self.weight = np.ones((n_train, self.G)) if weight is None else weight
        return

    def __len__(self):
        return len(self.node_features)

    def __getitem__(self, idx):
        sample_idx = self.train_idx[idx]
        neigh_idx = self.neighbor_indices[sample_idx]
        if self.u0 is not None and self.s0 is not None and self.t0 is not None:
            return (self.data[sample_idx],
                    self.data[neigh_idx, :],
                    self.labels[sample_idx],
                    self.weight[sample_idx],
                    sample_idx,
                    self.u0[sample_idx],
                    self.s0[sample_idx],
                    self.t0[sample_idx])

        return (self.data[sample_idx],
                self.data[neigh_idx, :],
                self.labels[sample_idx],
                self.weight[sample_idx],
                sample_idx)