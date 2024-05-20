import numpy as np
import scipy as sp
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch_geometric.transforms as T
# from torch_geometric.loader import NeighborLoader


class SCData(Dataset):
    """This is a simple pytorch dataset class for batch training.
    Each sample represents a cell. Each dimension represents a single gene.
    The dataset also contains the cell labels (types).
    """
    def __init__(self, D, labels, u0=None, s0=None, t0=None, weight=None):
        """Class constructor

        Args:
            D (:class:`numpy.ndarray`):
                Cell-by-gene count matrix.
                Unspliced and spliced counts are concatenated in the gene dimension.
            labels (:class:`numpy.ndarray`):
                Cell type annotation encoded in integer.
            u0 (:class:`numpy.ndarray`, optional):
                Cell-specific initial unspliced count. Defaults to None.
            s0 (:class:`numpy.ndarray`, optional):
                Cell-specific initial spliced count. Defaults to None.
            t0 (:class:`numpy.ndarray`, optional):
                Cell-specific initial time. Defaults to None.
            weight (:class:`numpy.ndarray`, optional):
                Training weight of each sample. Defaults to None.
        """
        self.N, self.G = D.shape[0], D.shape[1]//2
        self.data = D
        self.labels = labels
        self.u0 = u0
        self.s0 = s0
        self.t0 = t0
        self.weight = None if weight is None else weight

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

        Args:
            D (:class:`numpy.ndarray`):
                Cell-by-gene count matrix.
                Unspliced and spliced counts are concatenated in the gene dimension.
            labels (:class:`numpy.ndarray`):
                Cell type annotation encoded in integer.
            t (:class:`numpy.ndarray`):
                Cell time.
            u0 (:class:`numpy.ndarray`, optional):
                Cell-specific initial unspliced count. Defaults to None.
            s0 (:class:`numpy.ndarray`, optional):
                Cell-specific initial spliced count. Defaults to None.
            t0 (:class:`numpy.ndarray`, optional):
                Cell-specific initial time. Defaults to None.
            weight (:class:`numpy.ndarray`, optional):
                Training weight of each sample. Defaults to None.
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
    """This class wraps around torch_geometric.data to include graph structured datasets.
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
                 edge_attr=None,
                 batch=None,
                 enable_edge_weight=False,
                 normalize_xy=False):
        """Construct a graph dataset.

        Args:
            data (:class:`numpy.ndarray`):
                Cell-by-gene count matrix.
                Unspliced and spliced counts are concatenated in the gene dimension.
            labels (:class:`numpy.ndarray`):
                Cell type annotation encoded in integer.
            graph (:class:`scipy.sparse.csr_matrix`):
                Cell-cell connectivity graph.
            xy (:class:`numpy.ndarray`):
                Spatial coordinates of cells.
            train_idx (:class:`numpy.ndarray`):
                Indices of training samples.
            validation_idx (:class:`numpy.ndarray`):
                Indices of validation samples.
            test_idx (:class:`numpy.ndarray`):
                Indices of test samples.
            device (:class:`torch.device`):
                {'cpu' or 'cuda'}
            edge_attr (:class:`numpy.ndarray`, optional):
                Edge attributes of shape (|E|,|E|,dim_edge_attr). Defaults to None.
            batch (:class:`numpy.ndarray`, optional):
                Batch indices of cells. Defaults to None.
            enable_edge_weight (bool, optional):
                Whether to use edge weight in the graph. Defaults to False.
            normalize_xy (bool, optional):
                Whether to normalize spatial coordinates. Defaults to False.
        """
        self.N, self.G = data.shape[0], data.shape[1]//2
        self.graph = graph.A
        self.edges = np.stack(graph.nonzero())
        if edge_attr is not None:
            edge_attr_ = self._extract_edge_attr(edge_attr)
        else:
            edge_attr_ = None

        self.data = T.ToSparseTensor()(Data(x=torch.tensor(data,
                                                           dtype=torch.float32,
                                                           requires_grad=False),
                                            edge_index=torch.tensor(self.edges,
                                                                    dtype=torch.long,
                                                                    requires_grad=False),
                                            edge_attr=edge_attr_,
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
        
        self.train_idx = None
        self.validation_idx = None
        self.test_idx = None
        if train_idx is not None:
            self.train_idx = torch.tensor(train_idx,
                                          dtype=torch.int32,
                                          requires_grad=False,
                                          device=device)
        if validation_idx is not None:
            self.validation_idx = torch.tensor(validation_idx,
                                               dtype=torch.int32,
                                               requires_grad=False,
                                               device=device)
        if test_idx is not None:
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
    
    def _extract_edge_attr(self, edge_attr):
        """Extract edge attributes from the graph.

        Args:
            edge_attr (:class:`numpy.ndarray`):
                Edge attributes.

        Returns:
            :class:`torch.tensor`: Edge attributes.
        """            
        edge_attr_ = np.zeros((self.edges.shape[1], edge_attr.shape[-1]))
        ptr = 0
        prev = 0
        # edge case: there is only one edge
        if self.edges.shape[1] == 1:
            return torch.tensor(edge_attr, dtype=torch.float32, requires_grad=False)
        # general case
        while ptr < self.edges.shape[1]:
            while self.edges[0, ptr] == self.edges[0, ptr+1]:
                ptr += 1
                if ptr == self.edges.shape[1]-1:
                    break
            edge_attr_[prev:ptr+1] = edge_attr[self.edges[0, ptr], self.edges[1, prev:ptr+1]]
            prev = ptr+1
        return torch.tensor(edge_attr, dtype=torch.float32, requires_grad=False)


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