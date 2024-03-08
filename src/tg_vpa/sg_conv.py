import torch

from typing import Callable, Optional
from torch_geometric.typing import Adj, OptTensor, SparseTensor, torch_sparse
from torch import Tensor

from torch_geometric.nn.conv import SGConv
from torch_geometric.nn.aggr import SumAggregation, MeanAggregation, MaxAggregation
from src.agg import VariancePreservingAggregation

from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import add_self_loops as add_self_loops_fn
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    spmm,
    to_edge_index,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import set_sparse_value


def gcn_norm(
    edge_index: Adj,
    edge_weight: OptTensor = None,
    num_nodes: Optional[int] = None,
    improved: bool = False,
    add_self_loops: bool = True,
    flow: str = "source_to_target",
    dtype: Optional[torch.dtype] = None,
    aggr: str = "mean"
):

    '''
    Adaptation of gcn_norm function from torch-geometric's SGConv layer to variance preserving aggregation.
    '''

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        assert edge_index.size(0) == edge_index.size(1)

        adj_t = edge_index

        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = torch_sparse.fill_diag(adj_t, fill_value)

        deg = torch_sparse.sum(adj_t, dim=1)
        if aggr == ['default', 'mean', 'average']:
            # original SGC version with degree normalization, approximatly equivalent to mean aggregation
            deg_inv_sqrt = deg.pow_(-0.5)
        elif aggr in ['sum', 'add']:
            # version without degree normalization, equivalent to sum aggregation
            deg_inv_sqrt = deg.pow_(0)
        elif aggr in ['vpa', 'vpp', 'vp']:
            # variance preserving version of degree normalization
            deg_inv_sqrt = deg.pow_(-0.25)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(1, -1))

        return adj_t

    if is_torch_sparse_tensor(edge_index):
        assert edge_index.size(0) == edge_index.size(1)

        if edge_index.layout == torch.sparse_csc:
            raise NotImplementedError("Sparse CSC matrices are not yet "
                                      "supported in 'gcn_norm'")

        adj_t = edge_index
        if add_self_loops:
            adj_t, _ = add_self_loops_fn(adj_t, None, fill_value, num_nodes)

        edge_index, value = to_edge_index(adj_t)
        col, row = edge_index[0], edge_index[1]

        deg = scatter(value, col, 0, dim_size=num_nodes, reduce='sum')
        if aggr == 'default':
            deg_inv_sqrt = deg.pow_(-0.5)
        elif aggr in ['mean', 'average']:
            deg_inv_sqrt = deg.pow_(-0.5)
        elif aggr in ['sum', 'add']:
            deg_inv_sqrt = deg.pow_(0)
        elif aggr in ['vpa', 'vpp', 'vp']:
            deg_inv_sqrt = deg.pow_(-0.25)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        value = deg_inv_sqrt[row] * value * deg_inv_sqrt[col]

        return set_sparse_value(adj_t, value), None

    assert flow in ['source_to_target', 'target_to_source']
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == 'source_to_target' else row
    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce='sum')
    if aggr == 'default':
        deg_inv_sqrt = deg.pow_(-0.5)
    elif aggr in ['mean', 'average']:
        deg_inv_sqrt = deg.pow_(-0.5)
    elif aggr in ['sum', 'add']:
        deg_inv_sqrt = deg.pow_(0)
    elif aggr in ['vpa', 'vpp', 'vp']:
        deg_inv_sqrt = deg.pow_(-0.25)
    else:
        raise NotImplementedError('Invalid aggregation function.')

    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_weight


class SGConv(SGConv):

    '''
    Adaptation of gcn_norm function from torch-geometric's SGConv layer to variance preserving aggregation.
    '''

    def __init__(self, in_channels: int, out_channels: int, K: int, aggr: str, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(in_channels, out_channels, **kwargs)

        self.aggr = aggr
        self.aggr_module = SumAggregation()


    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        cache = self._cached_x
        if cache is None:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = gcn_norm(
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, self.flow, dtype=x.dtype, aggr=self.aggr)
            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm( 
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, self.flow, dtype=x.dtype, aggr=self.aggr)

            for k in range(self.K):
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight)
                if self.cached:
                    self._cached_x = x
        else:
            x = cache.detach()

        return self.lin(x)


    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        r"""Aggregates messages from neighbors as
        :math:`\bigoplus_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to the underlying
        :class:`~torch_geometric.nn.aggr.Aggregation` module to reduce messages
        as specified in :meth:`__init__` by the :obj:`aggr` argument.
        """
 
        return self.aggr_module(inputs, index, ptr=ptr, dim_size=dim_size, dim=self.node_dim)