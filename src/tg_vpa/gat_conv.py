from typing import Callable, Optional
import torch
from torch import Tensor
from torch_geometric.nn.aggr import SumAggregation
from torch_geometric.nn.conv import GATConv


class GATConv(GATConv):

    '''
    Adaptation of torch-geometric's GATConv layer to variance preserving aggregation.
    '''
    
    def __init__(self, in_channels: int, out_channels: int, aggr: str, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(in_channels, out_channels, **kwargs)

        self.aggr = aggr
        if self.aggr in ['default', 'sum', 'add', 'vpa', 'vpp', 'vp']:
            self.aggr_module = SumAggregation()
        else:
            raise NotImplementedError('Invalid aggregation function.')

    def aggregate(self, inputs: Tensor, index: Tensor, alpha: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None,
                  ) -> Tensor:
        r"""Aggregates messages from neighbors as
        :math:`\bigoplus_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to the underlying
        :class:`~torch_geometric.nn.aggr.Aggregation` module to reduce messages
        as specified in :meth:`__init__` by the :obj:`aggr` argument.
        """

        if self.aggr in ['vpa', 'vpp', 'vp']:
            with torch.no_grad():
                # normalization factor for variance preserving aggregation in GAT
                factor = torch.sqrt(self.aggr_module(alpha.pow(2), index, ptr=ptr, dim_size=dim_size, dim=self.node_dim))
                factor = factor.unsqueeze(-1)
            return self.aggr_module(inputs, index, ptr=ptr, dim_size=dim_size, dim=self.node_dim)/factor
        elif self.aggr in ['default', 'sum', 'add']:
            return self.aggr_module(inputs, index, ptr=ptr, dim_size=dim_size, dim=self.node_dim)