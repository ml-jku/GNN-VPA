from typing import Callable, Optional

from torch import Tensor

from torch_geometric.nn.conv import GraphConv
from torch_geometric.nn.aggr import SumAggregation, MeanAggregation, MaxAggregation
from src.agg import VariancePreservingAggregation


class GraphConv(GraphConv):

    '''
    Adaptation of torch-geometric's GraphConv layer to variance preserving aggregation.
    '''

    def __init__(self, in_channels: int, out_channels: int, aggr: str, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(in_channels, out_channels, **kwargs)

        self.aggr = aggr
        if self.aggr in ['sum', 'add']:
            self.aggr_module = SumAggregation()
        elif self.aggr in ['mean', 'average']:
            self.aggr_module = MeanAggregation()
        elif self.aggr in ['vpa', 'vpp', 'vp']:
            self.aggr_module = VariancePreservingAggregation()
        elif self.aggr == 'max':
            self.aggr_module = MaxAggregation()
        else:
            raise NotImplementedError('Invalid aggregation function.')

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