from typing import Optional

import torch
from torch import Tensor
from torch_geometric.nn.aggr import Aggregation


class VariancePreservingAggregation(Aggregation):
    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -1,
    ) -> Tensor:
        sorted_indices, argsort = torch.sort(index)
        sorted_x = x[argsort]

        sum_aggregation = self.reduce(sorted_x, sorted_indices, ptr, dim_size, dim, reduce="sum")
        counts = self.reduce(torch.ones_like(x), sorted_indices, ptr, dim_size, dim, reduce="sum")

        return torch.nan_to_num(sum_aggregation / torch.sqrt(counts))

