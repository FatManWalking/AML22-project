import torch
import torch.nn as nn
import torch.nn.functional as F
from sparse_softmax import Sparsemax
from torch.nn.parameter import Parameter  # Path changed in newer version
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.pool import SAGPooling
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch_geometric.utils import softmax, dense_to_sparse, add_remaining_self_loops
from torch_scatter import scatter_add
from torch_sparse import spspmm, coalesce  # Coalesce is not backward compatible


class TwoHopNeighborhood(object):
    """
    Takes the data and returns the two-hop neighbors
    as in getting adjecency matrix A and returning A^2
    and when getting A^2 returning A^3
    """

    def __call__(self, data: Data):
        edge_index, edge_attr = (
            data.edge_index,
            data.edge_attr,
        )  # Get the edges from the homogenous graph represented in torch.geometric.data
        n = data.num_nodes  # Note how many nodes there are

        fill = 1e16  # previously used to prevent a overflow when this threshold was reached the edge attribute was reset to 0
        value = edge_index.new_full(
            (edge_index.size(1),), fill, dtype=torch.float)

        index, value = spspmm(
            edge_index, value, edge_index, value, n, n, n, True
        )  # Matrix product of two sparse tensors aka our adjaceny matrices

        edge_index = torch.cat(
            [edge_index, index], dim=1
        )  # Concatinating the edge indices to match otherwise you have the edges twice
        if edge_attr is None:
            data.edge_index, _ = coalesce(
                edge_index, None, n, n
            )  # If the edges have no values you can just combine the indices
        else:  # Otherwise you go through this to combine the values of the double edges
            value = value.view(-1, *[1 for _ in range(edge_attr.dim() - 1)])
            value = value.expand(-1, *list(edge_attr.size())[1:])
            edge_attr = torch.cat(
                [edge_attr, value], dim=0
            )  # Concatinating the edge attributes
            # Since the fill_value argument is missing too large neighborhoods might explode the edge attributes
            data.edge_index, edge_attr = coalesce(
                edge_index, edge_attr, n, n, op="min"
            )  # Option min calls torch.ops.torch_scatter.segment_min_csr(src, indptr, out) other options are "add", "max", and "mean" for combining the attributes
            edge_attr[
                edge_attr >= fill
            ] = 0  # This will reset the value if the edge attribute value explodes
            data.edge_attr = edge_attr  # Reset the Attribute in the original graph

        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class GCN(MessagePassing):
    def __init__(self, in_channels, out_channels, cached=False, bias=True, **kwargs):
        super(GCN, self).__init__(aggr="add", **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cached = cached
        self.cached_result = None
        self.cached_num_edges = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        # A Xavier uniform weight distribution to init the neural net is mostly more efficient then starting of a random distribution
        nn.init.xavier_uniform_(self.weight.data)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            nn.init.zeros_(self.bias.data)
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones(
                (edge_index.size(1),), dtype=dtype, device=edge_index.device
            )

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    "Cached {} number of edges, but found {}".format(
                        self.cached_num_edges, edge_index.size(1)
                    )
                )

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(
                edge_index, x.size(0), edge_weight, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )


class NodeInformationScore(MessagePassing):
    def __init__(self, improved=False, cached=False, **kwargs):
        super(NodeInformationScore, self).__init__(aggr="add", **kwargs)

        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones(
                (edge_index.size(1),), dtype=dtype, device=edge_index.device
            )

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, 0, num_nodes
        )

        row, col = edge_index
        expand_deg = torch.zeros(
            (edge_weight.size(0),), dtype=dtype, device=edge_index.device
        )
        expand_deg[-num_nodes:] = torch.ones(
            (num_nodes,), dtype=dtype, device=edge_index.device
        )

        return (
            edge_index,
            expand_deg - deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col],
        )

    def forward(self, x, edge_index, edge_weight):
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    "Cached {} number of edges, but found {}".format(
                        self.cached_num_edges, edge_index.size(1)
                    )
                )

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(
                edge_index, x.size(0), edge_weight, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out


class SAGPool(torch.nn.pool.SAGPooling):
    def __init__(
        self,
        in_channels,
        ratio=0.8,
    ):
        super(SAGPooling, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio

    def forward(self, x, edge_index, edge_attr, batch=None):
        x, edge_index, edge_attr, batch, _, _ = super(
            SAGPooling, self).forward(x, edge_index, edge_attr, batch)

        return x, edge_index, edge_attr, batch
