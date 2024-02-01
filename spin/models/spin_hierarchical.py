import math
from typing import Optional

import torch
from icecream import ic
from torch import Tensor, nn
from torch.distributed.fsdp.wrap import wrap
from torch.nn import LayerNorm
from torch_geometric.nn import inits
from torch_geometric.typing import List, OptTensor, Union
from tsl.nn.blocks.encoders import MLP
from tsl.nn.layers import PositionalEncoding

from ..layers import DiffPool, HierarchicalTemporalGraphAttention, PositionalEncoder


class StaticGraphEmbedding(nn.Module):
    r"""Creates a table of embeddings with the specified size.

    Args:
        n_tokens (int): Number of elements for which to store an embedding.
        emb_size (int): Size of the embedding.
        initializer (str or Tensor): Initialization methods.
            (default :obj:`'uniform'`)
        requires_grad (bool): Whether to compute gradients for the embeddings.
            (default :obj:`True`)
        bind_to (nn.Module, optional): Bind the embedding to a nn.Module for
            lazy init. (default :obj:`None`)
        infer_tokens_from_pos (int): Index of the element of input data from
            which to infer the number of embeddings for lazy init.
            (default :obj:`0`)
        dim (int): Token dimension. (default :obj:`-2`)
    """

    def __init__(
        self,
        n_tokens: int,
        emb_size: int,
        initializer: Union[str, Tensor] = "uniform",
        requires_grad: bool = True,
        bind_to: Optional[nn.Module] = None,
        infer_tokens_from_pos: int = 0,
        dim: int = -2,
    ):
        super(StaticGraphEmbedding, self).__init__()
        assert emb_size > 0
        self.n_tokens = int(n_tokens)
        self.emb_size = int(emb_size)
        self.dim = int(dim)
        self.infer_tokens_from_pos = infer_tokens_from_pos

        if isinstance(initializer, Tensor):
            self.initializer = "from_values"
            self.register_buffer("_default_values", initializer.float())
        else:
            self.initializer = initializer
            self.register_buffer("_default_values", None)

        if self.n_tokens > 0:
            self.emb = nn.Parameter(
                Tensor(self.n_tokens, self.emb_size), requires_grad=requires_grad
            )
        else:
            assert isinstance(bind_to, nn.Module)
            self.emb = nn.parameter.UninitializedParameter(requires_grad=requires_grad)
            bind_to._hook = bind_to.register_forward_pre_hook(
                self.initialize_parameters
            )

        self.reset_parameters()

    def reset_parameters(self):
        if self.n_tokens > 0:
            if self.initializer == "from_values":
                self.emb.data = self._default_values.data
            if self.initializer == "glorot":
                inits.glorot(self.emb)
            elif self.initializer == "uniform" or self.initializer is None:
                inits.uniform(self.emb_size, self.emb)
            elif self.initializer == "kaiming_normal":
                nn.init.kaiming_normal_(self.emb, nonlinearity="relu")
            elif self.initializer == "kaiming_uniform":
                inits.kaiming_uniform(self.emb, fan=self.emb_size, a=math.sqrt(5))
            else:
                raise RuntimeError(
                    f"Embedding initializer '{self.initializer}'" " is not supported"
                )

    def extra_repr(self) -> str:
        return f"n_tokens={self.n_tokens}, embedding_size={self.emb_size}"

    @torch.no_grad()
    def initialize_parameters(self, module, input):
        if isinstance(self.emb, torch.nn.parameter.UninitializedParameter):
            self.n_tokens = input[self.infer_tokens_from_pos].size(self.dim)
            self.emb.materialize((self.n_tokens, self.emb_size))
            self.reset_parameters()
        module._hook.remove()
        delattr(module, "_hook")

    def forward(
        self,
        expand: Optional[List] = None,
        token_index: OptTensor = None,
        tokens_first: bool = True,
    ):
        """"""
        emb = self.emb if token_index is None else self.emb[token_index]
        if not tokens_first:
            emb = emb.T
        if expand is None:
            return emb
        shape = [*emb.size()]
        view = [1 if d > 0 else shape.pop(0 if tokens_first else -1) for d in expand]
        return emb.view(*view).expand(*expand)


class SPINHierarchicalModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        h_size: int,
        z_size: int,
        n_nodes: int,
        z_heads: int = 1,
        u_size: Optional[int] = None,
        output_size: Optional[int] = None,
        n_layers: int = 5,
        eta: int = 3,
        message_layers: int = 1,
        reweight: Optional[str] = "softmax",
        update_z_cross: bool = True,
        norm: bool = True,
        spatial_aggr: str = "add",
    ):
        super(SPINHierarchicalModel, self).__init__()
        dont_pool = False
        self.dont_pool = dont_pool
        u_size = u_size or input_size
        output_size = output_size or input_size
        self.h_size = h_size
        self.z_size = z_size
        self.og_nodes = n_nodes
        if dont_pool:
            pass
        else:
            n_nodes = (2 * n_nodes) // 3
        steps = 24
        self.shrink_pool = DiffPool(steps, input_size, n_nodes)
        self.expand_pool = DiffPool(steps, input_size, self.og_nodes)
        self.n_nodes = n_nodes
        self.z_heads = z_heads
        self.n_layers = n_layers
        self.eta = eta

        self.v = StaticGraphEmbedding(n_nodes, h_size)
        self.lin_v = nn.Linear(h_size, z_size, bias=False)
        self.z = nn.Parameter(torch.Tensor(1, z_heads, n_nodes, z_size))
        inits.uniform(z_size, self.z)
        self.z_norm = LayerNorm(z_size)

        self.u_enc = PositionalEncoder(
            in_channels=u_size, out_channels=h_size, n_layers=2
        )

        self.h_enc = MLP(input_size, h_size, n_layers=2)
        self.h_norm = LayerNorm(h_size)

        self.v1 = StaticGraphEmbedding(n_nodes, h_size)
        self.m1 = StaticGraphEmbedding(n_nodes, h_size)

        self.v2 = StaticGraphEmbedding(n_nodes, h_size)
        self.m2 = StaticGraphEmbedding(n_nodes, h_size)

        self.x_skip = nn.ModuleList()
        self.encoder, self.readout = nn.ModuleList(), nn.ModuleList()
        for l in range(n_layers):
            x_skip = nn.Linear(input_size, h_size)
            encoder = HierarchicalTemporalGraphAttention(
                h_size=h_size,
                z_size=z_size,
                msg_size=h_size,
                msg_layers=message_layers,
                reweight=reweight,
                mask_temporal=True,
                mask_spatial=l < eta,
                update_z_cross=update_z_cross,
                norm=norm,
                root_weight=True,
                aggr=spatial_aggr,
                dropout=0.0,
            )
            readout = MLP(h_size, z_size, output_size, n_layers=2)
            self.x_skip.append(x_skip)
            self.encoder.append(encoder)
            self.readout.append(readout)

    def pool_data(self, pool, x, edge_index, mask, num_nodes):
        assign_mat = pool(x, edge_index)
        assign_mat = assign_mat.mean(0)
        assign_mat = nn.Softmax(dim=1)(assign_mat)
        x = torch.einsum("bsnc, nm -> bsmc", x, assign_mat)
        adj = torch.zeros(num_nodes, num_nodes).to(x.device)
        adj[edge_index[0], edge_index[1]] = 1
        adj_new = assign_mat.T @ adj @ assign_mat
        # to binary adjacency matrix
        threshold = 0.5
        adj_new[adj_new > threshold] = 1
        adj_new[adj_new <= threshold] = 0
        adj_new = adj_new.to(x.device)
        edge_index = torch.nonzero(adj_new).T

        mask = mask.to(torch.float16)
        mask = torch.einsum("bsnc, nm -> bsmc", mask, assign_mat)
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        mask = mask.to(x.device)

        return x, edge_index, mask

    def forward(
        self,
        x: Tensor,
        u: Tensor,
        mask: Tensor,
        edge_index: Tensor,
        edge_weight: OptTensor = None,
        node_index: OptTensor = None,
        target_nodes: OptTensor = None,
    ):
        if self.dont_pool:
            pass
        else:
            x, edge_index, mask = self.pool_data(
                self.shrink_pool, x, edge_index, mask, self.og_nodes
            )

        if target_nodes is None:
            target_nodes = slice(None)
        if node_index is None:
            node_index = slice(None)

        # POSITIONAL ENCODING #################################################
        # Obtain spatio-temporal positional encoding for every node-step pair #
        # in both observed and target sets. Encoding are obtained by jointly  #
        # processing node and time positional encoding.                       #
        # Condition also embeddings Z on V.                                   #
        v_nodes = self.v(token_index=node_index)
        z = self.z[..., node_index, :] + self.lin_v(v_nodes)

        # Build (node, timestamp) encoding
        q = self.u_enc(u, node_index=node_index, node_emb=v_nodes)
        # Condition value on key
        h = self.h_enc(x) + q

        # ENCODER #############################################################
        # Obtain representations h^i_t for every (i, t) node-step pair by     #
        # only taking into account valid data in representation set.          #

        # Replace H in missing entries with queries Q. Then, condition H on two
        # different embeddings to distinguish valid values from masked ones.
        h = torch.where(mask.bool(), h + self.v1(), q + self.m1())
        # Normalize features
        h, z = self.h_norm(h), self.z_norm(z)

        imputations = []

        for l in range(self.n_layers):
            if l == self.eta:
                # Condition H on two different embeddings to distinguish
                # valid values from masked ones
                h = torch.where(mask.bool(), h + self.v2(), h + self.m2())
            # Skip connection from input x
            h = h + self.x_skip[l](x) * mask
            # Masked Temporal GAT for encoding representation
            h, z = self.encoder[l](h, z, edge_index, mask=mask)
            target_readout = self.readout[l](h[..., target_nodes, :])

            if self.dont_pool:
                pass
            else:
                target_readout, _, _ = self.pool_data(
                    self.expand_pool, target_readout, edge_index, mask, self.n_nodes
                )

            imputations.append(target_readout)

        x_hat = imputations.pop(-1)
        breakpoint()
        return x_hat, imputations

    @staticmethod
    def add_model_specific_args(parser):
        parser.opt_list(
            "--h-size", type=int, tunable=True, default=32, options=[16, 32]
        )
        parser.opt_list(
            "--z-size", type=int, tunable=True, default=32, options=[32, 64, 128]
        )
        parser.opt_list(
            "--z-heads", type=int, tunable=True, default=2, options=[1, 2, 4, 6]
        )
        parser.add_argument("--u-size", type=int, default=None)
        parser.add_argument("--output-size", type=int, default=None)
        parser.opt_list(
            "--encoder-layers", type=int, tunable=True, default=2, options=[1, 2, 3, 4]
        )
        parser.opt_list(
            "--decoder-layers", type=int, tunable=True, default=2, options=[1, 2, 3, 4]
        )
        parser.add_argument("--message-layers", type=int, default=1)
        parser.opt_list(
            "--reweight",
            type=str,
            tunable=True,
            default="softmax",
            options=[None, "softmax"],
        )
        parser.add_argument("--update-z-cross", type=bool, default=True)
        parser.opt_list(
            "--norm", type=bool, default=True, tunable=True, options=[True, False]
        )
        parser.opt_list(
            "--spatial-aggr",
            type=str,
            tunable=True,
            default="add",
            options=["add", "softmax"],
        )
        return parser
