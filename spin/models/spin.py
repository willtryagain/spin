import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import LayerNorm
from torch_geometric.nn import inits
from torch_geometric.typing import List, OptTensor, Union
from tsl.nn.blocks.encoders import MLP
from tsl.nn.layers import PositionalEncoding

from ..layers import PositionalEncoder, TemporalGraphAdditiveAttention


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


class SPINModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_nodes: int,
        u_size: Optional[int] = None,
        output_size: Optional[int] = None,
        temporal_self_attention: bool = True,
        reweight: Optional[str] = "softmax",
        n_layers: int = 4,
        eta: int = 3,
        message_layers: int = 1,
    ):
        super(SPINModel, self).__init__()

        u_size = u_size or input_size
        output_size = output_size or input_size
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.eta = eta
        self.temporal_self_attention = temporal_self_attention

        self.u_enc = PositionalEncoder(
            in_channels=u_size, out_channels=hidden_size, n_layers=2, n_nodes=n_nodes
        )

        self.h_enc = MLP(input_size, hidden_size, n_layers=2)
        self.h_norm = LayerNorm(hidden_size)

        self.valid_emb = StaticGraphEmbedding(n_nodes, hidden_size)
        self.mask_emb = StaticGraphEmbedding(n_nodes, hidden_size)

        self.x_skip = nn.ModuleList()
        self.encoder, self.readout = nn.ModuleList(), nn.ModuleList()
        for l in range(n_layers):
            x_skip = nn.Linear(input_size, hidden_size)
            encoder = TemporalGraphAdditiveAttention(
                input_size=hidden_size,
                output_size=hidden_size,
                msg_size=hidden_size,
                msg_layers=message_layers,
                temporal_self_attention=temporal_self_attention,
                reweight=reweight,
                mask_temporal=True,
                mask_spatial=l < eta,
                norm=True,
                root_weight=True,
                dropout=0.0,
            )
            readout = MLP(hidden_size, hidden_size, output_size, n_layers=2)
            self.x_skip.append(x_skip)
            self.encoder.append(encoder)
            self.readout.append(readout)

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
        if target_nodes is None:
            target_nodes = slice(None)

        # Whiten missing values
        x = x * mask

        # POSITIONAL ENCODING #################################################
        # Obtain spatio-temporal positional encoding for every node-step pair #
        # in both observed and target sets. Encoding are obtained by jointly  #
        # processing node and time positional encoding.                       #

        # Build (node, timestamp) encoding
        q = self.u_enc(u, node_index=node_index)
        # Condition value on key
        h = self.h_enc(x) + q

        # ENCODER #############################################################
        # Obtain representations h^i_t for every (i, t) node-step pair by     #
        # only taking into account valid data in representation set.          #

        # Replace H in missing entries with queries Q
        h = torch.where(mask.bool(), h, q)
        # Normalize features
        h = self.h_norm(h)

        imputations = []

        for l in range(self.n_layers):
            if l == self.eta:
                # Condition H on two different embeddings to distinguish
                # valid values from masked ones
                valid = self.valid_emb(token_index=node_index)
                masked = self.mask_emb(token_index=node_index)
                h = torch.where(mask.bool(), h + valid, h + masked)
            # Masked Temporal GAT for encoding representation
            h = h + self.x_skip[l](x) * mask  # skip connection for valid x
            h = self.encoder[l](h, edge_index, mask=mask)
            # Read from H to get imputations
            target_readout = self.readout[l](h[..., target_nodes, :])
            imputations.append(target_readout)

        # Get final layer imputations
        x_hat = imputations.pop(-1)

        return x_hat, imputations

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument(
            "--hidden-size",
            type=int,
            default=32,
            choices=[32, 64, 128, 256],
        )
        parser.add_argument("--u-size", type=int, default=None)
        parser.add_argument("--output-size", type=int, default=None)
        parser.add_argument("--temporal-self-attention", type=bool, default=True)
        parser.add_argument("--reweight", type=str, default="softmax")
        parser.add_argument("--n-layers", type=int, default=4)
        parser.add_argument("--eta", type=int, default=3)
        parser.add_argument("--message-layers", type=int, default=1)
        return parser
