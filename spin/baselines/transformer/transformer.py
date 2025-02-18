import math
from argparse import ArgumentParser as ArgParser
from typing import List, Optional, Union

import torch
from torch import Tensor, nn
from torch_geometric.nn import inits
from torch_geometric.typing import OptTensor

# from tsl.nn.base import StaticGraphEmbedding
from tsl.nn.blocks.encoders.mlp import MLP
from tsl.nn.blocks.encoders.transformer import (
    SpatioTemporalTransformerLayer,
    TransformerLayer,
)
from tsl.nn.layers import PositionalEncoding
from tsl.utils.parser_utils import str_to_bool
from icecream import ic


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


class TransformerModel(nn.Module):
    r"""Spatiotemporal Transformer for multivariate time series imputation.

    Args:
        input_size (int): Input size.
        hidden_size (int): Dimension of the learned representations.
        output_size (int): Dimension of the output.
        ff_size (int): Units in the MLP after self attention.
        u_size (int): Dimension of the exogenous variables.
        n_heads (int, optional): Number of parallel attention heads.
        n_layers (int, optional): Number of layers.
        dropout (float, optional): Dropout probability.
        axis (str, optional): Dimension on which to apply attention to update
            the representations.
        activation (str, optional): Activation function.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        ff_size: int,
        u_size: int,
        n_heads: int = 1,
        n_layers: int = 1,
        dropout: float = 0.0,
        condition_on_u: bool = True,
        axis: str = "both",
        activation: str = "elu",
    ):
        super(TransformerModel, self).__init__()

        self.condition_on_u = condition_on_u
        if condition_on_u:
            self.u_enc = MLP(u_size, hidden_size, n_layers=2)
        self.h_enc = MLP(input_size, hidden_size, n_layers=2)

        self.mask_token = StaticGraphEmbedding(1, hidden_size)

        self.pe = PositionalEncoding(hidden_size)

        kwargs = dict(
            input_size=hidden_size,
            hidden_size=hidden_size,
            ff_size=ff_size,
            n_heads=n_heads,
            activation=activation,
            causal=False,
            dropout=dropout,
        )

        if axis in ["steps", "nodes"]:
            transformer_layer = TransformerLayer
            kwargs["axis"] = axis
        elif axis == "both":
            transformer_layer = SpatioTemporalTransformerLayer
        else:
            raise ValueError(f'"{axis}" is not a valid axis.')

        self.encoder = nn.ModuleList()
        self.readout = nn.ModuleList()
        for _ in range(n_layers):
            self.encoder.append(transformer_layer(**kwargs))
            self.readout.append(
                MLP(
                    input_size=hidden_size,
                    hidden_size=ff_size,
                    output_size=output_size,
                    n_layers=2,
                    dropout=dropout,
                )
            )

    def forward(self, x, u, mask):
        # x: [batches steps nodes features]
        # u: [batches steps (nodes) features]
        # ic(x.shape)
        # ic(mask.shape)
        x = x * mask

        h = self.h_enc(x)
        # ic(h.shape)
        h = mask * h + (1 - mask) * self.mask_token()
        # ic(h.shape)
        # ic(self.mask_token().shape)
        # exit()

        if self.condition_on_u:
            h = h + self.u_enc(u).unsqueeze(-2)

        h = self.pe(h)

        out = []
        for encoder, mlp in zip(self.encoder, self.readout):
            h = encoder(h)
            out.append(mlp(h))

        x_hat = out.pop(-1)
        return x_hat, out

    @staticmethod
    def add_model_specific_args(parser: ArgParser):
        parser.opt_list(
            "--hidden-size",
            type=int,
            default=32,
            tunable=True,
            options=[16, 32, 64, 128, 256],
        )
        parser.opt_list(
            "--ff-size",
            type=int,
            default=32,
            tunable=True,
            options=[32, 64, 128, 256, 512, 1024],
        )
        parser.opt_list(
            "--n-layers", type=int, default=1, tunable=True, options=[1, 2, 3]
        )
        parser.opt_list(
            "--n-heads", type=int, default=1, tunable=True, options=[1, 2, 3]
        )
        parser.opt_list(
            "--dropout",
            type=float,
            default=0.0,
            tunable=True,
            options=[0.0, 0.1, 0.25, 0.5],
        )
        parser.add_argument(
            "--condition-on-u", type=str_to_bool, nargs="?", const=True, default=True
        )
        parser.opt_list(
            "--axis", type=str, default="both", tunable=True, options=["steps", "both"]
        )
        return parser
