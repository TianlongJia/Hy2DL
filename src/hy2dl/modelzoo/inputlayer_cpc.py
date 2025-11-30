from itertools import chain
from typing import Optional

import torch
import torch.nn as nn

from hy2dl.utils.config import Config


class InputLayer_CPC(nn.Module):
    """Input layer to preprocess static and dynamic inputs.

    This layer prepares the data before passing it to the main models. This can include running the dynamic and static
    attributes through embedding networks, preprocessing and assembling data at different temporal frequencies (e.g.
    daily, hourly), doing probabilistic masking and handling missing data.

    In the simplest case, the layer takes the dictionary containing the sample information and assembles the tensor to
    be sent to the main model.

    Parameters
    ----------
    cfg : Config
        Configuration file.
    embedding_type : str
        Type of embedding to use (hindcast or forecast).

    """

    def __init__(self, cfg: Config, embedding_type: str = "hindcast"):
        super().__init__()

        self.embedding_type = embedding_type
        if embedding_type == "hindcast":
            self.dynamic_input = cfg.dynamic_input
            self._x_d_key = "x_d"
        elif embedding_type == "forecast":
            self.dynamic_input = cfg.forecast_input
            self._x_d_key = "x_d_fc"
        else:
            raise ValueError("embedding_type must be either 'hindcast' or 'forecast'")
        
        if cfg.CPC_embedding is None:
            raise NotImplementedError(f"{cfg.CPC_embedding} not defined")
        
        # Get dynamic input size
        dynamic_input_size = len(self.dynamic_input)

        # Get static input size
        static_input_size = len(cfg.static_input) if cfg.static_input else 0

        # # Get embedding networks
        # self._get_embeddings(cfg)

        self.total_input_size = dynamic_input_size + static_input_size

        # ---------- One embedding layer for all inputs ----------
        self.emb_x = InputLayer_CPC.build_embedding(
            input_dim=self.total_input_size,
            embedding=cfg.CPC_embedding
        )

        # ---------- final output size ----------
        self.output_size = cfg.CPC_embedding["hiddens"][-1]

        # Save config
        self.cfg = cfg

    def forward(
        self, sample: dict[str, torch.Tensor | dict[str, torch.Tensor]], assemble: bool = True
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Forward pass of embedding networks.

        Parameters
        ----------
        sample: dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Dictionary with the different tensors / dictionaries that will be used for the forward pass.

        assemble: bool
            Whether to assemble the different tensors into a single tensor or return a dictionary with the different

        Returns
        -------
        torch.Tensor | dict[str, torch.Tensor]
            Either the processed tensor or a dictionary with the different tensors that then have the be assembled
            manually

        """
        # -------------------------
        # Dynamic inputs
        # -------------------------
        # if self.cfg.nan_handling_method == "masked_mean":
        #     x_d = self._masked_mean(sample)
        # elif self.cfg.nan_handling_method == "input_replacement":
        #     x_d = self._input_replacement(sample)
        # else:
        #     x_d = torch.cat([v(torch.stack(list(sample[k].values()), dim=-1)) for k, v in self.emb_x_d.items()], dim=1)
        x_d_raw = torch.stack(
            list(sample[self._x_d_key].values()),
            dim=-1
        )

        # # -------------------------
        # # Frequency flags
        # # -------------------------
        # freq_flag = (
        #     self.flag_info["flag"].unsqueeze(0).expand(x_d.shape[0], -1, -1)
        #     if self.flag_info.get("flag") is not None
        #     else x_d.new_zeros(x_d.shape[0], x_d.shape[1], 0)
        # )

        # -------------------------
        # Static inputs
        # -------------------------
        # x_s = (
        #     self.emb_x_s(sample["x_s"]).unsqueeze(1).expand(-1, x_d.shape[1], -1)
        #     if self.cfg.static_input
        #     else x_d.new_zeros(x_d.shape[0], x_d.shape[1], 0)
        # )
        
        if self.cfg.static_input:
        # sample["x_s"] is already [B, S]
            x_s_raw = sample["x_s"].unsqueeze(1).expand(-1, x_d_raw.shape[1], -1)
        else:
            x_s_raw = x_d_raw.new_zeros(x_d_raw.shape[0], x_d_raw.shape[1], 0)

        # print(type(sample["x_s"]))
        # print(sample["x_s"].shape)
        # print(sample["x_s"])
       
        # --------------------------------
        # 3) Concat dynamic + static
        # --------------------------------
        x_all = torch.cat([x_d_raw, x_s_raw], dim=-1)   # [bs, seq_length, dyn+sta]

        # --------------------------------
        # 4) Pass through shared embedding
        # --------------------------------
        x_emb = self.emb_x(x_all)


        # return torch.cat([x_d, x_s], dim=2) if assemble else {"x_d": x_d, "x_s": x_s}
        return x_emb if assemble else {"x": x_emb, "x_d": x_d_raw, "x_s": x_s_raw}

    @staticmethod
    def build_embedding(input_dim: int, embedding: Optional[dict[str, str | float | list[int]]]):
        """Build embedding

        Parameters
        ----------
        input_dim: int
            Input dimension of the first layer.
        embedding: dict[str, str | float | list[int]]
            Dictionary with the embedding characteristics

        Returns
        -------
        nn.Sequential | nn.Identity
            Embedding network or nn.Identity

        """

        return (
            InputLayer_CPC.build_ffnn(
                input_dim=input_dim,
                spec=embedding["hiddens"],
                activation=embedding["activation"],
                dropout=embedding["dropout"],
            )
            if isinstance(embedding, dict)
            else nn.Identity()
        )

    @staticmethod
    def build_ffnn(input_dim: int, spec: list[int], activation: str = "relu", dropout: float = 0.0) -> nn.Sequential:
        """Builds a feedforward neural network based on the given specification.

        Parameters
        ----------
        input_dim: int
            Input dimension of the first layer.
        spec: List[int]
            Dimension of the different hidden layers.
        activation: str
            Activation function to use between layers (relu, linear, tanh, sigmoid).
            Default is 'relu'.
        dropout: float
            Dropout rate to apply after each layer (except the last one).
            Default is 0.0 (no dropout).

        Returns
        -------
        nn.Sequential
            A sequential model containing the feedforward neural network layers.

        """

        activation = InputLayer_CPC._get_activation_function(activation)
        ffnn_layers = []
        for i, out_dim in enumerate(spec):
            ffnn_layers.append(nn.Linear(input_dim, out_dim))
            ffnn_layers.append(activation)
   
            if dropout > 0.0:
                ffnn_layers.append(nn.Dropout(dropout)) 

            input_dim = out_dim  # updates next layer’s input size

        return nn.Sequential(*ffnn_layers)

    @staticmethod
    def _get_activation_function(activation: str) -> nn.Module:
        """Returns the activation function based on the given string.

        Parameters
        ----------
        activation: str
            Name of the activation function (e.g., 'relu', 'linear', 'tanh', 'sigmoid').

        Returns
        -------
        nn.Module
            The corresponding activation function module.

        """

        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation == "linear":
            return nn.Identity()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
