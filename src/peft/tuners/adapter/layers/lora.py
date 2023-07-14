from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings

from ....utils import transpose
from ....import_utils import is_bnb_available, is_bnb_4bit_available
from .base import AdapterLayer, LinearAdapter, Conv2dAdapter, EmbeddingAdapter
from ..config import AdapterConfig, AdapterLayersConfig

from typing import List, Optional, Tuple, Union, Any

class LoraAdapter(AdapterLayer):
    def __init__(self, in_features: int, out_features: int, **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        self.lora_alpha = {}
        self.r = {}
        self.scaling = {}

    def update_settings(self, adapter_name: str, adapter_config: AdapterConfig):
        self.r[adapter_name] = adapter_config.r
        self.lora_alpha[adapter_name] = adapter_config.lora_alpha
        self.scaling[adapter_name] = adapter_config.lora_alpha / adapter_config.r

class LoraLinearAdapter(LoraAdapter, LinearAdapter):
    def materialize_adapter(self, adapter_name: str):
        if self.r[adapter_name] > 0:
            return (
                transpose(
                    self.adapter_B[adapter_name].weight @ self.adapter_A[adapter_name].weight,
                    self.fan_in_fan_out,
                )
                * self.scaling[adapter_name]
            )
        else:
            return None

class LoraConv2dAdapter(LoraAdapter, Conv2dAdapter):
    def materialize_adapter(self, adapter_name: str):
        if self.r[adapter_name] > 0:
            # https://github.com/bmaltais/kohya_ss/blob/feb6728762a8f463d15ba936d189d4c3abfaa1ab/networks/lora.py#L117
            if self.weight.size()[2:4] == (1, 1):
                # conv2d 1x1
                return (
                    self.adapter_B[adapter_name].weight.squeeze(3).squeeze(2)
                    @ self.adapter_A[adapter_name].weight.squeeze(3).squeeze(2)
                ).unsqueeze(2).unsqueeze(3) * self.scaling[adapter_name]
            else:
                # conv2d 3x3
                return (
                    F.conv2d(
                        self.adapter_A[adapter_name].weight.permute(1, 0, 2, 3),
                        self.adapter_B[adapter_name].weight,
                    ).permute(1, 0, 2, 3)
                    * self.scaling[adapter_name]
                )
        else:
            return None

class LoraEmbeddingAdapter(LoraAdapter, EmbeddingAdapter):
    def materialize_adapter(self, adapter_name: str):
        if self.r[adapter_name] > 0:
            return (
                transpose(
                    self.adapter_embedding_B[adapter_name] @ self.adapter_embedding_A[adapter_name], True
                )
                * self.scaling[adapter_name]
            )
        else:
            return None

class LoraLinear(nn.Linear, LoraLinearAdapter):
    # Lora implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        adapter_config: AdapterConfig,
        **kwargs
    ):

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLinearAdapter.__init__(self, in_features=in_features, out_features=out_features)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = adapter_config.fan_in_fan_out
        self.adapter_config = adapter_config
        if adapter_config.fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        LoraLinearAdapter.update_layer(self, adapter_name, adapter_config)
        self.active_adapter = adapter_name

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        if self.active_adapter not in self.adapter_A.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            x = x.to(self.adapter_A[self.active_adapter].weight.dtype)

            result += (
                self.adapter_B[self.active_adapter](
                    self.adapter_A[self.active_adapter](self.adapter_dropout[self.active_adapter](x))
                )
                * self.scaling[self.active_adapter]
            )
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)
        return result


class LoraEmbedding(nn.Embedding, LoraEmbeddingAdapter):
    # LoRA implemented in a Embedding layer
    def __init__(
        self,
        adapter_name: str,
        num_embeddings: int,
        embedding_dim: int,
        adapter_config: AdapterConfig,
        **kwargs,
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoraEmbeddingAdapter.__init__(self, in_features=num_embeddings, out_features=embedding_dim)
        self.weight.requires_grad = False

        nn.Embedding.reset_parameters(self)
        self.update_layer(adapter_name, adapter_config)
        self.active_adapter = adapter_name

    def forward(self, x: torch.Tensor):
        if self.disable_adapters:
            if self.r[self.active.adapter] > 0 and self.merged:
                self.unmerge()
            return nn.Embedding.forward(self, x)

        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            after_A = F.embedding(
                x,
                self.adapter_embedding_A[self.active_adapter].T,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
            result += (after_A @ self.adapter_embedding_B[self.active_adapter].T) * self.scaling[self.active_adapter]
            return result
        else:
            return nn.Embedding.forward(self, x)


class LoraConv2d(nn.Conv2d, LoraConv2dAdapter):
    # Lora implemented in a conv2d layer
    def __init__(
        self,
        adapter_name: str,
        in_channels: int,
        out_channels: int,
        adapter_config: AdapterConfig,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        **kwargs,
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding)
        LoraConv2dAdapter.__init__(
            self,
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        nn.Conv2d.reset_parameters(self)
        self.update_layer(adapter_name, adapter_config)
        self.active_adapter = adapter_name

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        if self.active_adapter not in self.adapter_A.keys():
            return F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

            x = x.to(self.adapter_A[self.active_adapter].weight.dtype)

            result += (
                self.adapter_B[self.active_adapter](
                    self.adapter_A[self.active_adapter](self.adapter_dropout[self.active_adapter](x))
                )
                * self.scaling[self.active_adapter]
            )
        else:
            result = F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

        result = result.to(previous_dtype)

        return result

if is_bnb_available():
    import bitsandbytes as bnb

    class Linear8bitLt(bnb.nn.Linear8bitLt, LoraLinearAdapter):
        # Lora implemented in a dense layer
        def __init__(
            self,
            adapter_name: str,
            in_features: int,
            out_features: int,
            adapter_config: AdapterConfig,
            **kwargs,
        ):
            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                has_fp16_weights=kwargs.get("has_fp16_weights", True),
                memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
                threshold=kwargs.get("threshold", 0.0),
                index=kwargs.get("index", None),
            )
            LoraLinearAdapter.__init__(self, in_features=in_features, out_features=out_features)

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            self.update_layer(adapter_name, adapter_config)
            self.active_adapter = adapter_name

        def forward(self, x: torch.Tensor):
            result = super().forward(x)

            if self.disable_adapters or self.active_adapter not in self.adapter_A.keys():
                return result
            elif self.r[self.active_adapter] > 0:
                if not torch.is_autocast_enabled():
                    expected_dtype = result.dtype

                    if x.dtype != torch.float32:
                        x = x.float()
                    output = (
                        self.adapter_B[self.active_adapter](
                            self.adapter_A[self.active_adapter](self.adapter_dropout[self.active_adapter](x))
                        ).to(expected_dtype)
                        * self.scaling[self.active_adapter]
                    )
                else:
                    output = (
                        self.adapter_B[self.active_adapter](
                            self.adapter_A[self.active_adapter](self.adapter_dropout[self.active_adapter](x))
                        )
                        * self.scaling[self.active_adapter]
                    )
                result += output
            return result

    if is_bnb_4bit_available():

        class Linear4bit(bnb.nn.Linear4bit, LoraLinearAdapter):
            # Lora implemented in a dense layer
            def __init__(
                self,
                adapter_name: str,
                in_features: int,
                out_features: int,
                adapter_config: AdapterConfig,
                **kwargs
            ):
                bnb.nn.Linear4bit.__init__(
                    self,
                    in_features,
                    out_features,
                    bias=kwargs.get("bias", True),
                    compute_dtype=kwargs.get("compute_dtype", torch.float32),
                    compress_statistics=kwargs.get("compress_statistics", True),
                    quant_type=kwargs.get("quant_type", "nf4"),
                )
                LoraLinearAdapter.__init__(self, in_features=in_features, out_features=out_features)

                # Freezing the pre-trained weight matrix
                self.weight.requires_grad = False
                self.update_layer(adapter_name, adapter_config)
                self.active_adapter = adapter_name

            def forward(self, x: torch.Tensor):
                result = super().forward(x)

                if self.disable_adapters or self.active_adapter not in self.adapter_A.keys():
                    return result
                elif self.r[self.active_adapter] > 0:
                    result = result.clone()
                    if not torch.is_autocast_enabled():
                        expected_dtype = result.dtype
                        x = x.to(self.adapter_A[self.active_adapter].weight.dtype)
                        output = (
                            self.adapter_B[self.active_adapter](
                                self.adapter_A[self.active_adapter](self.adapter_dropout[self.active_adapter](x))
                            ).to(expected_dtype)
                            * self.scaling[self.active_adapter]
                        )
                    else:
                        output = (
                            self.adapter_B[self.active_adapter](
                                self.adapter_A[self.active_adapter](self.adapter_dropout[self.active_adapter](x))
                            )
                            * self.scaling[self.active_adapter]
                        )
                    result += output
                return result

DEFAULT_LORA_ADAPTER_LAYERS_CONFIG = AdapterLayersConfig(
    linear_layer=LoraLinear,
    linear_8bit_layer=Linear8bitLt if is_bnb_available() else None,
    linear_4bit_layer=Linear4bit if is_bnb_4bit_available() else None,
    conv2d_layer=LoraConv2d,
    embedding_layer=LoraEmbedding
)

@dataclass
class LoraConfig(AdapterConfig):
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    adapter_layers_config: AdapterLayersConfig = field(default=DEFAULT_LORA_ADAPTER_LAYERS_CONFIG, metadata={"help": "Adapter layers config"})
