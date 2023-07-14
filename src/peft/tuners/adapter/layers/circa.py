from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import warnings
from ....utils import transpose
from ....import_utils import is_bnb_available, is_bnb_4bit_available
from .base import AdapterLayer, LinearAdapter, Conv2dAdapter, EmbeddingAdapter
from ..config import AdapterConfig, AdapterLayersConfig

from typing import List, Optional, Tuple, Union, Any
from .utils import to_power_of_2, AdviceFormat

class CircA(nn.Module):

    def __init__(
            self, 
            in_features: int, 
            out_features: int, 
            *args, 
            r: int = 3, 
            pad: bool = False, 
            use_checkpointing: bool = False, 
            bias: bool = True, 
            device: str = None, 
            dtype=None, 
            **kwargs
        ):

        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(*args, **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        # Subclasses may override {in,out}_features_extended
        if not hasattr(self, 'in_features_extended'):
            self.in_features_extended = in_features
        if not hasattr(self, 'out_features_extended'):
            self.out_features_extended = out_features
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None) 

        self.r = r
        self.pad = pad
        self.use_checkpointing = use_checkpointing
        self.d = max(self.in_features, self.out_features)

        if pad:
            self.d = to_power_of_2(self.d)
        self.out_features_extended = self.d

        # initialize psi, omega, and g as learnable parameters
        self.psi = nn.Parameter(torch.randint(0, 2, (1, self.in_features), dtype=torch.float32) * 2.0 - 1.0)
        self.omega = nn.Parameter(torch.randint(0, 2, (self.r, self.d), dtype=torch.float32) * 2.0 - 1.0)

        scale = torch.sqrt(torch.tensor(2.0/self.d))
        self.g = nn.Parameter(scale*torch.randn(self.r, self.d, dtype=torch.float32))

    def reset_parameters(self):
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        scale = torch.sqrt(torch.tensor(2.0/self.d))
        nn.init.normal_(self.g, mean=0.0, std=scale)
        self.psi.data = torch.randint(0, 2, (1, self.in_features), dtype=torch.float32) * 2.0 - 1.0
        self.omega.data = torch.randint(0, 2, (self.r, self.d), dtype=torch.float32) * 2.0 - 1.0

    def convert_to_dense_weight(self):
        factory_kwargs = {'device': self.weight.device, 'dtype': self.weight.dtype}
        dense_weight = self.forward_matmul(torch.eye(self.in_features, **factory_kwargs)).T
        return dense_weight

    def preprocess(self, x):
        in_features = x.shape[-1]
        if in_features < self.in_features_extended:
            x = F.pad(x, (0, self.in_features_extended - in_features))
        return x

    def postprocess(self, output):
        out_features_extended = output.shape[-1]
        if out_features_extended > self.out_features:
            output = output[..., :self.out_features]
        return output

    def forward_matmul(self, x):
        assert self.in_features == x.shape[-1], "x does not have in_features features"

        def inner_forward(x):
            # cast to fp16 if needed
            autocast = torch.is_autocast_enabled()
            power_of_2 = self.d & (self.d - 1) == 0
            if autocast and power_of_2:
                x = x.half()
                psi = self.psi.half()
                omega = self.omega.half()
                g = self.g.half()
            else:
                if autocast and not power_of_2:
                    with AdviceFormat():
                        warnings.warn("Autocast is enabled but dimension not a power of 2. Will use fp32. Specify pad=True to pad to power of 2.", UserWarning)
                psi = self.psi
                omega = self.omega
                g = self.g
            
            # process data once
            # no need to pad x since fft will do it for us
            fx = torch.fft.rfft(x * psi, n=self.d) # b x d
            # process parameters once
            fg = torch.fft.rfft(g) # r x d
                    
            # Reshape fx and fg to match dimensions for elementwise operations
            fx = fx.unsqueeze(-2) # Now fx is b (x seq_len) x 1 x d
            while fg.ndim < fx.ndim:
                fg = fg.unsqueeze(0) # Now fg is (1 x) 1 x r x d
                omega = omega.unsqueeze(0) # Now omega is (1 x) r x d

            # Apply irfft
            z = torch.fft.irfft(fg * fx, n=self.d) * omega # b (x seq_len) x r x d
            y = z.sum(dim=-2) # b (x seq_len) x d

            return y
        
        if self.use_checkpointing:
            y = checkpoint(inner_forward, x, use_reentrant=False, preserve_rng_state=False)
        else:
            y = inner_forward(x)
        
        return self.postprocess(y)

    def forward(self, x):
        output = self.forward_matmul(x)
        # Convert bias to output.dtype in case of AMP, otherwise bias and activation will be in FP32
        return (output + self.bias.to(dtype=output.dtype)) if self.bias is not None else output
    
class CircaAdapter(AdapterLayer):
    def __init__(self, in_features: int, out_features: int, **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        self.r = {}

    def update_settings(self, adapter_name: str, adapter_config: AdapterConfig):
        self.r[adapter_name] = adapter_config.r

class CircaLinearAdapter(CircaAdapter, LinearAdapter):
    def update_modules(self, adapter_name: str, adapter_config: AdapterConfig):
        self.adapter_A.update(nn.ModuleDict({adapter_name: CircA(self.in_features, self.out_features, r=adapter_config.r, bias=False, pad=True)}))

    def reset_adapter_parameters(self, adapter_name: str):
        self.adapter_A[adapter_name].reset_parameters()

    def materialize_adapter(self, adapter_name: str):
        if self.r[adapter_name] > 0:
            return transpose(
                    self.adapter_A[adapter_name].convert_to_dense_weight(),
                    self.fan_in_fan_out,
                )
        else:
            return None

class CircaConv2dAdapter(CircaAdapter, Conv2dAdapter):
    def update_modules(self, adapter_name: str, adapter_config: AdapterConfig):
        raise NotImplementedError("Conv2d is not implemented yet")
    
    def materialize_adapter(self, adapter_name: str):
        raise NotImplementedError("Conv2d is not implemented yet")

class CircaEmbeddingAdapter(CircaAdapter, EmbeddingAdapter):
    def update_modules(self, adapter_name: str, adapter_config: AdapterConfig):
        self.adapter_embedding_A.update(nn.ModuleDict({adapter_name: CircA(self.out_features, self.in_features, r=adapter_config.r, bias=False, pad=True)}))

    def reset_adapter_parameters(self, adapter_name: str):
        self.adapter_embedding_A[adapter_name].reset_parameters()
        nn.init.zeros_(self.adapter_embedding_B[adapter_name])

    def materialize_adapter(self, adapter_name: str):
        if self.r[adapter_name] > 0:
            return transpose(
                    self.adapter_embedding_A[adapter_name].convert_to_dense_weight(),
                    self.fan_in_fan_out,
                )
        else:
            return None

class CircaLinear(nn.Linear, CircaLinearAdapter):
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        adapter_config: AdapterConfig,
        **kwargs
    ):

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        CircaLinearAdapter.__init__(self, in_features=in_features, out_features=out_features)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = adapter_config.fan_in_fan_out
        self.adapter_config = adapter_config
        if adapter_config.fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        CircaLinearAdapter.update_layer(self, adapter_name, adapter_config)
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
            # x = x.to(self.adapter_A[self.active_adapter].weight.dtype)

            result += (
                self.adapter_A[self.active_adapter].forward(x)
            )
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)
        return result


class CircaEmbedding(nn.Embedding, CircaEmbeddingAdapter):
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
        CircaEmbeddingAdapter.__init__(self, in_features=num_embeddings, out_features=embedding_dim)
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
            after_B = F.embedding(
                x,
                self.adapter_embedding_B[self.active_adapter].T,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )

            result += (self.adapter_embedding_A[self.active_adapter].forward(after_B))
            return result
        else:
            return nn.Embedding.forward(self, x)


class CircaConv2d(nn.Conv2d, CircaConv2dAdapter):
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
        raise NotImplementedError("Conv2d is not implemented yet")

if is_bnb_available():
    import bitsandbytes as bnb

    class CircaLinear8bitLt(bnb.nn.Linear8bitLt, CircaLinearAdapter):
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
            CircaLinearAdapter.__init__(self, in_features=in_features, out_features=out_features)

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
                    output = self.circa_A[self.active_adapter](x).to(expected_dtype)
                else:
                    output = self.circa_A[self.active_adapter](x)
                            
                result += output
            return result

    if is_bnb_4bit_available():

        class CircaLinear4bit(bnb.nn.Linear4bit, CircaLinearAdapter):
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
                CircaLinearAdapter.__init__(self, in_features=in_features, out_features=out_features)

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
                        x = x.float() #.to(self.circa_A[self.active_adapter].weight.dtype)
                        output = (self.adapter_A[self.active_adapter](x)).to(expected_dtype)
                    else:
                        output = self.adapter_A[self.active_adapter](x)
                    result += output
                return result

DEFAULT_CIRCA_ADAPTER_LAYERS_CONFIG = AdapterLayersConfig(
    linear_layer=CircaLinear,
    linear_8bit_layer=CircaLinear8bitLt if is_bnb_available() else None,
    linear_4bit_layer=CircaLinear4bit if is_bnb_4bit_available() else None,
    conv2d_layer=CircaConv2d,
    embedding_layer=CircaEmbedding
)

@dataclass
class CircaConfig(AdapterConfig):
    adapter_layers_config: AdapterLayersConfig = field(default=DEFAULT_CIRCA_ADAPTER_LAYERS_CONFIG, metadata={"help": "Adapter layers config"})
