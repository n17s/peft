# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D
from torch.utils.checkpoint import checkpoint

from ..import_utils import is_bnb_4bit_available, is_bnb_available
from ..utils import (
    COMMON_LAYERS_PATTERN,
    TRANSFORMERS_MODELS_TO_CIRCA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    PeftConfig,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
)

from contextlib import contextmanager


if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class CircaConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`CircaModel`].

    Args:
        r (`int`): Circa attention dimension.
        target_modules (`Union[List[str],str]`): The names of the modules to apply Circa to.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out).
        For example, gpt-2 uses `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.:
        bias (`str`): Bias type for Circa. Can be 'none', 'all' or 'circa_only'
        modules_to_save (`List[str]`):List of modules apart from CircA layers to be set as trainable
            and saved in the final checkpoint.
        layers_to_transform (`Union[List[int],int]`):
            The layer indexes to transform, if this argument is specified, it will apply the CircA transformations on
            the layer indexes that are specified in this list. If a single integer is passed, it will apply the CircA
            transformations on the layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None` and if the layer
            pattern is not in the common layers pattern.
    """

    r: int = field(default=8, metadata={"help": "Circa r dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Circa."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for Circa. Can be 'none', 'all' or 'circa_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from CircA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_circa_weights: bool = field(
        default=True,
        metadata={"help": "Whether to initialize the weights of the Circa layers."},
    )
    layers_to_transform: Optional[Union[List, int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index."
        },
    )
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.CIRCA


class CircaModel(torch.nn.Module):
    """
    Creates a Circulant Adapter (Circa) model from a pretrained transformers model.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`CircaConfig`]): The configuration of the Circa model.

    Returns:
        `torch.nn.Module`: The Circa model.

    Example:

        ```py
        >>> from transformers import AutoModelForSeq2SeqLM, CircaConfig
        >>> from peft import CircaModel, CircaConfig

        >>> config = CircaConfig(
        ...     peft_type="CIRCA",
        ...     task_type="SEQ_2_SEQ_LM",
        ...     r=8,
        ...     target_modules=["q", "v"],
        ... )

        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> circa_model = CircaModel(config, model)
        ```

        ```py
        >>> import transformers
        >>> from peft import CircaConfig, PeftModel, get_peft_model, prepare_model_for_int8_training

        >>> target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"]
        >>> config = CircaConfig(
        ...     r=4, target_modules=target_modules, bias="none", task_type="CAUSAL_LM"
        ... )

        >>> model = transformers.GPTJForCausalLM.from_pretrained(
        ...     "kakaobrain/kogpt",
        ...     revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
        ...     pad_token_id=tokenizer.eos_token_id,
        ...     use_cache=False,
        ...     device_map={"": rank},
        ...     torch_dtype=torch.float16,
        ...     load_in_8bit=True,
        ... )
        >>> model = prepare_model_for_int8_training(model)
        >>> circa_model = get_peft_model(model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`CircaConfig`]): The configuration of the Circa model.
    """

    def __init__(self, model, config, adapter_name):
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_circa_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "CircaModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )
        mark_only_circa_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def _check_quantization_dependency(self):
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if (loaded_in_4bit or loaded_in_8bit) and not is_bnb_available():
            raise ImportError(
                "To use Circa with 8-bit or 4-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )

    def _check_target_module_exists(self, circa_config, key):
        if isinstance(circa_config.target_modules, str):
            target_module_found = re.fullmatch(circa_config.target_modules, key)
        else:
            target_module_found = any(key.endswith(target_key) for target_key in circa_config.target_modules)
            is_using_layer_indexes = getattr(circa_config, "layers_to_transform", None) is not None
            layer_indexing_pattern = getattr(circa_config, "layers_pattern", None)

            if is_using_layer_indexes and target_module_found:
                layers_pattern = COMMON_LAYERS_PATTERN if layer_indexing_pattern is None else layer_indexing_pattern
                layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern

                for pattern in layers_pattern:
                    layer_index = re.match(f".*.{pattern}\.(\d+)\.*", key)
                    if layer_index is not None:
                        layer_index = int(layer_index.group(1))
                        if isinstance(circa_config.layers_to_transform, int):
                            target_module_found = layer_index == circa_config.layers_to_transform
                        else:
                            target_module_found = layer_index in circa_config.layers_to_transform

                        break
                    else:
                        target_module_found = False
        return target_module_found

    def _create_new_module(self, circa_config, adapter_name, target):
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": circa_config.r,
            "fan_in_fan_out": circa_config.fan_in_fan_out,
            "init_circa_weights": circa_config.init_circa_weights,
        }
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)

        if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target.state.has_fp16_weights,
                    "memory_efficient_backward": target.state.memory_efficient_backward,
                    "threshold": target.state.threshold,
                    "index": target.index,
                }
            )
            new_module = Linear8bitLt(
                adapter_name, target.in_features, target.out_features, bias=bias, **eightbit_kwargs
            )
        elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target.compute_dtype,
                    "compress_statistics": target.weight.compress_statistics,
                    "quant_type": target.weight.quant_type,
                }
            )
            new_module = Linear4bit(adapter_name, target.in_features, target.out_features, bias=bias, **fourbit_kwargs)
        elif isinstance(target, torch.nn.Embedding):
            embedding_kwargs = kwargs.copy()
            embedding_kwargs.pop("fan_in_fan_out", None)
            in_features, out_features = target.num_embeddings, target.embedding_dim
            new_module = Embedding(adapter_name, in_features, out_features, **embedding_kwargs)
        elif isinstance(target, torch.nn.Conv2d):
            out_channels, in_channels = target.weight.size()[:2]
            kernel_size = target.weight.size()[2:]
            stride = target.stride
            padding = target.padding
            new_module = Conv2d(adapter_name, in_channels, out_channels, kernel_size, stride, padding, **kwargs)
        else:
            if isinstance(target, torch.nn.Linear):
                in_features, out_features = target.in_features, target.out_features
                if kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                        "Setting fan_in_fan_out to False."
                    )
                    kwargs["fan_in_fan_out"] = circa_config.fan_in_fan_out = False
            elif isinstance(target, Conv1D):
                in_features, out_features = (
                    target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                )
                if not kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                        "Setting fan_in_fan_out to True."
                    )
                    kwargs["fan_in_fan_out"] = circa_config.fan_in_fan_out = True
            else:
                raise ValueError(
                    f"Target module {target} is not supported. "
                    f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                )
            new_module = Linear(adapter_name, in_features, out_features, bias=bias, **kwargs)

        return new_module

    def _find_and_replace(self, adapter_name):
        circa_config = self.peft_config[adapter_name]
        self._check_quantization_dependency()
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]

        for key in key_list:
            if not self._check_target_module_exists(circa_config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)

            if isinstance(target, CircaLayer) and isinstance(target, torch.nn.Conv2d):
                target.update_layer_conv2d(
                    adapter_name,
                    circa_config.r,
                    circa_config.init_circa_weights,
                )
            elif isinstance(target, CircaLayer):
                target.update_layer(
                    adapter_name,
                    circa_config.r,
                    circa_config.init_circa_weights,
                )
            else:
                new_module = self._create_new_module(circa_config, adapter_name, target)
                self._replace_module(parent, target_name, new_module, target)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {circa_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if hasattr(old_module, "bias"):
            if old_module.bias is not None:
                new_module.bias = old_module.bias

        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "circa_" in name:
                module.to(old_module.weight.device)
            if "ranknum" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, CircaLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, CircaLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.active_adapter = adapter_name

    def merge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, CircaLayer):
                module.merge()

    def unmerge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, CircaLayer):
                module.unmerge()

    @staticmethod
    def _prepare_circa_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_CIRCA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_CIRCA_TARGET_MODULES_MAPPING[model_config["model_type"]]
        return peft_config

    def merge_and_unload(self):
        r"""
        This method merges the CircA layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.
        """
        if getattr(self.config, "model_type", None) == "gpt2":
            raise ValueError("GPT2 models are not supported for merging CircA layers")

        if getattr(self.model, "is_loaded_in_8bit", False) or getattr(self.model, "is_loaded_in_4bit", False):
            raise ValueError("Cannot merge CIRCA layers when the model is loaded in 8-bit mode")

        key_list = [key for key, _ in self.model.named_modules() if "circa" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, CircaLayer):
                if isinstance(target, nn.Embedding):
                    new_module = torch.nn.Embedding(target.in_features, target.out_features)
                elif isinstance(target, nn.Conv2d):
                    new_module = torch.nn.Conv2d(
                        target.in_channels,
                        target.out_channels,
                        kernel_size=target.kernel_size,
                        stride=target.stride,
                        padding=target.padding,
                        dilation=target.dilation,
                    )
                else:
                    bias = target.bias is not None
                    new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                target.merge()
                self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

    def add_weighted_adapter(self, adapters, weights, adapter_name):
        if len({self.peft_config[adapter].r for adapter in adapters}) != 1:
            raise ValueError("All adapters must have the same r value")
        self.peft_config[adapter_name] = self.peft_config[adapters[0]]
        self.peft_config[adapter_name].circa_alpha = self.peft_config[adapters[0]].r
        self._find_and_replace(adapter_name)
        mark_only_circa_as_trainable(self.model, self.peft_config[adapter_name].bias)
        _freeze_adapter(self.model, adapter_name)
        key_list = [key for key, _ in self.model.named_modules() if "circa" not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, CircaLayer):
                if adapter_name in target.circa_A:
                    target.circa_A[adapter_name].weight.data = target.circa_A[adapter_name].weight.data * 0.0
                    target.circa_B[adapter_name].weight.data = target.circa_B[adapter_name].weight.data * 0.0
                    for adapter, weight in zip(adapters, weights):
                        if adapter not in target.circa_A:
                            continue
                        target.circa_A[adapter_name].weight.data += (
                            target.circa_A[adapter].weight.data * weight * target.scaling[adapter]
                        )
                        target.circa_B[adapter_name].weight.data += target.circa_B[adapter].weight.data * weight

                elif adapter_name in target.circa_embedding_A:
                    target.circa_embedding_A[adapter_name].data = target.circa_embedding_A[adapter_name].data * 0.0
                    target.circa_embedding_B[adapter_name].data = target.circa_embedding_B[adapter_name].data * 0.0
                    for adapter, weight in zip(adapters, weights):
                        if adapter not in target.circa_embedding_A:
                            continue
                        target.circa_embedding_A[adapter_name].data += (
                            target.circa_embedding_A[adapter].data * weight * target.scaling[adapter]
                        )
                        target.circa_embedding_B[adapter_name].data += target.circa_embedding_B[adapter].data * weight


# Below code is based on https://github.com/microsoft/CircA/blob/main/circalib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# had to adapt it for `circa_only` to work
def mark_only_circa_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "circa_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "circa_only":
        for m in model.modules():
            if isinstance(m, CircaLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


@contextmanager
def AdviceFormat():
    old_format = warnings.formatwarning
    try:
        def custom_format(message, category, filename, lineno, line=None):
            return f'{filename}:{lineno} {category.__name__}: {message}\n'
        warnings.formatwarning = custom_format
        yield
    finally:
        warnings.formatwarning = old_format
    

def to_power_of_2(n):
    """
    As of pytorch 2.0.1 ffts for 16 bit are only supported for powers of 2.
    This convenience method returns the smallest power of 2 >= n.
    """
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    # n |= n >> 32  # not necessary for commonly used layer sizes
    n += 1
    return n
    

class CircA(nn.Module):

    def __init__(self, in_features, out_features, *args, r=3, pad=False, use_checkpointing=False, bias=True, device=None, dtype=None, **kwargs):
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

        scale = torch.sqrt(torch.tensor(2.0/self.d))
        # initialize psi, omega, and g as learnable parameters
        self.psi = nn.Parameter(torch.randint(0, 2, (1, self.in_features), dtype=torch.float32) * 2.0 - 1.0)
        self.omega = nn.Parameter(torch.randint(0, 2, (self.r, self.d), dtype=torch.float32) * 2.0 - 1.0)
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
        assert self.in_features == x.shape[1], "x does not have in_features features"

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
            fx = fx.unsqueeze(1) # Now fx is b x 1 x d
            fg = fg.unsqueeze(0) # Now fg is 1 x r x d

            # Apply irfft
            z = torch.fft.irfft(fg * fx, n=self.d) * omega.unsqueeze(0) # b x r x d
            y = z.sum(dim=1) # b x d

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
    

class CircaLayer:
    def __init__(self, in_features: int, out_features: int, **kwargs):
        self.r = {}
        self.circa_A = nn.ModuleDict({})
        # For Embedding layer
        self.circa_embedding_A = nn.ModuleDict({})
        # Mark the weight as unmerged
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    def update_layer(self, adapter_name, r, circa_alpha, circa_dropout, init_circa_weights):
        self.r[adapter_name] = r
        # Actual trainable parameters
        if r > 0:
            self.circa_A.update(nn.ModuleDict({adapter_name: CircA(self.in_features, self.out_features, r=r, bias=False)}))
        if init_circa_weights:
            self.reset_circa_parameters(adapter_name)
        self.to(self.weight.device)

    def update_layer_conv2d(self, adapter_name, r, circa_alpha, circa_dropout, init_circa_weights):
        raise NotImplementedError

    def update_layer_embedding(self, adapter_name, r, circa_alpha, circa_dropout, init_circa_weights):
        self.r[adapter_name] = r

        # Actual trainable parameters
        if r > 0:
            self.circa_embedding_A.update(
                nn.ModuleDict({adapter_name: CircA(self.out_features, self.in_features, r=r, bias=False)})
            )
        if init_circa_weights:
            self.reset_circa_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_circa_parameters(self, adapter_name):
        if adapter_name in self.circa_A.keys():
            self.circa_A[adapter_name].reset_parameters()

        if adapter_name in self.circa_embedding_A.keys():
            self.circa_embedding_A[adapter_name].reset_parameters()
            # OLD CODE
            # initialize a the same way as the default for nn.linear and b to zero
            #nn.init.zeros_(self.circa_embedding_A[adapter_name])
            #nn.init.normal_(self.circa_embedding_B[adapter_name])


class Linear(nn.Linear, CircaLayer):
    # Circa implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs,
    ):
        init_circa_weights = kwargs.pop("init_circa_weights", True)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        CircaLayer.__init__(self, in_features=in_features, out_features=out_features)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, init_circa_weights)
        self.active_adapter = adapter_name

    def merge(self):
        if self.active_adapter not in self.circa_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data += (
                transpose(
                    self.circa_A[self.active_adapter].convert_to_dense_weight(),
                    self.fan_in_fan_out,
                )
            )
            self.merged = True

    def unmerge(self):
        if self.active_adapter not in self.circa_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data -= (
                transpose(
                    self.circa_A[self.active_adapter].convert_to_dense_weight(),
                    self.fan_in_fan_out,
                )
            )
            self.merged = False

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        if self.active_adapter not in self.circa_A.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

            #x = x.to(self.circa_A[self.active_adapter].weight.dtype)

            result += (
                self.circa_A[self.active_adapter].forward(x)
            )
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)

        return result


class Embedding(nn.Embedding, CircaLayer):
    # CircA implemented in a Embedding layer
    def __init__(
        self,
        adapter_name: str,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        circa_alpha: int = 1,
        circa_dropout: float = 0.0,
        **kwargs,
    ):
        init_circa_weights = kwargs.pop("init_circa_weights", True)

        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        CircaLayer.__init__(self, in_features=num_embeddings, out_features=embedding_dim)

        self.weight.requires_grad = False

        nn.Embedding.reset_parameters(self)
        self.update_layer_embedding(adapter_name, r, circa_alpha, circa_dropout, init_circa_weights)
        self.active_adapter = adapter_name

    def unmerge(self, mode: bool = True):
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data -= (
                transpose(
                    self.circa_embedding_B[self.active_adapter] @ self.circa_embedding_A[self.active_adapter], True
                )
                * self.scaling[self.active_adapter]
            )
            self.merged = False

    def merge(self):
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data += (
                transpose(
                    self.circa_embedding_B[self.active_adapter] @ self.circa_embedding_A[self.active_adapter], True
                )
                * self.scaling[self.active_adapter]
            )
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.disable_adapters:
            if self.r[self.active.adapter] > 0 and self.merged:
                self.weight.data -= (
                    transpose(
                        self.circa_embedding_B[self.active_adapter].weight
                        @ self.circa_embedding_A[self.active_adapter].weight,
                        True,
                    )
                    * self.scaling[self.active_adapter]
                )
                self.merged = False
            return nn.Embedding.forward(self, x)

        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            if self.r[self.active_adapter] > 0:
                after_A = F.embedding(
                    x,
                    self.circa_embedding_A[self.active_adapter].T,
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                )
                result += (after_A @ self.circa_embedding_B[self.active_adapter].T) * self.scaling[self.active_adapter]
            return result
        else:
            return nn.Embedding.forward(self, x)


class Conv2d(nn.Conv2d, CircaLayer):
    # Circa implemented in a conv2d layer
    def __init__(
        self,
        adapter_name: str,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        r: int = 0,
        circa_alpha: int = 1,
        circa_dropout: float = 0.0,
        **kwargs,
    ):
        raise NotImplementedError



if is_bnb_available():

    class Linear8bitLt(bnb.nn.Linear8bitLt, CircaLayer):
        # Circa implemented in a dense layer
        def __init__(
            self,
            adapter_name,
            in_features,
            out_features,
            r: int = 0,
            circa_alpha: int = 1,
            circa_dropout: float = 0.0,
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
            CircaLayer.__init__(self, in_features=in_features, out_features=out_features)

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            init_circa_weights = kwargs.pop("init_circa_weights", True)
            self.update_layer(adapter_name, r, circa_alpha, circa_dropout, init_circa_weights)
            self.active_adapter = adapter_name

        def forward(self, x: torch.Tensor):
            result = super().forward(x)

            if self.disable_adapters or self.active_adapter not in self.circa_A.keys():
                return result
            elif self.r[self.active_adapter] > 0:
                if not torch.is_autocast_enabled():
                    expected_dtype = result.dtype

                    if x.dtype != torch.float32:
                        x = x.float()
                    output = (
                            self.circa_A[self.active_adapter](x)
                        ).to(expected_dtype)
                else:
                    output = (
                            self.circa_A[self.active_adapter](x)
                        )
                result += output
            return result

    if is_bnb_4bit_available():

        class Linear4bit(bnb.nn.Linear4bit, CircaLayer):
            # Circa implemented in a dense layer
            def __init__(
                self,
                adapter_name,
                in_features,
                out_features,
                r: int = 0,
                circa_alpha: int = 1,
                circa_dropout: float = 0.0,
                **kwargs,
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
                CircaLayer.__init__(self, in_features=in_features, out_features=out_features)

                # Freezing the pre-trained weight matrix
                self.weight.requires_grad = False

                init_circa_weights = kwargs.pop("init_circa_weights", True)
                self.update_layer(adapter_name, r, circa_alpha, circa_dropout, init_circa_weights)
                self.active_adapter = adapter_name

            def forward(self, x: torch.Tensor):
                result = super().forward(x)

                if self.disable_adapters or self.active_adapter not in self.circa_A.keys():
                    return result
                elif self.r[self.active_adapter] > 0:
                    result = result.clone()
                    if not torch.is_autocast_enabled():
                        expected_dtype = result.dtype
                        x = x.float() #.to(self.circa_A[self.active_adapter].weight.dtype)
                        output = (self.circa_A[self.active_adapter](x)).to(expected_dtype)
                    else:
                        output = self.circa_A[self.active_adapter](x)
                    result += output
                return result
