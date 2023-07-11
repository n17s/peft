import re
import copy
import warnings
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D
from transformers import PreTrainedModel

from ...import_utils import is_bnb_4bit_available, is_bnb_available
from ...utils import (
    COMMON_LAYERS_PATTERN,
    TRANSFORMERS_MODELS_TO_ADAPTER_MODULES_MAPPING,
    ModulesToSaveWrapper,
    PeftConfig,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
)

from .config import AdapterConfig
from .layers import AdapterLayer

if is_bnb_available():
    import bitsandbytes as bnb

class AdapterModel(torch.nn.Module):
    """
    Createsa base class for and adapter model from a pretrained transformers model.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`AdapterConfig`]): The configuration of the adapter model.

    Returns:
        `torch.nn.Module`: The adapter model.
    """

    def __init__(self, model: PreTrainedModel, config: AdapterConfig, adapter_name: str):
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name)

    def add_adapter(self, adapter_name: str, config: AdapterConfig = None):
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_adapter_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "AdapterModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )
        self.mark_only_adapter_layers_as_trainable(adapter_name)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_ADAPTER_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_ADAPTER_MODULES_MAPPING[model_config["model_type"]]
        return peft_config
    
    def _find_and_replace(self, adapter_name: str):
        adapter_config = self.peft_config[adapter_name]
        self._check_quantization_dependency()
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]

        for key in key_list:
            if not self._check_target_module_exists(adapter_config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)


            if isinstance(target, AdapterLayer):
                target.update_layer(adapter_name, adapter_config)
            else:
                new_module = self._create_new_module(adapter_name, target)
                self._replace_module(parent, target_name, new_module, target)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {adapter_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )
    
    def _check_quantization_dependency(self) -> None:
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if (loaded_in_4bit or loaded_in_8bit) and not is_bnb_available():
            raise ImportError(
                "To use adapter layers with 8-bit or 4-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )

    def _check_target_module_exists(self, adapter_config: AdapterConfig, key: str) -> bool:
        if isinstance(adapter_config.target_modules, str):
            target_module_found = re.fullmatch(adapter_config.target_modules, key)
        else:
            target_module_found = any(key.endswith(target_key) for target_key in adapter_config.target_modules)
            is_using_layer_indexes = getattr(adapter_config, "layers_to_transform", None) is not None
            layer_indexing_pattern = getattr(adapter_config, "layers_pattern", None)

            if is_using_layer_indexes and target_module_found:
                layers_pattern = COMMON_LAYERS_PATTERN if layer_indexing_pattern is None else layer_indexing_pattern
                layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern

                for pattern in layers_pattern:
                    layer_index = re.match(f".*.{pattern}\.(\d+)\.*", key)
                    if layer_index is not None:
                        layer_index = int(layer_index.group(1))
                        if isinstance(adapter_config.layers_to_transform, int):
                            target_module_found = layer_index == adapter_config.layers_to_transform
                        else:
                            target_module_found = layer_index in adapter_config.layers_to_transform

                        break
                    else:
                        target_module_found = False
        return target_module_found
    
    def _create_new_module(self, adapter_name: str, target: nn.Module) -> AdapterLayer:
        bias = hasattr(target, "bias") and target.bias is not None
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        adapter_config = copy.deepcopy(self.peft_config[adapter_name])
        adapter_layers_config = adapter_config.adapter_layers_config

        if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt) and adapter_layers_config.linear_8bit_layer is not None:
            eightbit_kwargs = {
                "has_fp16_weights": target.state.has_fp16_weights,
                "memory_efficient_backward": target.state.memory_efficient_backward,
                "threshold": target.state.threshold,
                "index": target.index,
            }
            new_module = adapter_layers_config.linear_8bit_layer(
                adapter_name, target.in_features, target.out_features, bias=bias, **eightbit_kwargs
            )
        elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target, bnb.nn.Linear4bit) and adapter_layers_config.linear_4bit_layer is not None:
            fourbit_kwargs = {
                "compute_dtype": target.compute_dtype,
                "compress_statistics": target.weight.compress_statistics,
                "quant_type": target.weight.quant_type,
            }
            new_module = adapter_layers_config.linear_4bit_layer(adapter_name, target.in_features, target.out_features, bias=bias, **fourbit_kwargs)
        elif isinstance(target, torch.nn.Embedding) and adapter_layers_config.embedding_layer is not None:
            in_features, out_features = target.num_embeddings, target.embedding_dim
            new_module = adapter_layers_config.embedding_layer(adapter_name, in_features, out_features, adapter_config)
        elif isinstance(target, torch.nn.Conv2d) and adapter_layers_config.conv2d_layer is not None:
            out_channels, in_channels = target.weight.size()[:2]
            kernel_size = target.weight.size()[2:]
            stride = target.stride
            padding = target.padding
            new_module = adapter_layers_config.conv2d_layer(adapter_name, in_channels, out_channels, kernel_size, stride, padding, adapter_config)
        elif adapter_layers_config.linear_layer is not None:
            if isinstance(target, torch.nn.Linear):
                in_features, out_features = target.in_features, target.out_features
                if adapter_config.fan_in_fan_out:
                    warnings.warn(
                        "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                        "Setting fan_in_fan_out to False."
                    )
                    adapter_config.fan_in_fan_out = False
            elif isinstance(target, Conv1D):
                in_features, out_features = (
                    target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                )
                if not adapter_config.fan_in_fan_out:
                    warnings.warn(
                        "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                        "Setting fan_in_fan_out to True."
                    )
                    adapter_config.fan_in_fan_out = True
            else:
                raise ValueError(
                    f"Target module {target} is not supported. "
                    f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                )
            new_module = adapter_layers_config.linear_layer(adapter_name, in_features, out_features, adapter_config, bias=bias)

        return new_module
    
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
            if "adapter_" in name:
                module.to(old_module.weight.device)
            if "ranknum" in name:
                module.to(old_module.weight.device)

    def mark_only_adapter_layers_as_trainable(self, adapter_name: str) -> None:
        model = self.model
        bias = self.peft_config[adapter_name].bias

        for n, p in model.named_parameters():
            if "adapter_" not in n:
                p.requires_grad = False

        if bias == "none":
            return
        elif bias == "all":
            for n, p in model.named_parameters():
                if "bias" in n:
                    p.requires_grad = True
        elif bias == "adapter_only":
            for m in model.modules():
                if isinstance(m, AdapterLayer) and hasattr(m, "bias") and m.bias is not None:
                    m.bias.requires_grad = True
        else:
            raise NotImplementedError
    
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
            if isinstance(module, AdapterLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, AdapterLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.active_adapter = adapter_name

    def merge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, AdapterLayer):
                module.merge()

    def unmerge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, AdapterLayer):
                module.unmerge()

    def merge_and_unload(self):
        r"""
        This method merges the LoRa layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.

        Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PeftModel

        >>> base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")
        >>> peft_model_id = "smangrul/falcon-40B-int4-peft-lora-sfttrainer-sample"
        >>> model = PeftModel.from_pretrained(base_model, peft_model_id)
        >>> merged_model = model.merge_and_unload()
        ```
        """
        if getattr(self.config, "model_type", None) == "gpt2":
            raise ValueError("GPT2 models are not supported for merging adapter layers")

        if getattr(self.model, "is_loaded_in_8bit", False) or getattr(self.model, "is_loaded_in_4bit", False):
            raise ValueError("Cannot merge LORA layers when the model is loaded in 8-bit mode")

        key_list = [key for key, _ in self.model.named_modules() if "adapter" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, AdapterLayer):
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
        self.peft_config[adapter_name] = replace(
            self.peft_config[adapters[0]], lora_alpha=self.peft_config[adapters[0]].r
        )
        self._find_and_replace(adapter_name)
        self.mark_only_adapter_layers_as_trainable(adapter_name)
        _freeze_adapter(self.model, adapter_name)
        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, AdapterLayer):
                if adapter_name in target.adapter_A:
                    target.adapter_A[adapter_name].weight.data = target.adapter_A[adapter_name].weight.data * 0.0
                    target.adapter_B[adapter_name].weight.data = target.adapter_B[adapter_name].weight.data * 0.0
                    for adapter, weight in zip(adapters, weights):
                        if adapter not in target.adapter_A:
                            continue
                        target.adapter_A[adapter_name].weight.data += (
                            target.adapter_A[adapter].weight.data * weight * target.scaling[adapter]
                        )
                        target.adapter_B[adapter_name].weight.data += target.adapter_B[adapter].weight.data * weight

                elif adapter_name in target.adapter_embedding_A:
                    target.adapter_embedding_A[adapter_name].data = target.adapter_embedding_A[adapter_name].data * 0.0
                    target.adapter_embedding_B[adapter_name].data = target.adapter_embedding_B[adapter_name].data * 0.0
                    for adapter, weight in zip(adapters, weights):
                        if adapter not in target.adapter_embedding_A:
                            continue
                        target.adapter_embedding_A[adapter_name].data += (
                            target.adapter_embedding_A[adapter].data * weight * target.scaling[adapter]
                        )
                        target.adapter_embedding_B[adapter_name].data += target.adapter_embedding_B[adapter].data * weight