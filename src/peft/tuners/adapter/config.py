import math
import re
import warnings
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import List, Optional, Tuple, Union

from ...utils import PeftConfig, PeftType


@dataclass
class AdapterLayersConfig:
    linear_layer: type = field(default=None, metadata={"help": "The adapter linear layer class to use"})
    linear_8bit_layer: type = field(default=None, metadata={"help": "The adapter linear 8bit layer class to use"})
    linear_4bit_layer: type = field(default=None, metadata={"help": "The adapter linear 4bit layer class to use"})
    conv2d_layer: type = field(default=None, metadata={"help": "The adapter conv layer class to use"})
    embedding_layer: type = field(default=None, metadata={"help": "The adapter embedding layer class to use"})

@dataclass
class AdapterConfig(PeftConfig):
    """
    This is the base configuration class to store the configuration of most adapter layer models.

    Args:
        r (`int`): Adapter layer attention dimension.
        target_modules (`Union[List[str],str]`): The names of the modules to apply adaptation to.
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
    r: int = field(default=8, metadata={"help": "Adapter layer attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with adapter layers."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    dropout: float = field(default=0.0, metadata={"help": "Adapter layer dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for the adapter layer. Can be 'none', 'all' or 'adapter_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_weights: bool = field(
        default=True,
        metadata={"help": "Whether to initialize the weights of the adapter layers layers."},
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
    adapter_layers_config: AdapterLayersConfig = field(default = AdapterLayersConfig(), metadata={"help": "The adapter layer classes to use"})
    peft_type: PeftType = field(default=PeftType.ADAPTER, metadata={"help": "Defaults to PeftType.ADAPTER"})

    


