import torch.nn as nn
import math

from abc import ABC, abstractmethod
from raft_lora.components.train.peft.src.peft.tuners.adapter.config import AdapterConfig 

from ..config import AdapterConfig

class AdapterLayer(ABC):
    def __init__(self, in_features: int, out_features: int, **kwargs):
        self.adapter_dropout = nn.ModuleDict({})
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    def update_layer(self, adapter_name: str, adapter_config: AdapterConfig):
        self.update_settings(adapter_name, adapter_config)
        self.update_dropout(adapter_name, adapter_config)
        if adapter_config.r > 0:
            self.update_modules(adapter_name, adapter_config)
        if adapter_config.init_weights:
            self.reset_parameters(adapter_name)
        self.to(self.weight.device)

    @abstractmethod
    def update_modules(self, adapter_name: str, adapter_config: AdapterConfig):
        raise NotImplementedError()
    
    @abstractmethod
    def reset_parameters(self, adapter_name: str):
        raise NotImplementedError()

    def update_dropout(self, adapter_name: str, adapter_config: AdapterConfig):
        if adapter_config.dropout > 0.0:
            adapter_dropout_layer = nn.Dropout(p=adapter_config.dropout)
        else:
            adapter_dropout_layer = nn.Identity()
        self.adapter_dropout.update(nn.ModuleDict({adapter_name: adapter_dropout_layer}))

    @abstractmethod
    def update_settings(self, adapter_name: str, adapter_config: AdapterConfig):
        raise NotImplementedError()

class LinearAdapter(AdapterLayer):
    def __init__(self, in_features: int, out_features: int, **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        self.adapter_A = nn.ModuleDict({})
        self.adapter_B = nn.ModuleDict({})

    def update_modules(self, adapter_name: str, adapter_config: AdapterConfig):
        self.adapter_A.update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, adapter_config.r, bias=False)}))
        self.adapter_B.update(nn.ModuleDict({adapter_name: nn.Linear(adapter_config.r, self.out_features, bias=False)}))

    def reset_parameters(self, adapter_name: str):
        if adapter_name in self.adapter_A.keys():
            nn.init.kaiming_uniform_(self.adapter_A[adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.adapter_B[adapter_name].weight)

class Conv2dAdapter(LinearAdapter):
    def update_modules(self, adapter_name: str, adapter_config: AdapterConfig):
        kernel_size = self.kwargs["kernel_size"]
        stride = self.kwargs["stride"]
        padding = self.kwargs["padding"]
        self.adapter_A.update(
            nn.ModuleDict({adapter_name: nn.Conv2d(self.in_features, adapter_config.r, kernel_size, stride, padding, bias=False)})
        )
        self.adapter_B.update(
            nn.ModuleDict({adapter_name: nn.Conv2d(adapter_config.r, self.out_features, (1, 1), (1, 1), bias=False)})
        )

class EmbeddingAdapter(AdapterLayer):
    def __init__(self, in_features: int, out_features: int, **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        self.adapter_embedding_A = nn.ModuleDict({})
        self.adapter_embedding_B = nn.ModuleDict({})

    def update_modules(self, adapter_name: str, adapter_config: AdapterConfig):
        self.adapter_embedding_A.update(
                nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros((adapter_config.r, self.in_features)))})
            )
        self.adapter_embedding_B.update(
            nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros((self.out_features, adapter_config.r)))})
        )

    def reset_parameters(self, adapter_name: str):
        if adapter_name in self.adapter_embedding_A.keys():
            nn.init.zeros_(self.adapter_embedding_A[adapter_name])
            nn.init.normal_(self.adapter_embedding_B[adapter_name])

