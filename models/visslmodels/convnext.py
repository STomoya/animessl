
from __future__ import annotations

import torch

from vissl.config import AttrDict
from vissl.models.model_helpers import (
    get_trunk_forward_outputs,
    transform_model_input_data_type,
)
from vissl.models.trunks import register_model_trunk

from models.torchmodels.convnext import ConvNeXt

@register_model_trunk('convnext')
class ConvNeXt(ConvNeXt):
    def __init__(self, model_config: AttrDict, model_name: str):
        self.model_config = model_config
        self.trunk_config = self.model_config.TRUNK.CONVNEXT
        in_channels    = 3
        depths         = self.trunk_config.DEPTHS
        dims           = self.trunk_config.DIMS
        drop_path_rate = self.trunk_config.DROP_PATH_RATE
        layer_scale_init_value = self.trunk_config.LAYER_SCALE_INIT_VALUE
        super().__init__(in_channels, depths, dims, drop_path_rate, layer_scale_init_value)

        self.use_checkpointing = (
            self.model_config.ACTIVATION_CHECKPOINTING.USE_ACTIVATION_CHECKPOINTING
        )
        self.num_checkpointing_splits = (
            self.model_config.ACTIVATION_CHECKPOINTING.NUM_ACTIVATION_CHECKPOINTING_SPLITS
        )

    def forward(self, x: torch.Tensor, out_feat_keys: list[str]):
        model_input = transform_model_input_data_type(
            x, self.model_config.INPUT_TYPE
        )
        out = get_trunk_forward_outputs(
            feat=model_input,
            out_feat_keys=out_feat_keys,
            feature_blocks=self._feature_blocks,
            use_checkpointing=self.use_checkpointing,
            checkpointing_splits=self.num_checkpointing_splits,
        )
        return out
