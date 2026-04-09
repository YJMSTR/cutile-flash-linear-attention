# -*- coding: utf-8 -*-
# flash-linear-attention Paddle migration entry point

import paddle
from fla_paddle.triton_utils import _is_package_installed

# No torch environment: enable triton scope compat globally (zero runtime overhead)
if not _is_package_installed("torch"):
    paddle.enable_compat(scope={"triton"})

from fla_paddle.ops.gated_delta_rule import (
    chunk_gated_delta_rule,
    chunk_gdn,
    fused_recurrent_gated_delta_rule,
    fused_recurrent_gdn,
)
from fla_paddle.ops.kda import (
    chunk_kda,
    fused_recurrent_kda,
)

__all__ = [
    'chunk_gated_delta_rule',
    'chunk_gdn',
    'fused_recurrent_gated_delta_rule',
    'fused_recurrent_gdn',
    'chunk_kda',
    'fused_recurrent_kda',
]
