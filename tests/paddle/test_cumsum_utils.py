# -*- coding: utf-8 -*-

import paddle

import fla_paddle.ops.utils.cumsum as cumsum_module


def test_chunk_local_cumsum_scalar_allocates_output_with_dtype(monkeypatch):
    recorded = {'dtype': None}
    real_empty_like = cumsum_module.paddle.empty_like

    def fake_empty_like(x, dtype=None, **kwargs):
        recorded['dtype'] = dtype
        return real_empty_like(x, dtype=dtype, **kwargs)

    class FakeKernel:
        def __getitem__(self, grid):
            def launch(**kwargs):
                return None

            return launch

    monkeypatch.setattr(cumsum_module.paddle, 'empty_like', fake_empty_like)
    monkeypatch.setattr(cumsum_module, 'chunk_local_cumsum_scalar_kernel', FakeKernel())

    g = paddle.randn([2, 64, 3], dtype=paddle.float32)
    cumsum_module.chunk_local_cumsum(g, chunk_size=64, output_dtype=paddle.float32)

    assert recorded['dtype'] == paddle.float32
