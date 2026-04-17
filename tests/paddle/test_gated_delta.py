# -*- coding: utf-8 -*-
# Tests for Gated Delta Rule operators on PaddlePaddle
# Aligned with tests/ops/test_gated_delta.py

import os

import paddle
import paddle.nn.functional as F
import pytest
from einops import repeat

import fla_paddle.ops.common.chunk_delta_h as chunk_delta_h_module
import fla_paddle.ops.common.chunk_o as chunk_o_module
import fla_paddle.ops.gated_delta_rule.chunk as chunk_module
import fla_paddle.ops.gated_delta_rule.chunk_fwd as chunk_fwd_module
from fla_paddle.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from fla_paddle.ops.gated_delta_rule.gate import fused_gdn_gate, naive_gdn_gate
from fla_paddle.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule

from .conftest import assert_close


def test_chunk_gdn_bwd_dhu_allocates_dh0_with_dtype(monkeypatch):
    recorded = {'h0_dtype': None}
    real_empty_like = chunk_delta_h_module.paddle.empty_like

    def fake_empty_like(x, dtype=None, **kwargs):
        if x is h0:
            recorded['h0_dtype'] = dtype
        return real_empty_like(x, dtype=dtype, **kwargs)

    class FakeKernel:
        def __getitem__(self, grid):
            def launch(**kwargs):
                return None

            return launch

    monkeypatch.setattr(chunk_delta_h_module.paddle, 'empty_like', fake_empty_like)
    monkeypatch.setattr(chunk_delta_h_module, 'chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64', FakeKernel())

    q = paddle.randn([1, 64, 2, 8], dtype=paddle.float32)
    k = paddle.randn([1, 64, 2, 8], dtype=paddle.float32)
    w = paddle.randn([1, 64, 2, 8], dtype=paddle.float32)
    dv = paddle.randn([1, 64, 2, 8], dtype=paddle.float32)
    do = paddle.randn([1, 64, 2, 8], dtype=paddle.float32)
    h0 = paddle.randn([1, 2, 8, 8], dtype=paddle.float32)

    chunk_delta_h_module.chunk_gated_delta_rule_bwd_dhu(
        q=q,
        k=k,
        w=w,
        dv=dv,
        do=do,
        h0=h0,
        scale=1.0,
    )

    assert recorded['h0_dtype'] == paddle.float32


def test_chunk_gdn_fwd_intra_allocates_A_with_zeros(monkeypatch):
    recorded = {'shape': None, 'dtype': None}
    real_zeros = chunk_fwd_module.paddle.zeros

    def fake_zeros(shape=None, dtype=None, **kwargs):
        recorded['shape'] = list(shape)
        recorded['dtype'] = dtype
        return real_zeros(shape=shape, dtype=dtype, **kwargs)

    class FakeKernel:
        def __getitem__(self, grid):
            def launch(**kwargs):
                return None

            return launch

    def fake_recompute_w_u_fwd(**kwargs):
        k = kwargs['k']
        v = kwargs['v']
        return paddle.zeros_like(k), paddle.zeros_like(v)

    monkeypatch.setattr(chunk_fwd_module.paddle, 'zeros', fake_zeros)
    monkeypatch.setattr(chunk_fwd_module, 'chunk_gated_delta_rule_fwd_kkt_solve_kernel', FakeKernel())
    monkeypatch.setattr(chunk_fwd_module, 'recompute_w_u_fwd', fake_recompute_w_u_fwd)

    k = paddle.randn([2, 64, 3, 16], dtype=paddle.float32)
    v = paddle.randn([2, 64, 3, 16], dtype=paddle.float32)
    g = paddle.randn([2, 64, 3], dtype=paddle.float32)
    beta = paddle.randn([2, 64, 3], dtype=paddle.float32)

    _, _, A = chunk_fwd_module.chunk_gated_delta_rule_fwd_intra(
        k=k,
        v=v,
        g=g,
        beta=beta,
    )

    assert A.shape == [2, 64, 3, 64]
    assert recorded['shape'] == [2, 64, 3, 64]
    assert recorded['dtype'] == k.dtype


def test_chunk_gdn_bwd_dqkwg_uses_compact_dg_buffer_when_nk_is_one(monkeypatch):
    captured = {'shape': None}

    class FakeKernel:
        def __getitem__(self, grid):
            def launch(**kwargs):
                dg = kwargs.get('dg')
                if dg is not None:
                    captured['shape'] = list(dg.shape)
                return None

            return launch

    monkeypatch.setattr(chunk_o_module, 'chunk_bwd_kernel_dqkwg', FakeKernel())
    monkeypatch.setattr(chunk_o_module, 'check_shared_mem', lambda arch=None, *args, **kwargs: arch == 'hopper')

    q = paddle.randn([2, 64, 2, 64], dtype=paddle.float32)
    k = paddle.randn([2, 64, 2, 64], dtype=paddle.float32)
    v = paddle.randn([2, 64, 2, 64], dtype=paddle.float32)
    do = paddle.randn([2, 64, 2, 64], dtype=paddle.float32)
    h = paddle.randn([2, 1, 2, 64, 64], dtype=paddle.float32)
    dh = paddle.randn([2, 1, 2, 64, 64], dtype=paddle.float32)
    w = paddle.randn([2, 64, 2, 64], dtype=paddle.float32)
    g = paddle.randn([2, 64, 2], dtype=paddle.float32)
    dv = paddle.randn([2, 64, 2, 64], dtype=paddle.float32)

    _, _, _, dg = chunk_o_module.chunk_bwd_dqkwg(
        q=q,
        k=k,
        v=v,
        do=do,
        h=h,
        dh=dh,
        w=w,
        g=g,
        dv=dv,
        scale=1.0,
        use_exp2=True,
    )

    assert captured['shape'] == list(g.shape)
    assert list(dg.shape) == list(g.shape)


def test_chunk_gdn_chunk_o_caches_const_tiling_per_device(monkeypatch):
    calls = {'count': 0}

    class FakeKernel:
        def __getitem__(self, grid):
            def launch(**kwargs):
                return None

            return launch

    def fake_check_shared_mem(arch=None, *args, **kwargs):
        calls['count'] += 1
        return arch == 'hopper'

    chunk_o_module._const_tiling.cache_clear()
    chunk_o_module._chunk_o_launch_meta.cache_clear()
    monkeypatch.setattr(chunk_o_module, 'chunk_bwd_kernel_dv_local', FakeKernel())
    monkeypatch.setattr(chunk_o_module, 'chunk_bwd_kernel_dqkwg', FakeKernel())
    monkeypatch.setattr(chunk_o_module, 'check_shared_mem', fake_check_shared_mem)

    q = paddle.randn([2, 64, 2, 64], dtype=paddle.float32)
    k = paddle.randn([2, 64, 2, 64], dtype=paddle.float32)
    v = paddle.randn([2, 64, 2, 64], dtype=paddle.float32)
    do = paddle.randn([2, 64, 2, 64], dtype=paddle.float32)
    h = paddle.randn([2, 1, 2, 64, 64], dtype=paddle.float32)
    dh = paddle.randn([2, 1, 2, 64, 64], dtype=paddle.float32)
    w = paddle.randn([2, 64, 2, 64], dtype=paddle.float32)
    g = paddle.randn([2, 64, 2], dtype=paddle.float32)
    dv = paddle.randn([2, 64, 2, 64], dtype=paddle.float32)

    chunk_o_module.chunk_bwd_dv_local(q=q, k=k, do=do, g=g, scale=1.0, use_exp2=True)
    chunk_o_module.chunk_bwd_dv_local(q=q, k=k, do=do, g=g, scale=1.0, use_exp2=True)

    assert calls['count'] == 1
    chunk_o_module._const_tiling.cache_clear()
    chunk_o_module._chunk_o_launch_meta.cache_clear()


def test_chunk_gdn_forward_enters_driver_fastpath_scope(monkeypatch):
    calls = {'driver': 0, 'fastpath': 0}

    class Ctx:
        def __init__(self, key):
            self.key = key

        def __enter__(self):
            calls[self.key] += 1

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_chunk_gdn_fwd(**kwargs):
        q = kwargs['q']
        v = kwargs['v']
        g = kwargs['g']
        B, T, H, K = q.shape
        V = v.shape[-1]
        o = paddle.zeros_like(v)
        A = paddle.zeros([B, T, H, 64], dtype=q.dtype)
        if kwargs.get('return_intermediates'):
            w = paddle.zeros([B, T, H, K], dtype=q.dtype)
            u = paddle.zeros_like(v)
            h = paddle.zeros([B, T // 64, H, K, V], dtype=paddle.float32)
            v_new = paddle.zeros_like(v)
            return g, o, A, None, kwargs['initial_state'], None, w, u, h, v_new
        return g, o, A, None, kwargs['initial_state'], None

    monkeypatch.setattr(chunk_module, 'activate_paddle_driver', lambda: Ctx('driver'))
    monkeypatch.setattr(chunk_module, 'compat_kernel_wrapper_fastpath', lambda: Ctx('fastpath'))
    monkeypatch.setattr(chunk_module, 'chunk_gated_delta_rule_fwd', fake_chunk_gdn_fwd)

    q = paddle.randn([1, 64, 2, 8], dtype=paddle.float32)
    k = paddle.randn([1, 64, 2, 8], dtype=paddle.float32)
    v = paddle.randn([1, 64, 2, 8], dtype=paddle.float32)
    g = paddle.randn([1, 64, 2], dtype=paddle.float32)
    beta = paddle.randn([1, 64, 2], dtype=paddle.float32)

    for tensor in [q, k, v, g, beta]:
        tensor.stop_gradient = False

    chunk_module.chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        use_qk_l2norm_in_kernel=False,
        output_final_state=False,
    )

    assert calls == {'driver': 1, 'fastpath': 1}


def test_chunk_gdn_backward_skips_recompute_by_default_for_equal_length_training(monkeypatch):
    calls = {'recompute': 0}

    def fake_chunk_gdn_fwd(**kwargs):
        q = kwargs['q']
        v = kwargs['v']
        g = kwargs['g']
        B, T, H, K = q.shape
        V = v.shape[-1]
        o = paddle.zeros_like(v)
        A = paddle.zeros([B, T, H, 64], dtype=q.dtype)
        w = paddle.ones([B, T, H, K], dtype=q.dtype)
        u = paddle.ones([B, T, H, V], dtype=v.dtype)
        h = paddle.ones([B, T // 64, H, K, V], dtype=paddle.float32)
        v_new = paddle.ones([B, T, H, V], dtype=v.dtype)
        if kwargs.get('return_intermediates'):
            return g, o, A, None, kwargs['initial_state'], g, w, u, h, v_new
        return g, o, A, None, kwargs['initial_state'], g

    def fake_recompute_w_u_fwd(**kwargs):
        calls['recompute'] += 1
        return paddle.zeros_like(kwargs['k']), paddle.zeros_like(kwargs['v'])

    monkeypatch.setattr(chunk_module, 'chunk_gated_delta_rule_fwd', fake_chunk_gdn_fwd)
    monkeypatch.setattr(chunk_module, 'recompute_w_u_fwd', fake_recompute_w_u_fwd)
    monkeypatch.setattr(chunk_module, 'chunk_bwd_dv_local', lambda **kwargs: paddle.zeros_like(kwargs['do']))
    monkeypatch.setattr(
        chunk_module,
        'chunk_gated_delta_rule_bwd_dhu',
        lambda **kwargs: (
            paddle.zeros_like(kwargs['w'], dtype=paddle.float32),
            paddle.zeros_like(kwargs['h0']),
            kwargs['dv'],
        ),
    )
    monkeypatch.setattr(
        chunk_module,
        'chunk_bwd_dqkwg',
        lambda **kwargs: (
            paddle.zeros_like(kwargs['q'], dtype=paddle.float32),
            paddle.zeros_like(kwargs['k'], dtype=paddle.float32),
            paddle.zeros_like(kwargs['w'], dtype=paddle.float32),
            paddle.zeros_like(kwargs['g'], dtype=paddle.float32),
        ),
    )
    monkeypatch.setattr(
        chunk_module,
        'prepare_wy_repr_bwd',
        lambda **kwargs: (
            paddle.zeros_like(kwargs['k'], dtype=paddle.float32),
            paddle.zeros_like(kwargs['v']),
            paddle.zeros_like(kwargs['beta'], dtype=paddle.float32),
            paddle.zeros_like(kwargs['g'], dtype=paddle.float32),
        ),
    )

    q = paddle.randn([1, 64, 2, 8], dtype=paddle.float32)
    k = paddle.randn([1, 64, 2, 8], dtype=paddle.float32)
    v = paddle.randn([1, 64, 2, 8], dtype=paddle.float32)
    g = paddle.randn([1, 64, 2], dtype=paddle.float32)
    beta = paddle.randn([1, 64, 2], dtype=paddle.float32)
    initial_state = paddle.zeros([1, 2, 8, 8], dtype=paddle.float32)
    for tensor in [q, k, v, g, beta, initial_state]:
        tensor.stop_gradient = False

    o, _ = chunk_module.chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        use_qk_l2norm_in_kernel=False,
        output_final_state=False,
    )
    paddle.autograd.backward([o], [paddle.ones_like(o)])

    assert calls['recompute'] == 0


def test_chunk_gdn_backward_keeps_recompute_when_output_final_state_requested(monkeypatch):
    calls = {'recompute': 0}

    def fake_recompute_w_u_fwd(**kwargs):
        calls['recompute'] += 1
        return paddle.zeros_like(kwargs['k']), paddle.zeros_like(kwargs['v'])

    monkeypatch.setattr(chunk_module, 'recompute_w_u_fwd', fake_recompute_w_u_fwd)
    monkeypatch.setattr(chunk_module, 'chunk_bwd_dv_local', lambda **kwargs: paddle.zeros_like(kwargs['do']))
    monkeypatch.setattr(
        chunk_module,
        'chunk_gated_delta_rule_bwd_dhu',
        lambda **kwargs: (
            paddle.zeros_like(kwargs['w'], dtype=paddle.float32),
            paddle.zeros_like(kwargs['h0']),
            kwargs['dv'],
        ),
    )
    monkeypatch.setattr(
        chunk_module,
        'chunk_bwd_dqkwg',
        lambda **kwargs: (
            paddle.zeros_like(kwargs['q'], dtype=paddle.float32),
            paddle.zeros_like(kwargs['k'], dtype=paddle.float32),
            paddle.zeros_like(kwargs['w'], dtype=paddle.float32),
            paddle.zeros_like(kwargs['g'], dtype=paddle.float32),
        ),
    )
    monkeypatch.setattr(
        chunk_module,
        'prepare_wy_repr_bwd',
        lambda **kwargs: (
            paddle.zeros_like(kwargs['k'], dtype=paddle.float32),
            paddle.zeros_like(kwargs['v']),
            paddle.zeros_like(kwargs['beta'], dtype=paddle.float32),
            paddle.zeros_like(kwargs['g'], dtype=paddle.float32),
        ),
    )

    q = paddle.randn([1, 64, 2, 8], dtype=paddle.float32)
    k = paddle.randn([1, 64, 2, 8], dtype=paddle.float32)
    v = paddle.randn([1, 64, 2, 8], dtype=paddle.float32)
    g = paddle.randn([1, 64, 2], dtype=paddle.float32)
    beta = paddle.randn([1, 64, 2], dtype=paddle.float32)
    initial_state = paddle.zeros([1, 2, 8, 8], dtype=paddle.float32)
    for tensor in [q, k, v, g, beta, initial_state]:
        tensor.stop_gradient = False

    o, ht = chunk_module.chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        use_qk_l2norm_in_kernel=False,
        output_final_state=True,
    )
    paddle.autograd.backward([o, ht], [paddle.ones_like(o), paddle.ones_like(ht)])

    assert calls['recompute'] == 1


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HV', 'D', 'scale', 'gate_logit_normalizer', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HV{}-D{}-scale{}-gate_logit_normalizer{}-{}".format(*test))
        for test in [
            (1, 63, 1, 1, 64, 1, 1, paddle.float32),
            (2, 500, 4, 4, 60, 1, 1, paddle.float32),
            (2, 1000, 2, 8, 128, 1, 0.1, paddle.float32),
            (3, 1024, 2, 2, 128, 0.1, 1, paddle.float32),
            (4, 1024, 3, 3, 128, 1, 10, paddle.float32),
            (4, 2048, 4, 4, 64, 0.1, 1, paddle.float32),
            (2, 1024, 4, 4, 128, 1, 0.1, paddle.float16),
            (2, 1024, 4, 8, 128, 1, 10, paddle.float16),
        ]
    ],
)
def test_fused_recurrent(
    framework_tracker,
    B: int,
    T: int,
    H: int,
    HV: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    dtype,
):
    paddle.seed(42)
    q = paddle.randn([B, T, H, D], dtype=paddle.float32)
    k = paddle.randn([B, T, H, D], dtype=paddle.float32)
    v = paddle.randn([B, T, HV, D], dtype=dtype)
    beta = paddle.rand([B, T, HV], dtype=dtype).sigmoid()
    g = F.log_sigmoid(paddle.rand([B, T, HV], dtype=paddle.float32))
    g = g / gate_logit_normalizer
    h0 = paddle.randn([B, HV, D, D], dtype=paddle.float32)

    ref, ref_ht = naive_recurrent_gated_delta_rule(
        q=F.normalize(repeat(q.clone(), 'b t h d -> b t (h g) d', g=HV // H), p=2, axis=-1).cast(dtype),
        k=F.normalize(repeat(k.clone(), 'b t h d -> b t (h g) d', g=HV // H), p=2, axis=-1).cast(dtype),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )
    tri, tri_ht = fused_recurrent_gated_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        initial_state=h0.clone(),
        use_qk_l2norm_in_kernel=True,
        output_final_state=True,
    )
    assert_close('o', ref, tri, 0.002)
    assert_close('ht', ref_ht, tri_ht, 0.002)
    assert framework_tracker.detect_tensor_framework(tri) == 'paddle'
    assert framework_tracker.detect_triton_driver() == 'paddle'


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'gate_logit_normalizer', 'mask_p', 'use_qk_l2norm_in_kernel', 'dtype'),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-D{}-scale{}-gate_logit_normalizer{}-mask_p{}-use_qk_l2norm_in_kernel{}-{}".format(*test),
        )
        for test in [
            (2, 75, 4, 64, 1, 0.01, 0, False, paddle.float16),
            (2, 500, 3, 60, 1, 1, 0, False, paddle.float16),
            (2, 1000, 3, 64, 0.1, 1, 0.5, False, paddle.float16),
            (3, 1024, 4, 100, 1, 0.1, 0, False, paddle.float16),
            (4, 1024, 4, 128, 0.1, 1, 0, False, paddle.float16),
            (4, 1024, 4, 128, 0.1, 1, 0, True, paddle.float16),
            (2, 1500, 4, 128, 0.1, 10, 0, False, paddle.float16),
            (4, 2048, 8, 64, 0.1, 1, 0, False, paddle.float16),
        ]
    ],
)
def test_chunk(
    framework_tracker,
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    mask_p: float,
    use_qk_l2norm_in_kernel: bool,
    dtype,
):
    paddle.seed(42)
    q = paddle.rand([B, T, H, D], dtype=dtype)
    k = paddle.rand([B, T, H, D], dtype=dtype)
    v = paddle.rand([B, T, H, D], dtype=dtype)
    beta = paddle.rand([B, T, H], dtype=paddle.float32).sigmoid()
    g = F.log_sigmoid(paddle.rand([B, T, H], dtype=paddle.float32))
    g = g / gate_logit_normalizer
    g = g * (paddle.rand(g.shape, dtype=g.dtype) > mask_p).cast(g.dtype)
    h0 = paddle.zeros([B, H, D, D], dtype=paddle.float32)
    for t in [q, k, v, beta, g, h0]:
        t.stop_gradient = False

    tri, tri_ht = chunk_gated_delta_rule(
        q=F.normalize(q.clone(), p=2, axis=-1) if not use_qk_l2norm_in_kernel else q.clone(),
        k=F.normalize(k.clone(), p=2, axis=-1) if not use_qk_l2norm_in_kernel else k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    do = paddle.randn(v.shape, dtype=v.dtype)
    dht = paddle.randn(h0.shape, dtype=h0.dtype)
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dh0 = (
        q.grad.clone(), k.grad.clone(), v.grad.clone(),
        beta.grad.clone(), g.grad.clone(), h0.grad.clone()
    )
    q.clear_gradient()
    k.clear_gradient()
    v.clear_gradient()
    beta.clear_gradient()
    g.clear_gradient()
    h0.clear_gradient()

    ref, ref_ht = naive_recurrent_gated_delta_rule(
        q=F.normalize(q.clone(), p=2, axis=-1),
        k=F.normalize(k.clone(), p=2, axis=-1),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
    )

    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dg, ref_dh0 = (
        q.grad.clone(), k.grad.clone(), v.grad.clone(),
        beta.grad.clone(), g.grad.clone(), h0.grad.clone()
    )
    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.008)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.008)
    assert_close('db', ref_dbeta, tri_dbeta, 0.02)
    assert_close('dg', ref_dg, tri_dg, 0.02)
    assert_close('dh0', ref_dh0, tri_dh0, 0.008)
    assert framework_tracker.detect_tensor_framework(tri) == 'paddle'
    assert framework_tracker.detect_triton_driver() == 'paddle'


@pytest.mark.parametrize(
    ('B', 'T', 'Hq', 'H', 'D', 'scale', 'gate_logit_normalizer', 'use_qk_l2norm_in_kernel', 'dtype'),
    [
        pytest.param(
            *test,
            id="B{}-T{}-Hq{}-H{}-D{}-scale{}-gate_logit_normalizer{}-use_qk_l2norm_in_kernel{}-{}".format(*test),
        )
        for test in [
            (2, 256, 2, 4, 64, 1, 1, False, paddle.float16),
            (2, 512, 1, 4, 64, 0.1, 1, False, paddle.float16),
            (2, 512, 2, 8, 64, 1, 0.1, True, paddle.float16),
            (2, 1024, 4, 8, 128, 0.1, 1, False, paddle.float16),
        ]
    ],
)
def test_chunk_gqa(
    framework_tracker,
    B: int,
    T: int,
    Hq: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    use_qk_l2norm_in_kernel: bool,
    dtype,
):
    paddle.seed(42)
    assert H % Hq == 0
    G = H // Hq

    q = paddle.rand([B, T, Hq, D], dtype=dtype)
    k = paddle.rand([B, T, Hq, D], dtype=dtype)
    v = paddle.rand([B, T, H, D], dtype=dtype)
    beta = paddle.rand([B, T, H], dtype=paddle.float32).sigmoid()
    g = F.log_sigmoid(paddle.rand([B, T, H], dtype=paddle.float32))
    g = g / gate_logit_normalizer
    h0 = paddle.zeros([B, H, D, D], dtype=paddle.float32)
    for t in [q, k, v, beta, g, h0]:
        t.stop_gradient = False

    tri, tri_ht = chunk_gated_delta_rule(
        q=F.normalize(q.clone(), p=2, axis=-1) if not use_qk_l2norm_in_kernel else q.clone(),
        k=F.normalize(k.clone(), p=2, axis=-1) if not use_qk_l2norm_in_kernel else k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    do = paddle.randn(v.shape, dtype=v.dtype)
    dht = paddle.randn(h0.shape, dtype=h0.dtype)
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dh0 = (
        q.grad.clone(), k.grad.clone(), v.grad.clone(),
        beta.grad.clone(), g.grad.clone(), h0.grad.clone()
    )
    q.clear_gradient()
    k.clear_gradient()
    v.clear_gradient()
    beta.clear_gradient()
    g.clear_gradient()
    h0.clear_gradient()

    ref, ref_ht = naive_recurrent_gated_delta_rule(
        q=F.normalize(repeat(q.clone(), 'b t h d -> b t (h g) d', g=G), p=2, axis=-1),
        k=F.normalize(repeat(k.clone(), 'b t h d -> b t (h g) d', g=G), p=2, axis=-1),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
    )

    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dg, ref_dh0 = (
        q.grad.clone(), k.grad.clone(), v.grad.clone(),
        beta.grad.clone(), g.grad.clone(), h0.grad.clone()
    )
    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.008)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.008)
    assert_close('db', ref_dbeta, tri_dbeta, 0.02)
    assert_close('dg', ref_dg, tri_dg, 0.02)
    assert_close('dh0', ref_dh0, tri_dh0, 0.008)
    assert framework_tracker.detect_tensor_framework(tri) == 'paddle'
    assert framework_tracker.detect_triton_driver() == 'paddle'


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'gate_logit_normalizer', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-scale{}-gate_logit_normalizer{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, 1, 1, paddle.float16),
            (2, 500, 3, 60, 1, 1, paddle.float16),
            (3, 1024, 4, 128, 0.1, 1, paddle.float16),
            (4, 2048, 8, 64, 0.1, 1, paddle.float16),
        ]
    ],
)
def test_chunk_transpose_state(
    framework_tracker,
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    dtype,
):
    paddle.seed(42)
    q = paddle.rand([B, T, H, D], dtype=dtype)
    k = paddle.rand([B, T, H, D], dtype=dtype)
    v = paddle.rand([B, T, H, D], dtype=dtype)
    beta = paddle.rand([B, T, H], dtype=dtype).sigmoid()
    g = F.log_sigmoid(paddle.rand([B, T, H], dtype=paddle.float32))
    g = g / gate_logit_normalizer
    h0_kv = paddle.randn([B, H, D, D], dtype=paddle.float32)
    h0_vk = h0_kv.transpose([0, 1, 3, 2]).contiguous()
    for t in [q, k, v, beta, g, h0_kv, h0_vk]:
        t.stop_gradient = False

    tri, tri_ht = chunk_gated_delta_rule(
        q=F.normalize(q.clone(), p=2, axis=-1),
        k=F.normalize(k.clone(), p=2, axis=-1),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0_vk.clone(),
        output_final_state=True,
        transpose_state_layout=True,
    )
    do = paddle.randn(v.shape, dtype=v.dtype)
    dht_vk = paddle.randn([B, H, D, D], dtype=paddle.float32)
    dht_kv = dht_vk.transpose([0, 1, 3, 2]).contiguous()
    ((tri * do).sum() + (tri_ht * dht_vk).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dh0 = (
        q.grad.clone(), k.grad.clone(), v.grad.clone(),
        beta.grad.clone(), g.grad.clone(), h0_vk.grad.clone()
    )
    q.clear_gradient()
    k.clear_gradient()
    v.clear_gradient()
    beta.clear_gradient()
    g.clear_gradient()
    h0_vk.clear_gradient()

    ref, ref_ht = chunk_gated_delta_rule(
        q=F.normalize(q.clone(), p=2, axis=-1),
        k=F.normalize(k.clone(), p=2, axis=-1),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0_kv.clone(),
        output_final_state=True,
        transpose_state_layout=False,
    )
    ((ref * do).sum() + (ref_ht * dht_kv).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dg, ref_dh0 = (
        q.grad.clone(), k.grad.clone(), v.grad.clone(),
        beta.grad.clone(), g.grad.clone(), h0_kv.grad.clone()
    )

    assert_close('o', ref, tri, 1e-4)
    assert_close('ht', ref_ht, tri_ht.transpose([0, 1, 3, 2]), 1e-4)
    assert_close('dq', ref_dq, tri_dq, 1e-4)
    assert_close('dk', ref_dk, tri_dk, 1e-4)
    assert_close('dv', ref_dv, tri_dv, 1e-4)
    assert_close('db', ref_dbeta, tri_dbeta, 1e-4)
    assert_close('dg', ref_dg, tri_dg, 1e-4)
    assert_close('dh0', ref_dh0, tri_dh0.transpose([0, 1, 3, 2]), 1e-4)
    assert framework_tracker.detect_tensor_framework(tri) == 'paddle'
    assert framework_tracker.detect_triton_driver() == 'paddle'


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HV', 'D', 'scale', 'gate_logit_normalizer', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HV{}-D{}-scale{}-gate_logit_normalizer{}-{}".format(*test))
        for test in [
            (1, 63, 1, 1, 64, 1, 1, paddle.float32),
            (2, 500, 4, 4, 60, 1, 1, paddle.float32),
            (2, 1000, 2, 8, 128, 1, 0.1, paddle.float32),
            (3, 1024, 2, 2, 128, 0.1, 1, paddle.float32),
            (4, 2048, 4, 4, 64, 0.1, 1, paddle.float32),
        ]
    ],
)
def test_fused_recurrent_transpose_state(
    framework_tracker,
    B: int,
    T: int,
    H: int,
    HV: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    dtype,
):
    paddle.seed(42)
    q = paddle.randn([B, T, H, D], dtype=paddle.float32)
    k = paddle.randn([B, T, H, D], dtype=paddle.float32)
    v = paddle.randn([B, T, HV, D], dtype=dtype)
    beta = paddle.rand([B, T, HV], dtype=dtype).sigmoid()
    g = F.log_sigmoid(paddle.rand([B, T, HV], dtype=paddle.float32))
    g = g / gate_logit_normalizer
    h0_kv = paddle.randn([B, HV, D, D], dtype=paddle.float32)
    h0_vk = h0_kv.transpose([0, 1, 3, 2]).contiguous()

    ref, ref_ht = fused_recurrent_gated_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        initial_state=h0_kv.clone(),
        use_qk_l2norm_in_kernel=True,
        output_final_state=True,
        transpose_state_layout=False,
    )
    tri, tri_ht = fused_recurrent_gated_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        initial_state=h0_vk.clone(),
        use_qk_l2norm_in_kernel=True,
        output_final_state=True,
        transpose_state_layout=True,
    )
    assert_close('o', ref, tri, 1e-4)
    assert_close('ht', ref_ht, tri_ht.transpose([0, 1, 3, 2]), 1e-4)
    assert framework_tracker.detect_tensor_framework(tri) == 'paddle'
    assert framework_tracker.detect_triton_driver() == 'paddle'


@pytest.mark.parametrize(
    ('H', 'D', 'mask_p', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-D{}-mask_p{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (4, 60, 0, [0, 15], paddle.float16),
            (4, 64, 0, [0, 256, 500, 1000], paddle.float16),
            (4, 64, 0.5, [0, 256, 500, 1000], paddle.float16),
            (4, 100, 0, [0, 15, 100, 300, 1200, 2000], paddle.float16),
        ]
    ],
)
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '1',
    reason='Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set',
)
def test_chunk_varlen(
    framework_tracker,
    H: int,
    D: int,
    mask_p: float,
    cu_seqlens: list,
    dtype,
):
    paddle.seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    cu_seqlens_t = paddle.to_tensor(cu_seqlens, dtype=paddle.int64)
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    q = paddle.randn([1, T, H, D], dtype=dtype)
    k = F.normalize(paddle.randn([1, T, H, D], dtype=paddle.float32), p=2, axis=-1).cast(dtype)
    v = paddle.randn([1, T, H, D], dtype=dtype)
    g = F.log_sigmoid(paddle.rand([1, T, H], dtype=dtype))
    g = g * (paddle.rand(g.shape, dtype=g.dtype) > mask_p).cast(g.dtype)
    beta = paddle.rand([1, T, H], dtype=paddle.float32).sigmoid()
    h0 = paddle.randn([N, H, D, D], dtype=dtype)

    for t in [q, k, v, beta, g, h0]:
        t.stop_gradient = False
    do = paddle.randn(v.shape, dtype=v.dtype)
    dht = paddle.rand(h0.shape, dtype=h0.dtype)

    tri, tri_ht = chunk_gated_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens_t,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dh0 = (
        q.grad.clone(), k.grad.clone(), v.grad.clone(),
        beta.grad.clone(), g.grad.clone(), h0.grad.clone()
    )
    q.clear_gradient()
    k.clear_gradient()
    v.clear_gradient()
    beta.clear_gradient()
    g.clear_gradient()
    h0.clear_gradient()

    ref_list = []
    ref_ht_list = []
    for i in range(N):
        s, e = cu_seqlens[i], cu_seqlens[i + 1]
        ref_i, ref_ht_i = naive_recurrent_gated_delta_rule(
            q=q[:, s:e],
            k=k[:, s:e],
            v=v[:, s:e],
            beta=beta[:, s:e],
            g=g[:, s:e],
            initial_state=h0[i],
            output_final_state=True,
        )
        ref_list.append(ref_i)
        ref_ht_list.append(ref_ht_i)
    ref = paddle.concat(ref_list, axis=1)
    ref_ht = paddle.concat(ref_ht_list, axis=0)

    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dg, ref_dh0 = (
        q.grad.clone(), k.grad.clone(), v.grad.clone(),
        beta.grad.clone(), g.grad.clone(), h0.grad.clone()
    )

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.007)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.007)
    assert_close('db', ref_dbeta, tri_dbeta, 0.015)
    assert_close('dg', ref_dg, tri_dg, 0.015)
    assert_close('dh0', ref_dh0, tri_dh0, 0.007)
    assert framework_tracker.detect_tensor_framework(tri) == 'paddle'
    assert framework_tracker.detect_triton_driver() == 'paddle'


@pytest.mark.parametrize(
    ('H', 'D', 'mask_p', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-D{}-mask_p{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (4, 60, 0, [0, 8192], paddle.float16),
            (4, 60, 0, [0, 15], paddle.float16),
            (4, 64, 0, [0, 256, 500, 1000], paddle.float16),
            (4, 64, 0.5, [0, 256, 500, 1000], paddle.float16),
            (4, 100, 0, [0, 15, 100, 300, 1200, 2000], paddle.float16),
        ]
    ],
)
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '1',
    reason='Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set',
)
def test_chunk_varlen_prefill(
    framework_tracker,
    H: int,
    D: int,
    mask_p: float,
    cu_seqlens: list,
    dtype,
):
    paddle.seed(42)
    with paddle.no_grad():
        os.environ['TRITON_F32_DEFAULT'] = 'ieee'
        cu_seqlens_t = paddle.to_tensor(cu_seqlens, dtype=paddle.int64)
        T = cu_seqlens[-1]
        N = len(cu_seqlens) - 1

        q = paddle.randn([1, T, H, D], dtype=dtype)
        k = F.normalize(paddle.randn([1, T, H, D], dtype=paddle.float32), p=2, axis=-1).cast(dtype)
        v = paddle.randn([1, T, H, D], dtype=dtype)
        g = F.log_sigmoid(paddle.rand([1, T, H], dtype=dtype))
        g = g * (paddle.rand(g.shape, dtype=g.dtype) > mask_p).cast(g.dtype)
        beta = paddle.rand([1, T, H], dtype=dtype).sigmoid()
        h0 = paddle.randn([N, H, D, D], dtype=dtype)

        tri, tri_ht = chunk_gated_delta_rule(
            q=q.clone(),
            k=k.clone(),
            v=v.clone(),
            beta=beta.clone(),
            g=g.clone(),
            initial_state=h0.clone(),
            output_final_state=True,
            cu_seqlens=cu_seqlens_t,
        )

        ref_list = []
        ref_ht_list = []
        for i in range(N):
            s, e = cu_seqlens[i], cu_seqlens[i + 1]
            ref_i, ref_ht_i = naive_recurrent_gated_delta_rule(
                q=q[:, s:e],
                k=k[:, s:e],
                v=v[:, s:e],
                beta=beta[:, s:e],
                g=g[:, s:e],
                initial_state=h0[i],
                output_final_state=True,
            )
            ref_list.append(ref_i)
            ref_ht_list.append(ref_ht_i)
        ref = paddle.concat(ref_list, axis=1)
        ref_ht = paddle.concat(ref_ht_list, axis=0)

        assert_close('o', ref, tri, 0.005)
        assert_close('ht', ref_ht, tri_ht, 0.005)
        assert framework_tracker.detect_tensor_framework(tri) == 'paddle'
        assert framework_tracker.detect_triton_driver() == 'paddle'


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'has_dt_bias', 'use_qk_l2norm_in_kernel', 'dtype'),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-D{}-scale{}-has_dt_bias{}-use_qk_l2norm{}-{}".format(*test),
        )
        for test in [
            (2, 75, 4, 64, 1, True, True, paddle.float16),
            (2, 500, 3, 60, 1, False, False, paddle.float16),
            (2, 1000, 3, 64, 0.1, True, False, paddle.float16),
            (3, 1024, 4, 100, 1, True, True, paddle.float16),
            (4, 1024, 4, 128, 0.1, False, True, paddle.float16),
            (4, 2048, 8, 64, 0.1, True, False, paddle.float16),
        ]
    ],
)
def test_chunk_gate_in_kernel(
    framework_tracker,
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    has_dt_bias: bool,
    use_qk_l2norm_in_kernel: bool,
    dtype,
):
    """Test use_gate_in_kernel=True path: fused gate activation + chunk cumsum inside kernel."""
    paddle.seed(42)
    q = paddle.rand([B, T, H, D], dtype=dtype)
    k = paddle.rand([B, T, H, D], dtype=dtype)
    v = paddle.rand([B, T, H, D], dtype=dtype)
    beta = paddle.rand([B, T, H], dtype=paddle.float32).sigmoid()
    g_raw = paddle.randn([B, T, H], dtype=paddle.float32)
    A_log = paddle.randn([H], dtype=paddle.float32)
    dt_bias = paddle.randn([H], dtype=paddle.float32) if has_dt_bias else None
    h0 = paddle.zeros([B, H, D, D], dtype=paddle.float32)

    for t in [q, k, v, beta, g_raw, h0]:
        t.stop_gradient = False
    A_log.stop_gradient = False
    if dt_bias is not None:
        dt_bias.stop_gradient = False

    # === Triton path: use_gate_in_kernel=True ===
    tri, tri_ht = chunk_gated_delta_rule(
        q=q.clone() if use_qk_l2norm_in_kernel else F.normalize(q.clone(), p=2, axis=-1),
        k=k.clone() if use_qk_l2norm_in_kernel else F.normalize(k.clone(), p=2, axis=-1),
        v=v.clone(),
        g=g_raw.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=True,
        A_log=A_log.clone(),
        dt_bias=dt_bias.clone() if dt_bias is not None else None,
    )
    do = paddle.randn(v.shape, dtype=v.dtype)
    dht = paddle.randn(h0.shape, dtype=h0.dtype)
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dh0 = (
        q.grad.clone(), k.grad.clone(), v.grad.clone(),
        beta.grad.clone(), g_raw.grad.clone(), h0.grad.clone()
    )
    tri_dA_log = A_log.grad.clone()
    tri_ddt_bias = dt_bias.grad.clone() if dt_bias is not None else None
    q.clear_gradient()
    k.clear_gradient()
    v.clear_gradient()
    beta.clear_gradient()
    g_raw.clear_gradient()
    h0.clear_gradient()
    A_log.clear_gradient()
    if dt_bias is not None:
        dt_bias.clear_gradient()

    # === Reference path: manually compute gate, then use_gate_in_kernel=False ===
    g_ref = naive_gdn_gate(g_raw, A_log, dt_bias)
    ref, ref_ht = chunk_gated_delta_rule(
        q=q.clone() if use_qk_l2norm_in_kernel else F.normalize(q.clone(), p=2, axis=-1),
        k=k.clone() if use_qk_l2norm_in_kernel else F.normalize(k.clone(), p=2, axis=-1),
        v=v.clone(),
        g=g_ref,
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dh0 = (
        q.grad.clone(), k.grad.clone(), v.grad.clone(),
        beta.grad.clone(), h0.grad.clone()
    )
    ref_dg = g_raw.grad.clone()
    ref_dA_log = A_log.grad.clone()
    ref_ddt_bias = dt_bias.grad.clone() if dt_bias is not None else None

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.008)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.008)
    assert_close('db', ref_dbeta, tri_dbeta, 0.02)
    assert_close('dg', ref_dg, tri_dg, 0.02)
    assert_close('dh0', ref_dh0, tri_dh0, 0.008)
    assert_close('dA_log', ref_dA_log, tri_dA_log, 0.02)
    if dt_bias is not None:
        assert_close('ddt_bias', ref_ddt_bias, tri_ddt_bias, 0.02)
    assert framework_tracker.detect_tensor_framework(tri) == 'paddle'
    assert framework_tracker.detect_triton_driver() == 'paddle'


@pytest.mark.parametrize(
    ('B', 'T', 'Hq', 'H', 'D', 'scale', 'has_dt_bias', 'dtype'),
    [
        pytest.param(
            *test,
            id="B{}-T{}-Hq{}-H{}-D{}-scale{}-has_dt_bias{}-{}".format(*test),
        )
        for test in [
            (2, 256, 2, 4, 64, 1, True, paddle.float16),
            (2, 512, 1, 4, 64, 0.1, False, paddle.float16),
            (2, 512, 2, 8, 64, 1, True, paddle.float16),
            (2, 1024, 4, 8, 128, 0.1, True, paddle.float16),
        ]
    ],
)
def test_chunk_gate_in_kernel_gqa(
    framework_tracker,
    B: int,
    T: int,
    Hq: int,
    H: int,
    D: int,
    scale: float,
    has_dt_bias: bool,
    dtype,
):
    """Test use_gate_in_kernel=True with grouped value attention (HV > H)."""
    paddle.seed(42)
    assert H % Hq == 0

    q = paddle.rand([B, T, Hq, D], dtype=dtype)
    k = paddle.rand([B, T, Hq, D], dtype=dtype)
    v = paddle.rand([B, T, H, D], dtype=dtype)
    beta = paddle.rand([B, T, H], dtype=paddle.float32).sigmoid()
    g_raw = paddle.randn([B, T, H], dtype=paddle.float32)
    A_log = paddle.randn([H], dtype=paddle.float32)
    dt_bias = paddle.randn([H], dtype=paddle.float32) if has_dt_bias else None
    h0 = paddle.zeros([B, H, D, D], dtype=paddle.float32)

    for t in [q, k, v, beta, g_raw, h0]:
        t.stop_gradient = False
    A_log.stop_gradient = False
    if dt_bias is not None:
        dt_bias.stop_gradient = False

    tri, tri_ht = chunk_gated_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g_raw.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        A_log=A_log.clone(),
        dt_bias=dt_bias.clone() if dt_bias is not None else None,
    )
    do = paddle.randn(v.shape, dtype=v.dtype)
    dht = paddle.randn(h0.shape, dtype=h0.dtype)
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dh0 = (
        q.grad.clone(), k.grad.clone(), v.grad.clone(),
        beta.grad.clone(), g_raw.grad.clone(), h0.grad.clone()
    )
    tri_dA_log = A_log.grad.clone()
    tri_ddt_bias = dt_bias.grad.clone() if dt_bias is not None else None
    q.clear_gradient()
    k.clear_gradient()
    v.clear_gradient()
    beta.clear_gradient()
    g_raw.clear_gradient()
    h0.clear_gradient()
    A_log.clear_gradient()
    if dt_bias is not None:
        dt_bias.clear_gradient()

    g_ref = naive_gdn_gate(g_raw, A_log, dt_bias)
    ref, ref_ht = chunk_gated_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g_ref,
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dh0 = (
        q.grad.clone(), k.grad.clone(), v.grad.clone(),
        beta.grad.clone(), h0.grad.clone()
    )
    ref_dg = g_raw.grad.clone()
    ref_dA_log = A_log.grad.clone()
    ref_ddt_bias = dt_bias.grad.clone() if dt_bias is not None else None

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.008)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.008)
    assert_close('db', ref_dbeta, tri_dbeta, 0.02)
    assert_close('dg', ref_dg, tri_dg, 0.02)
    assert_close('dh0', ref_dh0, tri_dh0, 0.008)
    assert_close('dA_log', ref_dA_log, tri_dA_log, 0.02)
    if dt_bias is not None:
        assert_close('ddt_bias', ref_ddt_bias, tri_ddt_bias, 0.02)
    assert framework_tracker.detect_tensor_framework(tri) == 'paddle'
    assert framework_tracker.detect_triton_driver() == 'paddle'


@pytest.mark.parametrize(
    ('H', 'D', 'has_dt_bias', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-D{}-has_dt_bias{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (4, 60, True, [0, 15], paddle.float16),
            (4, 64, False, [0, 256, 500, 1000], paddle.float16),
            (4, 64, True, [0, 256, 500, 1000], paddle.float16),
            (4, 100, True, [0, 15, 100, 300, 1200, 2000], paddle.float16),
        ]
    ],
)
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '1',
    reason='Skipping test because SKIP_TEST_CHUNK_VARLEN is set',
)
def test_chunk_gate_in_kernel_varlen(
    framework_tracker,
    H: int,
    D: int,
    has_dt_bias: bool,
    cu_seqlens: list,
    dtype,
):
    """Test use_gate_in_kernel=True with variable-length sequences."""
    paddle.seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    cu_seqlens_t = paddle.to_tensor(cu_seqlens, dtype=paddle.int64)
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    q = paddle.randn([1, T, H, D], dtype=dtype)
    k = paddle.randn([1, T, H, D], dtype=dtype)
    v = paddle.randn([1, T, H, D], dtype=dtype)
    beta = paddle.rand([1, T, H], dtype=paddle.float32).sigmoid()
    g_raw = paddle.randn([1, T, H], dtype=paddle.float32)
    A_log = paddle.randn([H], dtype=paddle.float32)
    dt_bias = paddle.randn([H], dtype=paddle.float32) if has_dt_bias else None
    h0 = paddle.randn([N, H, D, D], dtype=paddle.float32)

    for t in [q, k, v, beta, g_raw, h0]:
        t.stop_gradient = False
    A_log.stop_gradient = False
    if dt_bias is not None:
        dt_bias.stop_gradient = False

    do = paddle.randn(v.shape, dtype=v.dtype)
    dht = paddle.rand(h0.shape, dtype=h0.dtype)

    tri, tri_ht = chunk_gated_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g_raw.clone(),
        beta=beta.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens_t,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        A_log=A_log.clone(),
        dt_bias=dt_bias.clone() if dt_bias is not None else None,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dh0 = (
        q.grad.clone(), k.grad.clone(), v.grad.clone(),
        beta.grad.clone(), g_raw.grad.clone(), h0.grad.clone()
    )
    tri_dA_log = A_log.grad.clone()
    tri_ddt_bias = dt_bias.grad.clone() if dt_bias is not None else None
    q.clear_gradient()
    k.clear_gradient()
    v.clear_gradient()
    beta.clear_gradient()
    g_raw.clear_gradient()
    h0.clear_gradient()
    A_log.clear_gradient()
    if dt_bias is not None:
        dt_bias.clear_gradient()

    g_ref = naive_gdn_gate(g_raw, A_log, dt_bias)
    ref, ref_ht = chunk_gated_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g_ref,
        beta=beta.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens_t,
        use_qk_l2norm_in_kernel=True,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dh0 = (
        q.grad.clone(), k.grad.clone(), v.grad.clone(),
        beta.grad.clone(), h0.grad.clone()
    )
    ref_dg = g_raw.grad.clone()
    ref_dA_log = A_log.grad.clone()
    ref_ddt_bias = dt_bias.grad.clone() if dt_bias is not None else None

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.008)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.008)
    assert_close('db', ref_dbeta, tri_dbeta, 0.02)
    assert_close('dg', ref_dg, tri_dg, 0.02)
    assert_close('dh0', ref_dh0, tri_dh0, 0.008)
    assert_close('dA_log', ref_dA_log, tri_dA_log, 0.02)
    if dt_bias is not None:
        assert_close('ddt_bias', ref_ddt_bias, tri_ddt_bias, 0.02)
    assert framework_tracker.detect_tensor_framework(tri) == 'paddle'
    assert framework_tracker.detect_triton_driver() == 'paddle'


@pytest.mark.parametrize(
    ('B', 'T', 'HV', 'HAS_BIAS'),
    [
        pytest.param(*test, id="B{}-T{}-HV{}-bias{}".format(*test))
        for test in [
            (1, 32, 2, False),
            (2, 64, 4, True),
            (4, 128, 8, True),
            (4, 128, 16, False),
        ]
    ],
)
def test_gate(
    framework_tracker,
    B: int,
    T: int,
    HV: int,
    HAS_BIAS: bool,
):
    paddle.seed(42)
    g = paddle.randn([B, T, HV], dtype=paddle.float32)
    A_log = paddle.log(paddle.uniform([HV], dtype=paddle.float32, min=1, max=16))
    dt_bias = paddle.randn([HV], dtype=paddle.float32) if HAS_BIAS else None
    g.stop_gradient = False
    A_log.stop_gradient = False
    if dt_bias is not None:
        dt_bias.stop_gradient = False
    do = paddle.randn([B, T, HV], dtype=paddle.float32)

    ref = naive_gdn_gate(
        g.clone(), A_log.clone(), dt_bias.clone() if dt_bias is not None else None,
    )
    tri = fused_gdn_gate(
        g.clone(), A_log.clone(), dt_bias.clone() if dt_bias is not None else None,
    )
    (ref * do).sum().backward(retain_graph=True)

    ref_dg = g.grad.clone()
    ref_dA = A_log.grad.clone()
    ref_dbias = dt_bias.grad.clone() if dt_bias is not None else None
    g.clear_gradient()
    A_log.clear_gradient()
    if dt_bias is not None:
        dt_bias.clear_gradient()

    (tri * do).sum().backward(retain_graph=True)
    tri_dg = g.grad.clone()
    tri_dA = A_log.grad.clone()
    tri_dbias = dt_bias.grad.clone() if dt_bias is not None else None

    assert_close("o", ref, tri, 1e-4)
    assert_close("dg", ref_dg, tri_dg, 1e-4)
    assert_close("dA", ref_dA, tri_dA, 1e-4)
    if HAS_BIAS:
        assert_close("dbias", ref_dbias, tri_dbias, 1e-4)
    assert framework_tracker.detect_tensor_framework(tri) == 'paddle'
    assert framework_tracker.detect_triton_driver() == 'paddle'
