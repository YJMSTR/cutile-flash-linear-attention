# -*- coding: utf-8 -*-

import fla_paddle.utils as paddle_utils


def test_input_guard_resolves_signature_once(monkeypatch):
    calls = {'count': 0}
    real_signature = paddle_utils.inspect.signature

    def fake_signature(fn):
        calls['count'] += 1
        return real_signature(fn)

    monkeypatch.setattr(paddle_utils.inspect, 'signature', fake_signature)

    @paddle_utils.input_guard
    def fn(x, y=None):
        return x, y

    assert calls['count'] == 1

    fn(1, y=2)
    fn(3, y=4)

    assert calls['count'] == 1
