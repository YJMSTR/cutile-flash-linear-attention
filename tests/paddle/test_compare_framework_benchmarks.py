# -*- coding: utf-8 -*-

import importlib.util
import json
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / 'scripts' / 'compare_framework_benchmarks.py'
    spec = importlib.util.spec_from_file_location('compare_framework_benchmarks', script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_compare_results_computes_ratios_for_matching_rows(tmp_path):
    module = _load_module()

    torch_payload = {
        'machine_info': {'gpu_name': 'GPU'},
        'results': [
            {'op': 'chunk_gdn', 'mode': 'fwd', 'B': 1, 'T': 512, 'H': 8, 'D': 64, 'median_ms': 1.0, 'p20_ms': 0.9, 'p80_ms': 1.1},
            {'op': 'chunk_kda', 'mode': 'fwdbwd', 'B': 2, 'T': 1024, 'H': 8, 'D': 64, 'median_ms': 4.0, 'p20_ms': 3.8, 'p80_ms': 4.2},
        ],
    }
    paddle_payload = {
        'machine_info': {'gpu_name': 'GPU'},
        'results': [
            {'op': 'chunk_gdn', 'mode': 'fwd', 'B': 1, 'T': 512, 'H': 8, 'D': 64, 'median_ms': 1.5, 'p20_ms': 1.4, 'p80_ms': 1.6},
            {'op': 'chunk_kda', 'mode': 'fwdbwd', 'B': 2, 'T': 1024, 'H': 8, 'D': 64, 'median_ms': 5.0, 'p20_ms': 4.9, 'p80_ms': 5.2},
        ],
    }

    torch_path = tmp_path / 'torch.json'
    paddle_path = tmp_path / 'paddle.json'
    torch_path.write_text(json.dumps(torch_payload))
    paddle_path.write_text(json.dumps(paddle_payload))

    report = module.build_comparison_report(torch_path, paddle_path)

    assert report['summary']['row_count'] == 2
    assert report['rows'][0]['paddle_over_torch'] == 1.5
    assert report['rows'][1]['paddle_over_torch'] == 1.25


def test_write_chart_and_report_outputs_files(tmp_path):
    module = _load_module()

    report = {
        'torch_machine_info': {'gpu_name': 'GPU'},
        'paddle_machine_info': {'gpu_name': 'GPU'},
        'rows': [
            {
                'op': 'chunk_gdn', 'mode': 'fwd', 'B': 1, 'T': 512, 'H': 8, 'D': 64,
                'torch_median_ms': 1.0, 'paddle_median_ms': 1.2, 'paddle_over_torch': 1.2,
            },
            {
                'op': 'chunk_gdn', 'mode': 'fwd', 'B': 1, 'T': 1024, 'H': 8, 'D': 64,
                'torch_median_ms': 1.4, 'paddle_median_ms': 1.7, 'paddle_over_torch': 1.214286,
            },
            {
                'op': 'chunk_kda', 'mode': 'fwdbwd', 'B': 2, 'T': 1024, 'H': 8, 'D': 64,
                'torch_median_ms': 4.0, 'paddle_median_ms': 5.0, 'paddle_over_torch': 1.25,
            },
            {
                'op': 'chunk_kda', 'mode': 'fwdbwd', 'B': 2, 'T': 2048, 'H': 8, 'D': 64,
                'torch_median_ms': 7.0, 'paddle_median_ms': 8.4, 'paddle_over_torch': 1.2,
            },
        ],
        'summary': {'row_count': 4},
    }

    report_path = tmp_path / 'report.json'
    chart_path = tmp_path / 'chart.svg'

    module.write_comparison_report(report, report_path)
    module.plot_comparison_chart(report, chart_path)
    svg = chart_path.read_text()

    assert report_path.exists()
    assert chart_path.exists()
    assert chart_path.stat().st_size > 0
    assert 'chunk_gdn (Forward)' in svg
    assert 'chunk_kda (Forward + Backward)' in svg
    assert 'Speedup: &gt;1 means Torch faster, &lt;1 means Paddle faster' in svg
    assert '120.0%' in svg
    assert '125.0%' in svg
    assert '1024' in svg
    assert '2048' in svg


def test_build_benchmark_env_uses_requested_warmup_and_rep():
    module = _load_module()

    env = module.build_benchmark_env(warmup_ms=100, rep_ms=500, include_paddle_flag=True)

    assert env['FLA_BENCH_WARMUP_MS'] == '100'
    assert env['FLA_BENCH_REP_MS'] == '500'
    assert env['FLA_BENCHMARK'] == '1'


def test_parse_args_defaults_include_recurrent_ops():
    module = _load_module()

    args = module.parse_args([])

    assert args.op == ['chunk_gdn', 'recurrent_gdn', 'chunk_kda', 'recurrent_kda']
