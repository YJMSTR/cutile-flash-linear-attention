from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from html import escape
import math

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TORCH_JSON = 'torch_results.json'
DEFAULT_PADDLE_JSON = 'paddle_results.json'
DEFAULT_REPORT_JSON = 'comparison_report.json'
DEFAULT_CHART_SVG = 'comparison_chart.svg'
SERIES_COLORS = ['#2ca02c', '#d62728', '#9467bd', '#17becf', '#8c564b', '#e377c2']


def make_result_key(result: dict) -> tuple:
    return (
        result['op'],
        result['mode'],
        result['B'],
        result['T'],
        result['H'],
        result['D'],
    )


def make_row_label(row: dict) -> str:
    return f"{row['op']}\n{row['mode']}\nB{row['B']}_T{row['T']}_H{row['H']}_D{row['D']}"


def load_benchmark_payload(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def build_comparison_report(torch_json: str | Path, paddle_json: str | Path) -> dict:
    torch_payload = load_benchmark_payload(torch_json)
    paddle_payload = load_benchmark_payload(paddle_json)

    torch_results = torch_payload.get('results', [])
    paddle_results = paddle_payload.get('results', [])
    paddle_map = {make_result_key(result): result for result in paddle_results}

    rows = []
    missing_in_paddle = []
    for torch_result in torch_results:
        key = make_result_key(torch_result)
        paddle_result = paddle_map.get(key)
        if paddle_result is None:
            missing_in_paddle.append({
                'op': torch_result['op'],
                'mode': torch_result['mode'],
                'B': torch_result['B'],
                'T': torch_result['T'],
                'H': torch_result['H'],
                'D': torch_result['D'],
            })
            continue
        torch_ms = float(torch_result['median_ms'])
        paddle_ms = float(paddle_result['median_ms'])
        ratio = round(paddle_ms / torch_ms, 6) if torch_ms else None
        rows.append({
            'op': torch_result['op'],
            'mode': torch_result['mode'],
            'B': torch_result['B'],
            'T': torch_result['T'],
            'H': torch_result['H'],
            'D': torch_result['D'],
            'torch_median_ms': torch_ms,
            'paddle_median_ms': paddle_ms,
            'paddle_over_torch': ratio,
        })

    rows.sort(key=lambda row: (row['mode'], row['op'], row['B'], row['T'], row['H'], row['D']))
    ratios = [row['paddle_over_torch'] for row in rows if row['paddle_over_torch'] is not None]
    summary = {
        'row_count': len(rows),
        'missing_in_paddle_count': len(missing_in_paddle),
        'avg_paddle_over_torch': round(sum(ratios) / len(ratios), 6) if ratios else None,
        'max_paddle_over_torch': max(ratios) if ratios else None,
        'min_paddle_over_torch': min(ratios) if ratios else None,
    }
    return {
        'torch_machine_info': torch_payload.get('machine_info', {}),
        'paddle_machine_info': paddle_payload.get('machine_info', {}),
        'rows': rows,
        'missing_in_paddle': missing_in_paddle,
        'summary': summary,
    }


def write_comparison_report(report: dict, output_path: str | Path) -> None:
    path = Path(output_path)
    path.write_text(json.dumps(report, indent=2))


def format_mode_label(mode: str) -> str:
    return 'Forward' if mode == 'fwd' else 'Forward + Backward'


def build_benchmark_env(warmup_ms: int, rep_ms: int, include_paddle_flag: bool = False) -> dict[str, str]:
    env = dict(os.environ)
    env['FLA_BENCH_WARMUP_MS'] = str(warmup_ms)
    env['FLA_BENCH_REP_MS'] = str(rep_ms)
    if include_paddle_flag:
        env['FLA_BENCHMARK'] = '1'
    return env


def _panel_bounds(index: int, columns: int, panel_width: int, panel_height: int, gutter_x: int, gutter_y: int, top_offset: int):
    row = index // columns
    col = index % columns
    x = 30 + col * (panel_width + gutter_x)
    y = top_offset + row * (panel_height + gutter_y)
    return x, y


def _render_latency_panel(parts: list[str], title: str, rows: list[dict], x: int, y: int, width: int, height: int):
    inner_left = x + 50
    inner_right = x + width - 20
    inner_top = y + 35
    inner_bottom = y + height - 55
    chart_width = inner_right - inner_left
    chart_height = inner_bottom - inner_top
    max_latency = max(max(row['torch_median_ms'], row['paddle_median_ms']) for row in rows) * 1.15
    x_labels = [str(row['T']) for row in rows]
    step = chart_width / max(len(rows) - 1, 1)
    points_torch = []
    points_paddle = []

    parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="#fff" stroke="#bbb"/>')
    parts.append(f'<text x="{x + width / 2:.1f}" y="{y + 22}" text-anchor="middle" class="paneltitle">{escape(title)}</text>')
    parts.append(f'<line x1="{inner_left}" y1="{inner_bottom}" x2="{inner_right}" y2="{inner_bottom}" stroke="#333"/>')
    parts.append(f'<line x1="{inner_left}" y1="{inner_top}" x2="{inner_left}" y2="{inner_bottom}" stroke="#333"/>')
    parts.append(f'<text x="{x + 16}" y="{y + height / 2:.1f}" transform="rotate(-90 {x + 16} {y + height / 2:.1f})" class="small">Execution Time (ms)</text>')
    parts.append(f'<text x="{x + width / 2:.1f}" y="{y + height - 14}" text-anchor="middle" class="small">Sequence Length (T)</text>')

    for tick_index in range(5):
        tick_value = max_latency * tick_index / 4
        tick_y = inner_bottom - chart_height * tick_index / 4
        parts.append(f'<line x1="{inner_left}" y1="{tick_y:.1f}" x2="{inner_right}" y2="{tick_y:.1f}" stroke="#e5e5e5"/>')
        parts.append(f'<text x="{inner_left - 8}" y="{tick_y + 4:.1f}" text-anchor="end" class="small">{tick_value:.1f}</text>')

    for index, row in enumerate(rows):
        cx = inner_left + step * index if len(rows) > 1 else inner_left + chart_width / 2
        torch_y = inner_bottom - (row['torch_median_ms'] / max_latency) * chart_height
        paddle_y = inner_bottom - (row['paddle_median_ms'] / max_latency) * chart_height
        points_torch.append((cx, torch_y))
        points_paddle.append((cx, paddle_y))
        parts.append(f'<line x1="{cx:.1f}" y1="{inner_top}" x2="{cx:.1f}" y2="{inner_bottom}" stroke="#f0f0f0"/>')
        parts.append(f'<text x="{cx:.1f}" y="{inner_bottom + 18}" text-anchor="middle" class="small">{x_labels[index]}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{paddle_y - 8:.1f}" text-anchor="middle" class="small">{row["paddle_over_torch"] * 100:.1f}%</text>')

    parts.append(f'<polyline fill="none" stroke="#1f77b4" stroke-width="2" points="{" ".join(f"{px:.1f},{py:.1f}" for px, py in points_paddle)}"/>')
    parts.append(f'<polyline fill="none" stroke="#ff7f0e" stroke-width="2" stroke-dasharray="6,4" points="{" ".join(f"{px:.1f},{py:.1f}" for px, py in points_torch)}"/>')
    for px, py in points_paddle:
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.5" fill="#1f77b4"/>')
    for px, py in points_torch:
        parts.append(f'<rect x="{px - 3:.1f}" y="{py - 3:.1f}" width="6" height="6" fill="#ff7f0e"/>')
    parts.append(f'<text x="{inner_left + 10}" y="{inner_top + 10}" class="small">Paddle</text><circle cx="{inner_left + 2}" cy="{inner_top + 6}" r="3.5" fill="#1f77b4"/>')
    parts.append(f'<text x="{inner_left + 90}" y="{inner_top + 10}" class="small">Torch</text><rect x="{inner_left + 76}" y="{inner_top + 2}" width="6" height="6" fill="#ff7f0e"/>')


def _render_speedup_panel(parts: list[str], series_map: dict[tuple[str, str], list[dict]], x: int, y: int, width: int, height: int):
    inner_left = x + 50
    inner_right = x + width - 20
    inner_top = y + 35
    inner_bottom = y + height - 55
    chart_width = inner_right - inner_left
    chart_height = inner_bottom - inner_top
    t_values = sorted({row['T'] for series in series_map.values() for row in series})
    group_step = chart_width / max(len(t_values), 1)
    bar_group_width = group_step * 0.7
    series_keys = sorted(series_map)
    bar_width = bar_group_width / max(len(series_keys), 1)

    parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="#fff" stroke="#bbb"/>')
    parts.append(f'<text x="{x + width / 2:.1f}" y="{y + 22}" text-anchor="middle" class="paneltitle">Speedup: &gt;1 means Torch faster, &lt;1 means Paddle faster</text>')
    parts.append(f'<line x1="{inner_left}" y1="{inner_bottom}" x2="{inner_right}" y2="{inner_bottom}" stroke="#333"/>')
    parts.append(f'<line x1="{inner_left}" y1="{inner_top}" x2="{inner_left}" y2="{inner_bottom}" stroke="#333"/>')
    parts.append(f'<text x="{x + 16}" y="{y + height / 2:.1f}" transform="rotate(-90 {x + 16} {y + height / 2:.1f})" class="small">Speedup (Torch / Paddle)</text>')
    parts.append(f'<text x="{x + width / 2:.1f}" y="{y + height - 14}" text-anchor="middle" class="small">Sequence Length (T)</text>')
    max_speedup = max(max(1 / row['paddle_over_torch'] for row in series) for series in series_map.values()) * 1.1
    for tick_index in range(6):
        tick_value = max_speedup * tick_index / 5
        tick_y = inner_bottom - chart_height * tick_index / 5
        parts.append(f'<line x1="{inner_left}" y1="{tick_y:.1f}" x2="{inner_right}" y2="{tick_y:.1f}" stroke="#e5e5e5"/>')
        parts.append(f'<text x="{inner_left - 8}" y="{tick_y + 4:.1f}" text-anchor="end" class="small">{tick_value:.2f}</text>')
    unit_y = inner_bottom - chart_height * (1.0 / max_speedup)
    parts.append(f'<line x1="{inner_left}" y1="{unit_y:.1f}" x2="{inner_right}" y2="{unit_y:.1f}" stroke="#888" stroke-dasharray="6,4"/>')

    for group_index, t_value in enumerate(t_values):
        group_x = inner_left + group_step * group_index + (group_step - bar_group_width) / 2
        parts.append(f'<text x="{group_x + bar_group_width / 2:.1f}" y="{inner_bottom + 18}" text-anchor="middle" class="small">{t_value}</text>')
        for series_index, series_key in enumerate(series_keys):
            row = next((item for item in series_map[series_key] if item['T'] == t_value), None)
            if row is None:
                continue
            speedup = 1 / row['paddle_over_torch']
            bar_height = (speedup / max_speedup) * chart_height
            bar_x = group_x + series_index * bar_width
            bar_y = inner_bottom - bar_height
            color = SERIES_COLORS[series_index % len(SERIES_COLORS)]
            parts.append(f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_width * 0.9:.1f}" height="{bar_height:.1f}" fill="{color}"/>')

    legend_x = inner_left
    legend_y = inner_top + 10
    for series_index, (op, mode) in enumerate(series_keys):
        color = SERIES_COLORS[series_index % len(SERIES_COLORS)]
        label = f'{op} ({format_mode_label(mode)})'
        parts.append(f'<rect x="{legend_x:.1f}" y="{legend_y + series_index * 16:.1f}" width="10" height="10" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 16:.1f}" y="{legend_y + series_index * 16 + 9:.1f}" class="small">{escape(label)}</text>')


def plot_comparison_chart(report: dict, output_path: str | Path) -> None:
    rows = report.get('rows', [])
    if not rows:
        raise ValueError('No comparable rows found for plotting.')

    series_map: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        series_map.setdefault((row['op'], row['mode']), []).append(row)
    for key in series_map:
        series_map[key].sort(key=lambda row: (row['T'], row['B'], row['H'], row['D']))

    panel_titles = [f'{op} ({format_mode_label(mode)})' for op, mode in sorted(series_map)]
    total_panels = len(panel_titles) + 1
    columns = 2
    panel_width = 580
    panel_height = 300
    gutter_x = 20
    gutter_y = 20
    top_offset = 70
    total_rows = math.ceil(total_panels / columns)
    width = 30 + columns * panel_width + (columns - 1) * gutter_x + 30
    height = top_offset + total_rows * panel_height + (total_rows - 1) * gutter_y + 30

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;font-size:12px} .title{font-size:22px;font-weight:bold} .paneltitle{font-size:16px;font-weight:bold} .small{font-size:11px;fill:#444}</style>',
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" class="title">Framework Performance: Paddle vs Torch Backend</text>',
    ]

    sorted_keys = sorted(series_map)
    for index, key in enumerate(sorted_keys):
        x, y = _panel_bounds(index, columns, panel_width, panel_height, gutter_x, gutter_y, top_offset)
        _render_latency_panel(parts, f'{key[0]} ({format_mode_label(key[1])})', series_map[key], x, y, panel_width, panel_height)

    speedup_index = len(sorted_keys)
    x, y = _panel_bounds(speedup_index, columns, panel_width, panel_height, gutter_x, gutter_y, top_offset)
    _render_speedup_panel(parts, series_map, x, y, panel_width, panel_height)

    parts.append('</svg>')
    Path(output_path).write_text(''.join(parts))


def run_command(command: list[str], env: dict | None = None) -> None:
    result = subprocess.run(command, cwd=str(PROJECT_ROOT), text=True, env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)



def build_runner_commands(args, output_dir: Path) -> tuple[list[str], list[str], Path, Path]:
    torch_json = output_dir / DEFAULT_TORCH_JSON
    paddle_json = output_dir / DEFAULT_PADDLE_JSON

    torch_command = [
        sys.executable,
        '-m',
        'benchmarks.ops.run',
        '--op',
        *args.op,
        '--json',
        str(torch_json),
        '--modes',
        *args.modes,
    ]
    paddle_command = [
        sys.executable,
        '-m',
        'benchmarks.paddle_ops.run',
        '--op',
        *args.op,
        '--json',
        str(paddle_json),
        '--modes',
        *args.modes,
    ]
    if args.custom_shapes:
        torch_command.extend(['--custom-shapes', args.custom_shapes])
        paddle_command.extend(['--custom-shapes', args.custom_shapes])
    return torch_command, paddle_command, torch_json, paddle_json


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Run torch and paddle benchmarks together and generate comparison charts.')
    parser.add_argument('--op', nargs='+', default=['chunk_gdn', 'recurrent_gdn', 'chunk_kda', 'recurrent_kda'], help='Operator names to benchmark')
    parser.add_argument('--modes', nargs='+', default=['fwd', 'fwdbwd'], choices=['fwd', 'fwdbwd'], help='Benchmark modes')
    parser.add_argument('--custom-shapes', default=None, help='Optional JSON string for custom benchmark shapes')
    parser.add_argument('--output-dir', default='benchmark_outputs/framework_compare', help='Directory for raw results and charts')
    parser.add_argument('--skip-run', action='store_true', help='Skip benchmark execution and only compare existing JSON files')
    parser.add_argument('--torch-json', default=None, help='Existing torch benchmark JSON when using --skip-run')
    parser.add_argument('--paddle-json', default=None, help='Existing paddle benchmark JSON when using --skip-run')
    parser.add_argument('--warmup-ms', type=int, default=100, help='Benchmark warmup time in milliseconds')
    parser.add_argument('--rep-ms', type=int, default=500, help='Benchmark repetition time in milliseconds')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_run:
        if not args.torch_json or not args.paddle_json:
            raise SystemExit('--skip-run requires --torch-json and --paddle-json')
        torch_json = Path(args.torch_json)
        paddle_json = Path(args.paddle_json)
    else:
        torch_command, paddle_command, torch_json, paddle_json = build_runner_commands(args, output_dir)
        torch_env = build_benchmark_env(args.warmup_ms, args.rep_ms)
        paddle_env = build_benchmark_env(args.warmup_ms, args.rep_ms, include_paddle_flag=True)
        print('Running torch benchmark:')
        print('  ' + ' '.join(torch_command))
        print(f'  warmup={args.warmup_ms}ms rep={args.rep_ms}ms')
        run_command(torch_command, env=torch_env)

        print('Running paddle benchmark:')
        print('  ' + ' '.join(paddle_command))
        print(f'  warmup={args.warmup_ms}ms rep={args.rep_ms}ms')
        run_command(paddle_command, env=paddle_env)

    report = build_comparison_report(torch_json, paddle_json)
    report_path = output_dir / DEFAULT_REPORT_JSON
    chart_path = output_dir / DEFAULT_CHART_SVG
    write_comparison_report(report, report_path)
    plot_comparison_chart(report, chart_path)

    print(f"Compared rows: {report['summary']['row_count']}")
    print(f"Average paddle/torch ratio: {report['summary']['avg_paddle_over_torch']}")
    print(f"Report saved to {report_path}")
    print(f"Chart saved to {chart_path}")
    return report


if __name__ == '__main__':
    main()
