from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import matplotlib.pyplot as plt

WORD_LIST_URL = 'https://www.gutenberg.org/files/3201/files/crosswd.txt'
WORD_LIST_CACHE_PATH = Path(__file__).with_name('.word_list_cache.txt')


@dataclass
class DistractorParams:
    position: tuple[float, float]
    kwargs: dict


@dataclass
class TraceParams:
    x: np.ndarray
    y: np.ndarray
    label: str
    kwargs: dict


@dataclass
class PlotParams:
    traces: list[TraceParams]
    spacing: float
    distractor: DistractorParams | None
    intersect: bool
    hide_grid: bool = False


def _load_word_list(cache_path: Path = WORD_LIST_CACHE_PATH) -> list[str]:
    if cache_path.exists():
        words = [w.strip() for w in cache_path.read_text().splitlines() if w.strip().isalpha()]
        if len(words) >= 2:
            return words

    try:
        with urlopen(WORD_LIST_URL, timeout=5) as response:
            data = response.read().decode('utf-8')
        words = [w.strip() for w in data.splitlines() if w.strip().isalpha()]
        if len(words) >= 2:
            cache_path.write_text('\n'.join(words))
            return words
    except Exception:
        pass

    # Fallback prevents generation failures if network download is unavailable.
    words = ['alpha', 'beta', 'gamma', 'delta', 'omega', 'sigma', 'lambda', 'tau']
    return words


def _random_labels() -> tuple[str, str]:
    words = _load_word_list()
    selected = np.random.choice(words, size=2, replace=False)
    return selected[0].title(), selected[1].title()


def _random_colors() -> tuple[str, str]:
    palette = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['tab:blue', 'tab:orange'])
    selected = np.random.choice(palette, size=2, replace=False)
    return selected[0], selected[1]


def _random_line_kwargs(color: str, dashed_bias: bool) -> dict:
    marker = np.random.choice(['', '', '', 'o', 's', '^', 'x'])
    return dict(
        color=color,
        linewidth=float(np.random.uniform(0.8, 2.2)),
        alpha=float(np.random.uniform(0.7, 1.0)),
        linestyle=np.random.choice(['--', '-.', ':']) if dashed_bias else np.random.choice(['-', '-', '--']),
        marker=marker,
        markevery=(0, np.random.randint(40, 120)) if marker else None
    )


def gen_random_trace_pair_params(intersect_prob: float = 0.5) -> tuple[list[TraceParams], bool]:
    x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
    intersect_prob = float(np.clip(intersect_prob, 0.0, 1.0))
    intersect = np.random.rand() < intersect_prob
    line_label, curve_label = _random_labels()
    line_color, curve_color = _random_colors()
    line_is_dashed = np.random.rand() < 0.5

    m = np.random.uniform(-1.5, 1.5)
    b = np.random.uniform(-3.0, 3.0)
    y_line = m * x + b

    # second_is_line = np.random.rand() < 0.5
    second_is_line = False

    if second_is_line:
        if intersect:
            m2 = m
            while abs(m2 - m) < 0.15:
                m2 = np.random.uniform(-1.5, 1.5)
            b2 = np.random.uniform(-3.0, 3.0)
        else:
            m2 = np.random.uniform(-1.5, 1.5)
            k = m2 - m
            # Enforce a clear minimum gap at the closest approach inside the plot bounds.
            min_clear_gap = 1.0
            gap = np.random.uniform(min_clear_gap, 2.4)
            side_sign = np.random.choice([-1.0, 1.0])
            x_min = float(x.min())
            x_max = float(x.max())

            if side_sign > 0:
                # Ensure y_curve - y_line stays positive on [x_min, x_max].
                x_edge = x_min if k > 0 else x_max
                b2 = b + gap - k * x_edge
            else:
                # Ensure y_curve - y_line stays negative on [x_min, x_max].
                x_edge = x_max if k > 0 else x_min
                b2 = b - gap - k * x_edge
        y_curve = m2 * x + b2
    else:
        nonlinear_family = np.random.choice(['cubic', 'tanh', 'exp', 'asinh'])
        scale = np.max(np.abs(x))
        x_norm = x / scale

        if nonlinear_family == 'cubic':
            phi = x_norm ** 3
        elif nonlinear_family == 'tanh':
            k = np.random.uniform(1.2, 3.0)
            phi = np.tanh(k * x_norm)
        elif nonlinear_family == 'exp':
            k = np.random.uniform(1.5, 3.0)
            phi = np.expm1(k * x_norm)
        else:
            k = np.random.uniform(1.0, 3.0)
            phi = np.arcsinh(k * x_norm)

        phi_min = float(phi.min())
        phi_max = float(phi.max())
        span = phi_max - phi_min

        # margin = max(0.1 * span, 0.1)
        margin = 0

        if intersect:
            # c in (phi_min, phi_max) guarantees one crossing since phi is monotonic.
            c = np.random.uniform(phi_min + 0.1 * span, phi_max - 0.1 * span)
        else:
            c = (
                np.random.uniform(phi_max + margin, phi_max + 2.0 * margin)
                if np.random.rand() < 0.5
                else np.random.uniform(phi_min - 2.0 * margin, phi_min - margin)
            )

        amp = np.random.uniform(0.8, 3.0)
        y_curve = y_line + amp * (phi - c)

    traces = [
        TraceParams(
            x=x,
            y=y_line,
            label=line_label,
            kwargs=_random_line_kwargs(line_color, dashed_bias=line_is_dashed)
        ),

        TraceParams(
            x=x,
            y=y_curve,
            label=curve_label,
            kwargs=_random_line_kwargs(curve_color, dashed_bias=not line_is_dashed)
        )
    ]
    return traces, intersect


def gen_random_distractor_params(traces: list[TraceParams]):
    a, b = traces
    return DistractorParams(
        position=(
            (
                0.0,
                np.random.uniform(min(a.y.min(), b.y.min()), max(a.y.max(), b.y.max()))
            )
            if np.random.rand() >= 0.5 else
            (
                np.random.uniform(min(a.x.min(), b.x.min()), max(a.x.max(), b.x.max())),
                0.0
            )
        ),
        kwargs=dict(
            marker=np.random.choice(['o', 's', '^', 'D', '*', 'P']),
            color=np.random.choice(['black', '#333333', '#555555', '#1f1f1f']),
            markersize=float(np.random.uniform(5.0, 11.0)),
            alpha=float(np.random.uniform(0.7, 1.0))
        )
    )


def gen_random_plot_params(
    distractor_prop=0.5,
    intersect_prob=0.5
):
    traces, intersect = gen_random_trace_pair_params(intersect_prob=intersect_prob)
    return PlotParams(
        traces=traces,
        spacing=np.random.randint(5, 10),
        intersect=intersect,
        distractor=(
            gen_random_distractor_params(traces=traces)
            if np.random.rand() <= distractor_prop
            else None
        )
    )


def render_plot(params: PlotParams, show=False):
    width = float(np.random.uniform(8.5, 11.5))
    height = float(np.random.uniform(5.0, 7.0))
    fig = plt.figure(figsize=(width, height))
    ax = fig.gca()
    fig.patch.set_facecolor(np.random.choice(['white', '#fcfcff', '#fffdf7', '#f9fcff']))
    ax.set_facecolor(np.random.choice(['white', '#fdfdfd', '#fafafa']))

    for trace in params.traces:
        plt.plot(trace.x, trace.y, label=trace.label, **trace.kwargs)

    if params.distractor is not None:
        plt.plot(*params.distractor.position, **params.distractor.kwargs)

    if params.hide_grid:
        ax.tick_params(
            axis='both',
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False
        )
    else:
        axis_color = np.random.choice(['black', '#222222', '#444444'])
        plt.axhline(0, color=axis_color, linewidth=float(np.random.uniform(0.4, 0.9)))
        plt.axvline(0, color=axis_color, linewidth=float(np.random.uniform(0.4, 0.9)))
        plt.grid(
            True,
            alpha=float(np.random.uniform(0.2, 0.45)),
            linestyle=np.random.choice(['-', '--', ':']),
            linewidth=float(np.random.uniform(0.4, 0.9))
        )

    plt.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        framealpha=float(np.random.uniform(0.75, 1.0))
    )

    fig.subplots_adjust(right=0.8)

    if show:
        plt.show()

    return fig


if __name__ == '__main__':
    render_plot(gen_random_plot_params(), show=True)
