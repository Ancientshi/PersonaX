"""Small offline visualization of the released select_samples implementation.

Run from the repository root: python -m experiments.sampling_demo.generate
No algorithm is reimplemented: selection and internal scores come from PersonaX.
"""

from pathlib import Path
import hashlib
import json
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.spatial.distance import cdist, pdist

from personax.sampling import select_samples


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
ALPHAS = [1, 1.001, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07,
          1.08, 1.09, 1.1, 1.12, 1.16, 1.2, 1.3, 1.4]
SCALES = [0.5, 1, 2]


def synthetic_points():
    rng = np.random.default_rng(42)
    dense = rng.normal(0, 0.30, (24, 2))
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    ring = np.column_stack((np.cos(angles), np.sin(angles)))
    ring *= rng.uniform(0.78, 1.2, (16, 1))
    boundary = np.array([[1.55, 0.1], [-1.4, 0.2], [0.35, 1.5], [-0.2, -1.45]])
    return np.vstack((dense, ring, boundary))


def diagnostics(points, selected):
    center = points.mean(axis=0)
    return {
        "mean_center_distance": float(np.linalg.norm(selected - center, axis=1).mean()),
        "mean_pairwise_distance": float(pdist(selected).mean()) if len(selected) > 1 else None,
        "mean_coverage_distance": float(cdist(points, selected).min(axis=1).mean()),
    }


def label_positions(ax, chosen):
    # Place selection-order labels around the points, avoiding label overlaps.
    placed = []
    for step, point in enumerate(chosen, 1):
        screen = ax.transData.transform(point)
        candidates = [(radius * np.cos(angle), radius * np.sin(angle))
                      for radius in (15, 24, 34)
                      for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
        def clearance(offset):
            position = screen + np.asarray(offset) * ax.figure.dpi / 72
            others = [np.linalg.norm(position - prior) for prior in placed]
            others += [np.linalg.norm(position - ax.transData.transform(p)) for p in chosen]
            return min(others)
        offset = max(candidates, key=clearance)
        placed.append(screen + np.asarray(offset) * ax.figure.dpi / 72)
        ax.annotate(str(step), point, xytext=offset, textcoords="offset points",
                    ha="center", va="center", fontsize=9, color="#164a7b",
                    arrowprops={"arrowstyle": "-", "lw": 0.55, "color": "#7294b1"},
                    bbox={"boxstyle": "round,pad=0.12", "facecolor": "white",
                          "edgecolor": "none", "alpha": 0.9})


def draw_panel(ax, points, alpha, k, show_weight=True):
    chosen, indices, scores = select_samples(points, alpha=alpha, num_samples=k)
    center = points.mean(axis=0)
    ax.scatter(*points.T, s=20, color="#c7ccd2", zorder=1)
    ax.scatter(*chosen.T, s=52, color="#2166a6", edgecolor="white", lw=0.7, zorder=3)
    ax.scatter(*center, s=95, marker="x", color="#ba4037", lw=1.8, zorder=4)
    ax.set(xlim=(-1.85, 1.85), ylim=(-1.85, 1.85), aspect="equal",
           xticks=[-1.5, 0, 1.5], yticks=[-1.5, 0, 1.5])
    ax.tick_params(labelsize=9, length=0, pad=5, colors="#6b7280")
    ax.grid(color="#e5e7eb", lw=0.6, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    weights = f"Center weight {alpha ** -10:.1%} · Diversity weight {1 - alpha ** -10:.1%}"
    title = f"alpha = {alpha:.3f}   |   num_samples = {k}"
    if show_weight:
        title += "\n" + weights
    ax.set_title(title, fontsize=12, pad=11, color="#172b3a")
    metrics = diagnostics(points, chosen)
    ax.text(0.5, -0.09,
            f"Center distance {metrics['mean_center_distance']:.3f}  |  Pair distance {metrics['mean_pairwise_distance']:.3f}\n"
            f"Coverage distance {metrics['mean_coverage_distance']:.3f}",
            ha="center", va="top", transform=ax.transAxes, fontsize=10, color="#4b5563")
    label_positions(ax, chosen)
    return {"alpha": alpha, "num_samples": k, "indices": indices, "metrics": metrics,
            "last_score": list(scores[-1])}


def main():
    OUTPUT.mkdir(exist_ok=True)
    plt.rcParams["axes.unicode_minus"] = False
    points = synthetic_points()
    source = ROOT / "personax/sampling.py"
    demo = {"points": points.tolist(), "seed": 42, "alphas": ALPHAS, "scales": SCALES,
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "runs": {}}
    for scale in SCALES:
        demo["runs"][str(scale)] = {}
        for alpha in ALPHAS:
            _, indices, scores = select_samples(points * scale, alpha=alpha, num_samples=len(points))
            # k does not enter the candidate score. Confirm representative prefixes.
            for k in (3, 8, 14):
                _, direct, direct_scores = select_samples(points * scale, alpha=alpha, num_samples=k)
                assert direct == indices[:k]
                np.testing.assert_allclose(direct_scores, scores[:k])
            demo["runs"][str(scale)][str(alpha)] = {"indices": indices, "scores": [list(s) for s in scores]}
    (OUTPUT / "sampling-data.json").write_text(json.dumps(demo, ensure_ascii=False), encoding="utf-8")

    figure, axes = plt.subplots(2, 4, figsize=(17.6, 12.8))
    figure.patch.set_facecolor("white")
    figure.subplots_adjust(left=0.04, right=0.985, top=0.78, bottom=0.17, hspace=0.6, wspace=0.23)
    figure.suptitle("select_samples: how parameters change the selection", fontsize=23, y=0.975, color="#172b3a")
    figure.text(0.5, 0.929, "44 synthetic 2D points · seed = 42 · actual repository function · fixed input and order", ha="center", fontsize=13, color="#56616c")
    legend = [Line2D([], [], marker="o", linestyle="none", color="#c7ccd2", label="Input points"),
              Line2D([], [], marker="o", linestyle="none", color="#2166a6", label="Selected points (numbers = selection order)"),
              Line2D([], [], marker="x", linestyle="none", color="#ba4037", label="Full input centroid")]
    figure.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=3, frameon=False, fontsize=11)
    figure.text(0.04, 0.855, "A  Vary alpha; num_samples = 8", fontsize=15, color="#172b3a")
    figure.text(0.04, 0.455, "B  Vary num_samples; alpha = 1.06", fontsize=15, color="#172b3a")
    comparisons = []
    for ax, alpha in zip(axes[0], [1.001, 1.06, 1.10, 1.40]):
        comparisons.append(draw_panel(ax, points, alpha, 8))
    for ax, k in zip(axes[1], [3, 6, 10, 14]):
        comparisons.append(draw_panel(ax, points, 1.06, k, show_weight=False))
    figure.text(0.5, 0.045,
                 "Center distance: mean distance to the full centroid (lower). Pair distance: mean over unordered selected pairs (higher).\n"
                 "Coverage distance: mean nearest-selected distance over all input points (lower). These are geometric diagnostics, not recommendation scores.",
                 ha="center", fontsize=11, color="#56616c", linespacing=1.6)
    for suffix in ("png", "svg"):
        destination = OUTPUT / f"select-samples-comparison.{suffix}"
        figure.savefig(destination, dpi=160, facecolor="white")
        if suffix == "svg":
            destination.write_text("\n".join(line.rstrip() for line in destination.read_text().splitlines()) + "\n", encoding="utf-8")
    plt.close(figure)
    (OUTPUT / "comparison-summary.json").write_text(json.dumps(comparisons, ensure_ascii=False, indent=2), encoding="utf-8")

    template = OUTPUT / "template.html"
    if template.exists():
        text = template.read_text(encoding="utf-8")
        assert text.count("__DEMO_DATA__") == 1
        rendered = text.replace("__DEMO_DATA__", json.dumps(demo, ensure_ascii=False, separators=(",", ":")))
        (OUTPUT / "select-samples-explorer.html").write_text(rendered, encoding="utf-8")
    print(f"Saved comparison and 51 actual-function parameter runs under {OUTPUT}")
    print(json.dumps(comparisons, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
