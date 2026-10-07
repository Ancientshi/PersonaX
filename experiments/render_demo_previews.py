"""Render README animations from the demos' actual-function output tables.

Run from the repository root: python -m experiments.render_demo_previews
Requires Matplotlib and Pillow. No models, datasets, or network calls are used.
"""

from io import BytesIO
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
import numpy as np
from scipy.spatial.distance import cdist, pdist


DEMOS = Path(__file__).resolve().parent
BLUE = "#2166a6"
ORANGE = "#ae5827"
GRAY = "#d0d5db"
TEXT = "#20303e"


def frame_image(figure):
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=100, facecolor="white")
    buffer.seek(0)
    with Image.open(buffer) as image:
        frame = image.convert("RGB")
    plt.close(figure)
    return frame


def save_animation(frames, path):
    # Each frame is a directly rendered plot, rather than a browser recording.
    frames[0].save(path, save_all=True, append_images=frames[1:], format="GIF",
                   duration=1500, loop=0, disposal=2, optimize=True)


def sampling_preview():
    directory = DEMOS / "sampling_demo"
    data = json.loads((directory / "sampling-data.json").read_text())
    points = np.asarray(data["points"])
    center = points.mean(axis=0)
    frames = []
    settings = [(1.001, 8), (1.04, 8), (1.06, 8), (1.1, 8), (1.4, 8),
                (1.06, 3), (1.06, 6), (1.06, 10), (1.06, 14)]
    for alpha, count in settings:
        figure, axes = plt.subplots(1, 2, figsize=(9.6, 5.6))
        figure.subplots_adjust(left=.075, right=.98, bottom=.25, top=.76, wspace=.30)
        figure.suptitle("Sampling: center preference and diversity", y=.97,
                        fontsize=17, color=TEXT)
        figure.text(.5, .895, f"alpha = {alpha:g}   |   num_samples = {count}   |   fixed 44-point input",
                    ha="center", fontsize=12, color=TEXT)
        for axis, value, color, label in zip(axes, [1.001, alpha], [BLUE, ORANGE],
                                              ["Reference", "Current"]):
            indices = data["runs"]["1"][str(value)]["indices"][:count]
            chosen = points[indices]
            axis.scatter(*points.T, s=25, color=GRAY, zorder=1)
            axis.scatter(*chosen.T, s=57, color=color, edgecolor="white", linewidth=.6, zorder=3)
            axis.scatter(*chosen[0], s=125, marker="D", facecolors="none", edgecolors=TEXT, linewidth=1, zorder=4)
            axis.scatter(*center, s=90, marker="x", color=TEXT, linewidth=1.6, zorder=5)
            axis.set(xlim=(-1.85, 1.85), ylim=(-1.85, 1.85), aspect="equal",
                     xticks=[-1.5, 0, 1.5], yticks=[-1.5, 0, 1.5],
                     xlabel="Feature x (synthetic units)", ylabel="Feature y (synthetic units)")
            axis.set_title(f"{label}: alpha = {value:g}", fontsize=13, color=TEXT, pad=9)
            axis.tick_params(labelsize=10, length=0)
            axis.grid(color="#e9edf1", linewidth=.7, zorder=0)
            for spine in axis.spines.values():
                spine.set_color("#e0e4e8")
            coverage = cdist(points, chosen).min(axis=1).mean()
            pairwise = pdist(chosen).mean()
            axis.text(.5, -.25, f"Mean pair distance: {pairwise:.3f}\nMean coverage distance: {coverage:.3f}",
                      transform=axis.transAxes, ha="center", fontsize=11, color=TEXT)
        figure.text(.5, .04, "Gray: input points. Color: selected points. Diamond: first point. Cross: full centroid.",
                    ha="center", fontsize=11, color=TEXT)
        frames.append(frame_image(figure))
    save_animation(frames, directory / "sampling-preview.gif")


def budget_preview():
    directory = DEMOS / "budget_demo"
    data = json.loads((directory / "allocation-data.json").read_text())
    scenarios = {item["id"]: item for item in data["scenarios"]}
    largest = max(size for scenario in scenarios.values() for size in scenario["sizes"])
    axis_maximum = int(np.ceil(largest / 4) * 4)
    settings = [("capped", b) for b in [0, 4, 8, 12, 20, 28, 32]]
    settings += [("balanced", 20), ("skewed", 20), ("unsorted", 20)]
    frames = []
    for scenario_id, budget in settings:
        scenario = scenarios[scenario_id]
        run = scenario["runs"][str(budget)]
        sizes = scenario["sizes"]
        allocation = run["allocations"]
        figure, axis = plt.subplots(figsize=(9.6, 5.6))
        figure.subplots_adjust(left=.16, right=.87, bottom=.31, top=.72)
        figure.suptitle("Budget allocation: share budget within cluster capacities", y=.97,
                        fontsize=16, color=TEXT)
        figure.text(.5, .89, f"{scenario['label']}   |   cluster sizes {sizes}",
                    ha="center", fontsize=12, color=TEXT)
        figure.text(.5, .815,
                    f"Requested: {budget}     Effective: {run['effective_budget']}     Allocated: {run['allocated_total']}",
                    ha="center", fontsize=13, color=TEXT)
        positions = np.arange(len(sizes))
        axis.barh(positions, sizes, height=.54, color=GRAY)
        axis.barh(positions, allocation, height=.54, color=BLUE)
        axis.set(yticks=positions, yticklabels=[f"Cluster {i+1}" for i in positions],
                 xlim=(0, axis_maximum), xticks=np.arange(0, axis_maximum+1, 4),
                 xlabel="Items per cluster (input order)")
        axis.invert_yaxis()
        axis.tick_params(labelsize=12, length=0, pad=7)
        axis.grid(axis="x", color="#e9edf1", linewidth=.7)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_visible(False)
        for index, (size, selected) in enumerate(zip(sizes, allocation)):
            suffix = " · full" if selected == size else ""
            axis.text(size+.35, index, f"{selected} / {size}{suffix}", va="center", fontsize=12, color=TEXT)
        legend = [Line2D([], [], color=GRAY, linewidth=8, label="Cluster capacity"),
                  Line2D([], [], color=BLUE, linewidth=8, label="Allocated items")]
        figure.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .135),
                      ncol=2, frameon=False, fontsize=11)
        note = ("Requested budget is raised to 4 so each nonempty cluster receives an item."
                if budget < scenario["cluster_count"] else
                "Remaining budget is shared, while each cluster is capped by its size.")
        figure.text(.5, .075, note, ha="center", fontsize=11, color=TEXT)
        figure.text(.5, .025, "Actual get_allocation output · synthetic cluster sizes · not proportional allocation",
                    ha="center", fontsize=11, color=TEXT)
        frames.append(frame_image(figure))
    save_animation(frames, directory / "budget-allocation-preview.gif")


def main():
    sampling_preview()
    budget_preview()
    print("Saved both README preview animations from precomputed function outputs.")


if __name__ == "__main__":
    main()
