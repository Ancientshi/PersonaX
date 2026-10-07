"""Generate small offline observations of PersonaX budget allocation.

Run from the repository root: python -m experiments.budget_demo.generate
Allocations and step values come from the real get_allocation execution.
"""

import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from personax.sampling import get_allocation


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
SCENARIOS = [
    ("balanced", "Balanced clusters", "Equal capacities show how integer remainders are allocated.", [8, 8, 8, 8]),
    ("capped", "Small-cluster caps", "Once small clusters fill, the remaining budget goes to other clusters.", [2, 4, 10, 16]),
    ("skewed", "Skewed clusters", "Unequal capacities show the minimum allocation and small-cluster limits.", [1, 3, 6, 22]),
    ("unsorted", "Reordered clusters", "The same capacities as Small-cluster caps, reordered to show sorting and restoration.", [16, 2, 10, 4]),
]


def observe_allocation(sizes, budget):
    """Read locals before and after the real budget-decrement statement."""
    lines, first_line = inspect.getsourcelines(get_allocation)
    decrement_lines = [number for number, line in enumerate(lines, first_line)
                       if line.strip() == "budget -= allocation_list[-1]"]
    assert len(decrement_lines) == 1, "Allocation source changed; update the trace anchor."
    decrement_line = decrement_lines[0]
    steps = []
    pending = None

    def trace(frame, event, _arg):
        nonlocal pending
        if frame.f_code is not get_allocation.__code__:
            return None
        values = frame.f_locals
        if pending is not None and event in ("line", "return"):
            pending["remaining_after"] = int(values["budget"])
            pending = None
        if event == "line" and frame.f_lineno == decrement_line:
            pending = {
                "original_index": int(values["cluster_size_list_index"][values["i"]]),
                "size": int(values["cluster_size"]),
                "remaining_before": int(values["budget"]),
                "remaining_clusters": int(values["unallocated_strata"]),
                "floor_share": int(values["avg"]),
                "allocated": int(values["allocation_list"][-1]),
            }
            steps.append(pending)
        return trace

    previous_trace = sys.gettrace()
    try:
        sys.settrace(trace)
        allocations = [int(value) for value in get_allocation(sizes, budget)]
    finally:
        sys.settrace(previous_trace)
    assert pending is None
    return allocations, steps


def main():
    OUTPUT.mkdir(exist_ok=True)
    source = ROOT / "personax" / "sampling.py"
    data = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "scenarios": [],
    }
    checked = 0
    for identifier, label, description, sizes in SCENARIOS:
        total_size, cluster_count = sum(sizes), len(sizes)
        sort_order = [int(index) for index in np.argsort(sizes)]
        scenario = {
            "id": identifier, "label": label, "description": description,
            "sizes": sizes, "total_size": total_size, "cluster_count": cluster_count,
            "sort_order": sort_order, "runs": {},
        }
        for budget in range(total_size + 1):
            allocations, steps = observe_allocation(sizes, budget)
            direct = [int(value) for value in get_allocation(sizes, budget)]
            effective_budget = steps[0]["remaining_before"]
            allocated_total = sum(allocations)
            restored = [None] * cluster_count
            for step in steps:
                restored[step["original_index"]] = step["allocated"]
            assert allocations == direct == restored
            assert len(steps) == cluster_count
            assert [step["original_index"] for step in steps] == sort_order
            assert allocated_total == effective_budget == max(budget, cluster_count)
            assert all(1 <= amount <= size for amount, size in zip(allocations, sizes))
            assert steps[-1]["remaining_after"] == 0
            assert all(left["remaining_after"] == right["remaining_before"]
                       for left, right in zip(steps, steps[1:]))
            scenario["runs"][str(budget)] = {
                "requested_budget": budget, "effective_budget": effective_budget,
                "allocations": allocations, "allocated_total": allocated_total,
                "unallocated_budget": effective_budget - allocated_total, "steps": steps,
            }
            checked += 1
        data["scenarios"].append(scenario)
    assert checked == 132
    serialized = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    (OUTPUT / "allocation-data.json").write_text(serialized, encoding="utf-8")
    template = OUTPUT / "template.html"
    if template.exists():
        text = template.read_text(encoding="utf-8")
        assert text.count("__DEMO_DATA__") == 1
        (OUTPUT / "budget-allocation-explorer.html").write_text(
            text.replace("__DEMO_DATA__", serialized), encoding="utf-8")
    print(f"Saved and verified {checked} allocation runs under {OUTPUT}")


if __name__ == "__main__":
    main()
