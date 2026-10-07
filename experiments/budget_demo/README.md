# Budget allocation demo

This offline demo calls `personax.sampling.get_allocation` on four small
synthetic cluster-size scenarios. Open `budget-allocation-explorer.html` in a
browser to change the requested budget and inspect the allocation. The file is
self-contained. `budget-allocation-preview.gif` previews these changes in the
project README.

`allocation-data.json` contains the original sizes, actual sorting order, and
function outputs for every integer requested budget from 0 to the total
capacity. The step table is recorded from the running function. The data also
records the source commit and source file hash.

To regenerate the HTML from the repository root:

```bash
pip install -r requirements.txt
python -m experiments.budget_demo.generate
```

To regenerate both README animations after generating both demos' data:

```bash
pip install matplotlib pillow
python -m experiments.render_demo_previews
```

The function sorts clusters by size, allocates the smaller of each cluster's
capacity and the remaining budget's integer share, and restores the original
cluster order. This is approximately equal allocation with capacity limits,
rather than allocation proportional to cluster size. Budgets below the cluster
count are raised to that count. `sampling` obtains the requested budget as
`ceil(total_size * ratio)`; `alpha` affects within-cluster selection, not allocation.

The demo uses nonempty clusters with positive integer sizes and requested
budgets from 0 to total capacity. Budgets above total capacity are not safely
capped by the original function and can exceed cluster capacity or fail. The
demo does not change this behavior. Tied sizes follow the actual NumPy sorting
order recorded in the data; the function does not specify a stable tie order.

No models, datasets, API calls, or paper experiments are involved.
