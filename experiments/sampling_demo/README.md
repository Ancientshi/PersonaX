# Sampling parameter demo

This offline demo calls `personax.sampling.select_samples` on 44 fixed synthetic
2D points (seed 42). It illustrates the selection rule; it does not measure
recommendation performance or reproduce the paper's experiments.

Open `select-samples-explorer.html` in a browser. It is self-contained and uses
precomputed results for 17 values of `alpha`, three coordinate scales, and
`num_samples` from 1 to 44. `select-samples-comparison.png` and `.svg` show a fixed
comparison. `sampling-data.json` records the input, selection order, internal
scores, source commit, and source file hash.

`sampling-preview.gif` is an animated preview for the project README. It uses
the same precomputed selection orders as the HTML.

To regenerate from the repository root, install the dependencies and Matplotlib,
then run:

```bash
pip install -r requirements.txt
pip install matplotlib
python -m experiments.sampling_demo.generate
```

The script does not download models or datasets.

After generating both demos' data, regenerate the README animations with:

```bash
pip install pillow
python -m experiments.render_demo_previews
```

The displayed diagnostics are mean distance to the full input centroid, mean
distance over unordered selected pairs, and mean distance from every input
point to its nearest selected point. The latter includes selected points'
zero distances. These are external geometric summaries, distinct from the
function's internal diversity term (pairwise distance sum divided by the
number of selected points).

For ordinary nonnegative weights, use finite `alpha >= 1` and integer
`1 <= num_samples <= N` with a nonempty, finite numeric array of shape `(N, d)`.
Changing the count extends the same greedy order; changing coordinate scale
can alter the balance even when `alpha` stays fixed. On this particular input,
`alpha=1.10` and `1.40` select the same first eight points. No parameter value
shown here is a general recommendation for real embeddings.
