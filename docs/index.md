# adaptiveiv

`adaptiveiv` estimates IV models with group-level first-stage heterogeneity.

The main entry point is `adaptiveiv.AdaptiveIV`.

```python
from adaptiveiv import AdaptiveIV

model = AdaptiveIV.from_formula(
    "Y ~ 1 + X + [W ~ Z]",
    data=data,
    groups="group",
)
results = model.fit(random_state=123)
```

Inspect `results.group_diagnostics`, `results.selected_groups`, and
`results.selection_summary` before interpreting the estimate.

The default fit reports paper-baseline homoskedastic inference when the
configuration is supported. Use `cov_type="none"` for point-estimate-only
diagnostic workflows such as repeated split-sample fits.

## Validation

Run the recommended lightweight Monte Carlo validation with:

```bash
uv run --no-editable --group dev python simulations/validate_replication.py \
  --repetitions 10 \
  --n-groups 40 \
  --n-per-group 120 \
  --n-splits 3 \
  --output-dir validation/outputs/latest
```

The report is a qualitative replication of the Abadie, Gu, and Shen Section 4
simulation logic for the estimators implemented here.

Inference validation is separate:

```bash
uv run --no-editable --group dev python simulations/validate_inference.py \
  --preset smoke \
  --output-dir validation/outputs/inference
```

Direct paper-table comparison is also separate:

```bash
uv run --no-editable --group dev python simulations/validate_paper_tables.py \
  --preset smoke \
  --output-dir validation/outputs/paper_tables
```

Use `--preset full` when preparing release evidence against the paper's
reported Tables 2-4 for the implemented methods.

For the full paper-table run, use `--config-start` and `--config-stop` to run
the 30 configurations in chunks and keep each chunk's manifest auditable.
Then use `simulations/aggregate_paper_table_chunks.py` to produce the combined
release report and configuration-coverage check.
