# SelfCal

Sparse-LSQR self-calibration and mosaicking pipeline for the SPHEREx
all-sky survey, with helpers for Euclid.

## Install

```bash
pip install -e .
```

Runtime deps are pinned in [`pyproject.toml`](pyproject.toml) and the
conda-friendly [`environment.yml`](environment.yml).

## Where to look

- [`CLAUDE.md`](CLAUDE.md) — repo layout and conventions. Read first.
- [`PIPELINE.md`](PIPELINE.md) — operational runbook (tuning knobs,
  on-disk schemas, NVMe staging pattern).
- [`SelfCal/README.md`](SelfCal/README.md) — module-level code architecture.
- [`notebooks/spherex_selfcal_demo.ipynb`](notebooks/spherex_selfcal_demo.ipynb) — working demo.
