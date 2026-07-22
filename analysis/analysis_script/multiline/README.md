# Multi-line spectral fit — NEP D4 (aromatic + aliphatic + plateau)

Production multi-line self-cal: continuum + three 3.3um PAH-complex emission
blocks fit per pixel over the SPHEREx NEP W17 Detector-4 field. Promoted from the
`workspace/spectral-pah-fit` campaign to a tracked production 2026-07.

## What it is

| block | lam0 (um) | intrinsic FWHM | convolved FWHM |
|---|---|---|---|
| aromatic | 3.289 | 0.0423 | 105 nm |
| aliphatic | 3.400 | 0.0299 | 100 nm |
| plateau | 3.470 | 0.1500 | 186 nm |

Realistic templates = Drude intrinsic profile ⊗ measured SPHEREx Band-4 spectral
response (peak-normalized, in `selfcal/instruments/spherex/data/line_templates/`;
rebuild with `selfcal_scripts/drivers/build_line_template.py`). Recipe: hard
poly-basis offset (deg 2, NumCol 3, no ortho, no weight knob — the DC lives in the
per-frame scalar), `damp_weight_line=5e-3`, iter300.

## Run it

```bash
./selfcal_scripts/run.sh selfcal_scripts/configs/multiline_nep.toml
```

`task="tiled"` builds the full field as 16 **adaptive-overlap** tiles (Fisher-
stitched). For a compact probe region set `task="cal"` + `reproj_override` +
`n_frames` — the `multiline` mode is identical either way. The mode is
`selfcal_scripts/runner/modes/multiline.py`; the SPHEREx chunk encoding lives in
`SpherexAdapter.subchannel_poly_basis`, keeping the offset-basis core agnostic.

## Why adaptive-overlap tiling

Hard center-assignment truncates per-pixel wavelength diversity at tile seams
(the frame footprint, 3156 px, is larger than most tiles) -> the read-time I_P
mask blanks stripes along every seam. The overlap tiles grow ~1578 px (a
footprint half-extent) into the SPARSE outskirts only (cheap + where diversity is
missing); the dense hub stays hard-partitioned (diversity already high, and a
full-overlap disk there exceeds the memory budget). Result: aliphatic I_P<25
masked fraction 16.6% -> 12.9%, interior seam stripes gone (the residual is the
genuine shallow rim). Layout: `design_overlap_tiles.py` (reproduces the config's
inline `[tiled].tiles`); `prod_tiles_overlap.npz` is the shipped layout.

## Analysis scripts (this dir)

- `probe_multiline.py label=CAL ...` — per-block stats, all pairwise Pearson,
  per-line I_P (the per-cal health probe).
- `render_components.py --cal CAL --tag NAME [--ip TAU]` — 2x2 component maps +
  correlation table.
- `render_flux_ratio.py --cal CAL --tag NAME [--ip][--arom-floor]` — integrated
  flux (non-aromatic sum, aromatic) + ratio, linear + log.
- `render_overlaps.py --cal CAL --tag NAME --tiles NPZ` — tile-edge overlay,
  n_contrib map, seam/coverage contours (the seam-diagnostic figures).
- `stitch_partial.py` — Fisher-stitch whatever tiles exist so far (safe mid-run).
- `design_overlap_tiles.py --reproj-dir DIR` — regenerate the tile layout.
- `render_probe_grid.py` / `render_combined_flux.py` / `make_science_figs.py` —
  the iteration-sweep stability study that selected iter300 (probe-era).

## Key results (2026-07)

Adding the aliphatic/plateau blocks leaves the aromatic 3.29 map robust: it
correlates r=0.82 with the old single-line PAH fit (same cirrus morphology). The
aliphatic<->plateau pair is mutually degenerate (Gram 0.69, profiles overlap) —
report their combined integrated flux, and mask each line on its own I_P. The
tracked `multiline` mode is byte-equal to the workspace prototype (verified on a
40-frame cal). Full campaign log: `workspace/multiline-nep/brief.md`.
