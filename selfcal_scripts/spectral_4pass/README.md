# Spectral 4-pass chain (SEP PAH 3.29 µm)

The recipe behind the SEP D4 aromatic line map. It solves the joint
"sky + per-frame offset" problem by **alternating least squares**, with each
half solved *exactly*, and runs on the whole field without tiles after pass 1.

## Why four passes

The model per frame *k*, map pixel *p* is

```
d_k(p) = C(p) + L(p)·G(λ_k(p)) + O_k(chunk(p)) + s_k + noise
```

(`C`, `L`: per-pixel continuum and line amplitude; `G`: the measured line
template at the wavelength this pixel is seen through; `O_k`: a low-degree
polynomial in subchannel per column; `s_k`: per-frame scalar). Linear, but
badly shaped for one joint solve:

* given the offsets, the sky is **block-diagonal** — one 2×2 per pixel;
* given the sky, the offsets are **independent per frame** — 13 unknowns each;
* the joint problem has null spaces (uniform line floor ↔ static `G`-shaped
  detector pattern; uniform sky ↔ all `s_k`; the `C ↔ L` split of pixels seen
  through a narrow wavelength range).

A single joint LSQR therefore *semi-converges*: offsets converge fast, the
low-diversity pixels' `C ↔ L` split does not (the southern collapse, per-pixel
Pearson −0.9), and running longer drifts along the null spaces. So:

| pass | solves | how | tiles? |
| --- | --- | --- | --- |
| 1 | joint (offset + sky), LSQR iter 100 | `pahfit_lvf_polybasis` mode, strict per-subchannel clip | yes — the only coupled solve; staged halves |
| 2 | sky \| offsets | per-pixel **closed form** (`Calibrator.solve_sky_closed_form`) | no |
| 3 | offsets \| sky | per-frame dense `lstsq` over **all** frames, one fixed global sky | no |
| 4 | sky \| offsets | closed form via **additive moment dumps** summed across tiles | no |

Pass 1 only supplies *initial* offsets (they partially absorb sky the solver
had not built yet — e.g. LMC emission). Pass 2 is the exact sky for those
offsets. Pass 3 removes the absorbed-emission bumps (the sky now explains
them) and levels every frame against **one** sky, which is what removes tile
seams (a tiled pass 3 gives each tile its own offset↔sky gauge). Pass 4 is the
exact sky for the improved offsets. Stopping at four keeps the solution
anchored near the pass-1 min-norm gauges; the remaining zero points (uniform
line floor, continuum DC) are unobservable from the data and need external
anchors (`selfcal/line_floor.py`, `selfcal/zodi_anchor.py`).

Moment additivity: a pixel's normal equations are sums over its observations
(Σw², Σw²G, Σw²G², Σw²v, Σw²Gv), so summing per-tile dumps and solving once
*is* the full-field solve — provided each frame belongs to exactly one tile
(`make_frame_lists`).

## Files

| file | step |
| --- | --- |
| `hooks.py` | `subchannel_bc_edges`, `OffsetSubtractor`, `SkySubtractor` (POSTprocess hooks; see the weight-hook rule in the module docstring, and `SkySubtractor.window` for the edge-overhang fix) |
| `two_pass.py` | passes 1+2 for one staged tile → `<cal>_p1off.h5`, `<cal>_p2line.h5` |
| `global_pass3.py` | pass 3 over all frames against the stitched pass-2 sky → `..._GLOBALP3_offsets.h5` |
| `closed_form_pass4.py` | closed-form sky given any offsets cal; `--dump-moments` for the combine path |
| `combine_moments.py` | sum dumps, solve once → the final cal |
| `make_frame_lists.py` | disjoint per-tile frame lists |
| `run_4pass.py` | orchestrates 1→6 (subprocess per step, resumable) |
| `build_pah_template.py` | builds `data/pah_template.npz` (Drude ⊗ measured LVF) — needs the external `lvf_response` module |
| `configs/sep_d4_half_{WEST,EAST}.toml` | the SEP D4 halves used for the product |
| `../runner/modes/pahfit_lvf.py` | the `pahfit_subch` / `pahfit_lvf` / `pahfit_lvf_polybasis` modes |

Core primitives the chain relies on (in `selfcal/`): `sky_rhs_moments` in
`setup_lsqr`, `solve_sky_closed_form` / `Calibrator.solve_sky_closed_form`,
`outlier_subchannel_edges` (per-subchannel clip), `poly_basis` offsets,
`selfcal.pipeline.tiled.stitch`.

## Running

Stage each half's frames (see the NVMe staging pattern in `PIPELINE.md`), point
each config's `reproj_override` at its staged dir, then:

```bash
python -m selfcal_scripts.spectral_4pass.run_4pass --tag WE \
    --reproj-dir /data3/.../SPHEREx_SEP_2025_D4_6p2arcsec/reprojected \
    --work-dir workspace/4pass_WE \
    selfcal_scripts/spectral_4pass/configs/sep_d4_half_WEST.toml \
    selfcal_scripts/spectral_4pass/configs/sep_d4_half_EAST.toml
```

Product: `cal_<frame>_<job>_WE_p4line_SEAMFREE.h5` in the run's `calibration/`
dir. Steps whose outputs exist are skipped, so re-running the same command
resumes a failed chain. Individual steps can be run by module
(`python -m selfcal_scripts.spectral_4pass.<step> --help`).

Config knobs (`[params]`): `pass1_thresh` (2.5), `pass1_ignore_list`,
`pass2_thresh` (5.0), `pass2_subch_clip`, `pass3_thresh` (2.5),
`pass3_poly_degree` (4), `subch_poly_degree` (2, pass 1), `subch_poly_lo/hi`,
`line_template_npz`.

Wall/memory on the SEP D4 field (19,269 frames, 192-core box): pass 1 ≈ 8–14 h
per half at up to ~665 GB; pass 2 ≈ 1 h; global pass 3 ≈ 50 min at 48 workers;
moment dumps ≈ 30–40 min per half at ~75–300 GB; combine ≈ 1 min.

## Notes

* **Edge overhang.** 94 SEP frames overhang the map's bottom/left edge with
  negative `ref_coords`; a plain `map[y0:y1, x0:x1]` on them is an empty slice.
  `SkySubtractor.window` handles all four edges; anything that windows a
  ref-grid map by `ref_coords` must do the same.
* **Bright cut.** Pass 3 drops pixels whose modelled sky exceeds
  `BRIGHT_CUT` (0.05 MJy/sr) from the offset fit; frames left with fewer than
  `MIN_PIX` pixels use all of them (flagged in the run log).
* Two SEP frames (`exp_14704`, `exp_16866`) have too few valid pixels for a
  pass-3 fit and get offset 0; they contribute negligibly to pass 4.
