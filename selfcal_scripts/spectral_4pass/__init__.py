"""Spectral 4-pass chain — the SEP PAH 3.29um aromatic recipe.

Alternating least squares on the joint (sky | offset) problem, each half solved
EXACTLY, run on the whole field without tiles after pass 1:

  pass 1  joint LSQR (``pahfit_lvf_polybasis`` mode) on staged halves    -> offsets
  pass 2  sky | offsets, per-pixel CLOSED FORM (no iteration limit)      -> sky
  pass 3  offsets | sky, per-frame dense least squares over ALL frames   -> offsets
  pass 4  sky | offsets, closed form via ADDITIVE per-tile moment dumps  -> sky

See README.md in this directory for the rationale, the file map and how to run.
"""
