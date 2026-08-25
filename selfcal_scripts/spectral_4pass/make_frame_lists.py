"""Disjoint per-tile frame lists for the moment-combination path.

Staged tile directories overlap slightly (frames near a boundary are staged in
both). Summing moments would count those frames twice, so assign every frame
to exactly one tile: first-listed tile wins.

    python -m selfcal_scripts.spectral_4pass.make_frame_lists <out_dir> \\
        NAME=<staged_dir> [NAME=<staged_dir> ...]

Writes ``<out_dir>/<NAME>.txt`` (one basename per line) and prints the counts.
"""
import glob
import os
import sys


def main(out_dir, named_dirs):
    os.makedirs(out_dir, exist_ok=True)
    seen = set()
    total = 0
    paths = {}
    for name, d in named_dirs:
        names = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(d, "*.h5")))]
        keep = [n for n in names if n not in seen]
        seen.update(keep)
        total += len(keep)
        out = os.path.join(out_dir, f"{name}.txt")
        with open(out, "w") as f:
            f.write("\n".join(keep) + "\n")
        paths[name] = out
        print(f"[framelists] {name}: staged {len(names)} -> assigned {len(keep)}", flush=True)
    print(f"[framelists] total unique frames {total}", flush=True)
    return paths


if __name__ == "__main__":
    out_dir = sys.argv[1]
    named = [tuple(a.split("=", 1)) for a in sys.argv[2:]]
    main(out_dir, named)
