"""Generate the site's virtual pages at build time (run by mkdocs-gen-files).

1. The API reference: one page per public module of ``selfcal`` and of the run
   engine ``selfcal_scripts.runner``, and ``reference/SUMMARY.md`` for the
   navigation (mkdocs-literate-nav).
2. The guides that live next to the code (``PIPELINE.md``, the package and
   config READMEs, ...): each is imported at a site path with its relative links
   rewritten -- to the imported page, to a module's API page, or to the file on
   GitHub -- so every guide keeps one source, readable both on GitHub and here.
   A link to a file that does not exist is logged as a warning, which fails
   ``mkdocs build --strict``.
"""
from __future__ import annotations

import logging
import posixpath
import re
from pathlib import Path

import mkdocs_gen_files

log = logging.getLogger("mkdocs.plugins.gen_pages")

ROOT = Path(__file__).resolve().parents[2]
REPO_URL = "https://github.com/ThomasLiii/PySelfCal"
BRANCH = "main"
API_ROOTS = ["selfcal", "selfcal_scripts/runner"]

# repository path -> site path of the guides imported from outside docs/
GUIDES = {
    "selfcal_scripts/configs/README.md": "guide/configuration.md",
    "PIPELINE.md": "guide/pipeline.md",
    "selfcal_scripts/zodi_anchor/README.md": "tools/zodi-anchor.md",
    "selfcal_scripts/transfer_function/README.md": "tools/transfer-function.md",
    "selfcal_scripts/transfer_function/DETAILS.md": "tools/transfer-function-details.md",
    "selfcal/README.md": "developer/architecture.md",
    "selfcal_scripts/gates/README.md": "developer/gates.md",
}
# pages that already live in docs/: repository path -> site path
NATIVE = {
    f"docs/{p.relative_to(ROOT / 'docs').as_posix()}": p.relative_to(ROOT / "docs").as_posix()
    for p in (ROOT / "docs").rglob("*.md")
}


def _module_pages() -> dict[str, tuple[str, str]]:
    """repository path of each documented module -> (dotted name, site path of its page)."""
    pages = {}
    for root in API_ROOTS:
        for path in sorted((ROOT / root).rglob("*.py")):
            rel = path.relative_to(ROOT)
            parts = list(rel.with_suffix("").parts)
            if any(p.startswith("_") and p != "__init__" for p in parts) or "__pycache__" in parts:
                continue
            if parts[-1] == "__init__":
                parts = parts[:-1]
                doc = Path("reference", *parts, "index.md")
            else:
                doc = Path("reference", *parts).with_suffix(".md")
            pages[rel.as_posix()] = (".".join(parts), doc.as_posix())
    return pages


MODULES = _module_pages()


REFERENCE_INDEX = """# API reference

The reference is generated from the docstrings of two packages:

- [`selfcal`](selfcal/index.md): the library. The model
  ([`selfcal.models`](selfcal/models/index.md)), the solver
  ([`selfcal.core`](selfcal/core/index.md)), instruments
  ([`selfcal.instruments`](selfcal/instruments/index.md)), file I/O
  ([`selfcal.io`](selfcal/io/index.md)) and the calibration and mosaic passes
  ([`selfcal.pipeline`](selfcal/pipeline/index.md)).
- [`selfcal_scripts.runner`](selfcal_scripts/runner/index.md): the run engine
  behind `selfcal_scripts/run.sh`. It covers the TOML run config, tasks, modes
  and N-pass scheduling.

Each page documents one module. A page opens with the module docstring, then
summary tables of its classes and functions, then every public object with its
signature and source.
"""


def _write_reference() -> None:
    nav = mkdocs_gen_files.Nav()
    nav[("Overview",)] = "index.md"
    with mkdocs_gen_files.open("reference/index.md", "w") as fd:
        fd.write(REFERENCE_INDEX)
    for rel, (dotted, doc) in sorted(MODULES.items(), key=lambda kv: kv[1][0]):
        parts = tuple(dotted.split("."))
        nav[parts] = Path(doc).relative_to("reference").as_posix()
        with mkdocs_gen_files.open(doc, "w") as fd:
            fd.write(f"::: {dotted}\n")
        mkdocs_gen_files.set_edit_path(doc, Path("..", rel).as_posix())
    with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as fd:
        fd.writelines(nav.build_literate_nav())


LINK = re.compile(r"(!?\[[^\]\n]*\])\(([^)\s]+)((?:\s+\"[^\"]*\")?)\)")


def _rewrite_target(target: str, src_repo: str, page_site: str) -> str:
    if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
        return target
    path, hash_, anchor = target.partition("#")
    repo_path = posixpath.normpath(posixpath.join(posixpath.dirname(src_repo), path))
    page_dir = posixpath.dirname(page_site)
    site = GUIDES.get(repo_path) or NATIVE.get(repo_path)
    if site is None and repo_path in MODULES:
        site = MODULES[repo_path][1]
    if site is not None:
        return posixpath.relpath(site, page_dir or ".") + (hash_ + anchor if anchor else "")
    full = ROOT / repo_path
    if not full.exists():
        log.warning("gen_pages: %s links to %s, which does not exist", src_repo, target)
    kind = "tree" if full.is_dir() else "blob"
    return f"{REPO_URL}/{kind}/{BRANCH}/{repo_path}" + (hash_ + anchor if anchor else "")


def _rewrite_links(text: str, src_repo: str, page_site: str) -> str:
    out, fenced = [], False
    for line in text.split("\n"):
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
        if not fenced:
            line = LINK.sub(lambda m: f"{m.group(1)}({_rewrite_target(m.group(2), src_repo, page_site)}{m.group(3)})", line)
        out.append(line)
    return "\n".join(out)


def _write_guides() -> None:
    for src, site in GUIDES.items():
        text = (ROOT / src).read_text()
        with mkdocs_gen_files.open(site, "w") as fd:
            fd.write(_rewrite_links(text, src, site))
        mkdocs_gen_files.set_edit_path(site, Path("..", src).as_posix())


_write_reference()
_write_guides()
