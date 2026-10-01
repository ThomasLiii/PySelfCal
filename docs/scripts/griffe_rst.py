"""Griffe extension: render the docstrings' reStructuredText markup as Markdown.

The code's docstrings use Sphinx roles (``:class:`~selfcal.models.spec.ModelSpec```,
``:func:`setup_lsqr```), ``::`` literal-block markers and, in two places,
setext-underlined titles. mkdocstrings renders Markdown, so once a package is
loaded every docstring is rewritten:

- a role becomes a cross-reference to the object's absolute path, the name
  resolved in the docstring's scope as Python resolves it (the object, its
  parents, their imports), then among the project's public objects by unique
  short name; a target that is private, outside the project or unresolved
  stays plain code (unresolved ones are logged);
- ``text::`` becomes ``text:`` and the indented literal block after it a
  fenced code block;
- an underlined title that is not a NumPy section becomes bold text.

The ``RST_STATS`` counters record what was converted, for the build log.
"""
from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict

import griffe

log = logging.getLogger("mkdocs.plugins.griffe_rst")

PROJECT = ("selfcal", "selfcal_scripts")
ROLE = re.compile(r":(class|func|meth|mod|attr|data|obj|exc|const):`(~?)([^`]+)`")
UNDERLINE = re.compile(r"^\s*([=\-~^\"#*+])\1{2,}\s*$")
NUMPY_SECTIONS = {
    "Parameters", "Other Parameters", "Returns", "Yields", "Receives", "Raises", "Warns",
    "Warnings", "Attributes", "Methods", "Notes", "Examples", "See Also", "References",
}
RST_STATS: Counter = Counter()
# short name -> public paths, over every project package loaded in this build (shared because
# mkdocstrings may load each package through its own extension instance)
SHORT_NAMES: dict[str, set[str]] = defaultdict(set)


def _is_private(path: str) -> bool:
    return any(p.startswith("_") and not (p.startswith("__") and p.endswith("__")) for p in path.split("."))


def _walk(obj):
    yield obj
    for member in list(obj.members.values()):
        if member.is_alias:
            continue
        yield from _walk(member)


class RstMarkup(griffe.Extension):
    """Convert reStructuredText docstring markup to Markdown (see the module docstring)."""

    # -- resolution -------------------------------------------------------------------------------
    def _lookup(self, loader, path: str):
        top = path.split(".", 1)[0]
        if top not in loader.modules_collection.members:
            return None
        try:
            return loader.modules_collection[path]
        except Exception:  # KeyError, alias resolution errors
            return None

    def _canonical(self, loader, path: str) -> str | None:
        """The canonical path of ``path``, or None when it is known not to exist."""
        top = path.split(".", 1)[0]
        if top not in loader.modules_collection.members:
            return path  # a package not loaded in this build: trust the written path
        obj = self._lookup(loader, path)
        if obj is None:
            # a member of an alias target (``Alias.attr``): resolve the head, then append
            head, _, tail = path.rpartition(".")
            parent = self._lookup(loader, head) if head else None
            if parent is not None and tail:
                try:
                    return f"{parent.canonical_path}.{tail}" if tail in parent.members else None
                except Exception:
                    return None
            return None
        try:
            return obj.canonical_path
        except Exception:
            return path

    def _resolve(self, target: str, obj, loader) -> str | None:
        if target.startswith("."):  # a Python-relative module reference
            mod = obj.module
            base = mod if mod.is_package else mod.parent
            dots = len(target) - len(target.lstrip("."))
            for _ in range(dots - 1):
                base = base.parent if base is not None else None
            if base is None:
                return None
            rest = target.lstrip(".")
            return self._canonical(loader, f"{base.path}.{rest}" if rest else base.path)
        first, _, rest = target.partition(".")
        if first in PROJECT:
            return self._canonical(loader, target)
        scope = obj
        while scope is not None:
            member = scope.members.get(first) if hasattr(scope, "members") else None
            if member is not None:
                base = member.target_path if member.is_alias else member.path
                return self._canonical(loader, f"{base}.{rest}" if rest else base)
            scope = scope.parent
        candidates = set(SHORT_NAMES.get(first, set()))
        if not candidates:
            for top in PROJECT:
                pkg = loader.modules_collection.members.get(top)
                if pkg is not None:
                    candidates |= {o.path for o in _walk(pkg) if o.name == first and not _is_private(o.path)}
        if len(candidates) == 1:
            (base,) = candidates
            return self._canonical(loader, f"{base}.{rest}" if rest else base)
        return None

    # -- rewriting --------------------------------------------------------------------------------
    def _roles(self, text: str, obj, loader) -> str:
        def sub(m: re.Match) -> str:
            kind, tilde, target = m.group(1), m.group(2), m.group(3).strip()
            shown = target.rsplit(".", 1)[-1] if tilde else target.lstrip(".")
            if kind in ("func", "meth"):
                shown += "()"
            path = self._resolve(target, obj, loader)
            if path is None or _is_private(path) or path.split(".", 1)[0] not in PROJECT:
                RST_STATS["role_plain"] += 1
                if path is None and not _is_private(target):
                    RST_STATS["role_unresolved"] += 1
                    log.info("griffe_rst: %s: unresolved %s:`%s`", obj.path, kind, target)
                return f"`{shown}`"
            RST_STATS["role_link"] += 1
            return f"[`{shown}`][{path}]"

        out, fenced = [], False
        for line in text.split("\n"):
            if line.lstrip().startswith("```"):
                fenced = not fenced
            out.append(line if fenced else ROLE.sub(sub, line))
        return "\n".join(out)

    @staticmethod
    def _blocks(text: str) -> str:
        lines = text.split("\n")
        out: list[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.rstrip()
            # an underlined title that is not a NumPy section -> bold text
            if (i + 1 < len(lines) and stripped.strip() and UNDERLINE.match(lines[i + 1])
                    and stripped.strip() not in NUMPY_SECTIONS
                    and len(lines[i + 1].strip()) >= len(stripped.strip()) - 1):
                indent = line[: len(line) - len(line.lstrip())]
                out.append(f"{indent}**{stripped.strip()}**")
                RST_STATS["setext_title"] += 1
                i += 2
                continue
            if stripped.endswith("::"):
                head = stripped[:-2].rstrip()
                indent = len(line) - len(line.lstrip())
                if head.strip():
                    out.append(head + ":")
                # the literal block: the indented lines after the marker (blank lines inside allowed)
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                block = []
                k = j
                while k < len(lines) and (not lines[k].strip()
                                          or len(lines[k]) - len(lines[k].lstrip()) > indent):
                    block.append(lines[k])
                    k += 1
                while block and not block[-1].strip():
                    block.pop()
                    k -= 1
                if block:
                    cut = min(len(b) - len(b.lstrip()) for b in block if b.strip())
                    pad = " " * indent
                    out.append("")
                    out.append(pad + "```text")
                    out.extend((pad + b[cut:]) if b.strip() else "" for b in block)
                    out.append(pad + "```")
                    RST_STATS["literal_block"] += 1
                    i = k
                    continue
                i += 1
                continue
            out.append(line)
            i += 1
        return "\n".join(out)

    @staticmethod
    def _summary_line(text: str) -> str:
        """Put the first sentence of the first paragraph on the first line.

        mkdocstrings' summary tables show a docstring's first line. In Markdown a
        paragraph's line breaks are spaces, so re-breaking the paragraph after its
        first sentence leaves the rendered docstring unchanged."""
        lines = text.split("\n")
        n = 0
        while n < len(lines) and lines[n].strip():
            n += 1
        para = lines[:n]
        if not para or any(re.match(r"^\s*([-*+|>#]|\d+[.)]|```)", ln) for ln in para):
            return text
        joined = " ".join(ln.strip() for ln in para)
        masked = re.sub(r"(`+)(.+?)\1", lambda m: "x" * len(m.group(0)), joined)
        masked = re.sub(r"\[[^\]]*\]\[[^\]]*\]", lambda m: "x" * len(m.group(0)), masked)
        m = re.search(r"(?<!\be\.g)(?<!\bi\.e)(?<!\betc)(?<!\bvs)(?<!\bcf)\.(?=\s+[A-Z(`\[*])", masked)
        if m is None:
            first, rest = joined, ""
        else:
            first, rest = joined[: m.end()], joined[m.end():].strip()
        new_para = [first] + ([rest] if rest else [])
        if new_para == [ln.strip() for ln in para] and para == [ln.strip() for ln in para]:
            return text
        RST_STATS["summary_line"] += 1
        return "\n".join(new_para + lines[n:])

    # -- hook -------------------------------------------------------------------------------------
    def on_package(self, *, pkg, loader, **kwargs) -> None:
        if pkg.name not in PROJECT:
            return
        objects = list(_walk(pkg))
        for o in objects:
            if not _is_private(o.path):
                SHORT_NAMES[o.name].add(o.path)
        for o in objects:
            doc = o.docstring
            if doc is None or not doc.value:
                continue
            value = self._blocks(doc.value)
            value = self._roles(value, o, loader)
            value = self._summary_line(value)
            if value != doc.value:
                doc.value = value
