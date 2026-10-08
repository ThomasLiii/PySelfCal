# Documentation

This site is built with [MkDocs](https://www.mkdocs.org/) and the
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme. The API reference is
generated from the docstrings by [mkdocstrings](https://mkdocstrings.github.io/), which reads the
code statically: nothing is imported or run.

## Building

```bash
pip install -e ".[docs]"
mkdocs serve                                          # live preview at http://127.0.0.1:8000
DISABLE_MKDOCS_2_WARNING=true mkdocs build --strict   # the check CI runs
```

`--strict` turns every warning into an error: a broken link, a missing anchor, a cross-reference
to an object that does not exist, a page missing from the navigation. The environment variable
only silences a notice that Material for MkDocs prints about MkDocs 2.

The `[docs]` extra in `pyproject.toml` pins `mkdocs<2`. MkDocs 2 removes the plugin system that the
site depends on (gen-files, literate-nav, section-index, autorefs and mkdocstrings are all
plugins).

## How the site is put together

| Part | Where | What it does |
| --- | --- | --- |
| Configuration | `mkdocs.yml` | theme, navigation, plugins, Markdown extensions, mkdocstrings options, link validation |
| Native pages | `docs/*.md` | pages written for the site: home, getting started, concepts, the Python API, the TOML migration, glossary, developer pages, and the guide to new instruments |
| Imported guides | `docs/scripts/gen_pages.py` | the guides that live next to the code, imported at build time |
| API reference | `docs/scripts/gen_pages.py` | one page per module, generated at build time |
| Docstring markup | `docs/scripts/griffe_rst.py` | renders the docstrings' reStructuredText markup as Markdown |

`docs/scripts/` is excluded from the site itself.

### Imported guides

Some guides belong next to the code they describe and are read on GitHub too: `PIPELINE.md`,
`selfcal/README.md`, and the READMEs of the gates, the zodiacal-light anchor and the
transfer-function kit. They keep one source. `gen_pages.py` (run by mkdocs-gen-files) copies each
into the site at the path its `GUIDES` map gives, for example `selfcal_scripts/gates/README.md` to
`developer/gates.md`. It rewrites each relative link:

- to another imported guide, or to a page in `docs/`: the link to that page on the site;
- to a module that has an API page (`core/system.py`): that page;
- to any other file or directory of the repository: its URL on GitHub (`blob/main/...` or
  `tree/main/...`);
- to a path that does not exist: a warning, which fails the strict build.

The "edit" button of an imported page opens its source file. Write links in these guides as
ordinary relative repository links, so they work on GitHub and on the site.

### The API reference

`gen_pages.py` writes one page per public module of `selfcal` (the run engine `selfcal.run` included).
A package (`__init__.py`) becomes `reference/<package path>/index.md`, a module becomes
`reference/<module path>.md`, and modules whose name starts with an underscore are skipped. It
also writes `reference/SUMMARY.md`, the navigation that mkdocs-literate-nav reads, and the overview
page `reference/index.md`. A new public module therefore gets its page without any change here.

The mkdocstrings options in `mkdocs.yml`:

- read NumPy-style docstring sections;
- show public objects only, hiding `__repr__`, `__post_init__` and similar methods;
- show objects that have no docstring;
- merge `__init__` into its class;
- open each module page with summary tables of its classes and functions;
- show the source of every object.

### Docstring markup

The docstrings are written with reStructuredText conventions, while mkdocstrings renders Markdown.
`griffe_rst.py` is a [Griffe](https://mkdocstrings.github.io/griffe/) extension that rewrites
every docstring of the project once its package is loaded:

- **Sphinx roles become cross-references.** In `` :class:`~selfcal.io.calfile.CalFile` ``,
  `` :func:`setup_lsqr` `` and `` :mod:`.engine` ``, the name is resolved the way Python resolves names: an
  absolute project path is used as is; a leading dot is relative to the module's package; a bare
  name is looked up in the docstring's own object, then in its parents and their imports, then
  among the project's public objects when exactly one has that name. `~` shows only the last
  component, and functions and methods get `()`. A target that is private, outside the project or
  not found stays plain code. Unresolved names are logged at INFO level.
- **`text::` literal blocks become fenced code blocks.**
- **An underlined title that is not a NumPy section becomes bold text,** so that it does not add
  a heading to the page.
- **The first sentence of the first paragraph is put on a line of its own.** The summary tables
  show a docstring's first line, and a paragraph's line breaks do not change its rendering.

## Adding to the site

- **A page:** write `docs/<section>/<name>.md` and add it to `nav` in `mkdocs.yml`. The strict
  build fails on a page missing from the navigation.
- **A guide that lives next to the code:** add it to `GUIDES` in `docs/scripts/gen_pages.py` and to
  `nav` in `mkdocs.yml`.
- **A module:** nothing to do; give it a module docstring.

## Writing docstrings

- Start with one complete sentence that summarises the object. The summary tables show it alone.
- Refer to other objects with roles (`` :class:`~selfcal.models.spec.ModelSpec` ``,
  `` :func:`setup_lsqr` ``, `` :meth:`total_offsets` ``); on the site they become links. Use
  ``` ``double backticks`` ``` for literal code.
- Use NumPy sections (`Parameters`, `Returns`, ... with dashed underlines) where the parameters need
  explaining; prose is fine otherwise.
- Introduce a literal block with `::` and indent it.
- Do not indent a prose paragraph by four spaces or more: Markdown renders it as a code block.
- Run the strict build. It reports a role that points at an object that does not exist, and a
  malformed NumPy section.

## Writing pages

- Link other pages with relative paths to their `.md` files: `../guide/concepts.md`, or
  `../guide/concepts.md#the-model` for a section.
- Link an object of the API with an autorefs cross-reference:
  `` [`CalFile`][selfcal.io.calfile.CalFile] `` (an absolute dotted path; it works on any page).
- Link repository files that are not pages (scripts, notebooks, tests) with their GitHub
  URL, `https://github.com/ThomasLiii/PySelfCal/blob/main/<path>`.
- Include example files instead of copying them. The `pymdownx.snippets` extension is rooted at
  the repository, and a fenced block containing `--8<-- "examples/quickstart/simulate.py"`
  includes that file. A missing file fails the build. The quickstart's example files are run by
  `tests/test_quickstart_example.py`, so the code on the page is tested.
- Admonitions (`!!! note`), collapsible blocks (`??? note`), content tabs and tables are available.
  There is no math extension: write equations in code blocks.

## Hosting

The site is published on GitHub Pages, at <https://thomasliii.github.io/PySelfCal/>.
[`.github/workflows/docs.yml`](https://github.com/ThomasLiii/PySelfCal/blob/main/.github/workflows/docs.yml)
builds it with `mkdocs build --strict` on every push to `main` and deploys the result; it can also
be started by hand from the repository's Actions tab. The `docs` job of the CI workflow builds it
on every pull request, so a change that breaks the site is caught before it is merged.
