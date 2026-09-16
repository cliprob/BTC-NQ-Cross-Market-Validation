from __future__ import annotations

from pathlib import Path

import nbformat


def test_canonical_notebook_parses_and_code_compiles() -> None:
    path = Path("notebooks/research_summary.ipynb")
    notebook = nbformat.read(path, as_version=4)
    assert notebook.cells
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, str(path), "exec")
