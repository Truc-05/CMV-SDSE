"""Single source of truth for raw-run directory layout.

Every raw run is isolated by its context condition so an evidence-mode run can never
overwrite the truncate-mode data (or vice versa). The context mode is encoded in the
LEAF directory name:

    data/raw_cascade_runs/mast__3_3__ctx_truncate/
    data/raw_cascade_runs/mast__3_3__ctx_evidence/

Both the writer (a real run) and the reader (--analyse-only, freeze, the figure/audit
scripts) resolve their directory through raw_run_dir(), so there is no manual move or
overwrite step after a run.
"""
from __future__ import annotations

import os

RAW_ROOT = os.path.join("data", "raw_cascade_runs")


def raw_run_dir(mast_target, context_mode: str = "truncate", source: str = "mast") -> str:
    slug = f"{source}__{str(mast_target).replace('.', '_')}__ctx_{context_mode}"
    return os.path.join(RAW_ROOT, slug)


TABLES_DIR = os.path.join("results", "tables")


def table_path(name: str, context_mode: str = "truncate", tables_dir: str = TABLES_DIR) -> str:
    """Analysis-output path, namespaced by context condition.

    truncate = the canonical, un-suffixed name (what the paper/figures read); any other
    context mode gets a `__ctx_<mode>` suffix so a non-truncate run's analyse can NEVER
    overwrite the canonical truncate tables.
    """
    if context_mode and context_mode != "truncate":
        stem, ext = os.path.splitext(name)
        name = f"{stem}__ctx_{context_mode}{ext}"
    return os.path.join(tables_dir, name)
