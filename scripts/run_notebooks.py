"""Execute all notebooks headlessly with a workspace-local Jupyter runtime."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configure_env() -> None:
    os.environ.setdefault("UV_CACHE_DIR", str(ROOT / ".uv-cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("JUPYTER_DATA_DIR", str(ROOT / ".jupyter-data"))
    os.environ.setdefault("JUPYTER_CONFIG_DIR", str(ROOT / ".jupyter-config"))
    os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(ROOT / ".jupyter-runtime"))
    os.environ.setdefault("JUPYTER_PATH", str(ROOT / ".venv" / "share" / "jupyter"))
    for name in ("JUPYTER_DATA_DIR", "JUPYTER_CONFIG_DIR", "JUPYTER_RUNTIME_DIR"):
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None, help="comma-separated notebook stems")
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    configure_env()

    import nbformat
    from nbclient import NotebookClient

    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        notebooks = [p for p in notebooks if p.stem in wanted]
    if not notebooks:
        print("no notebooks found", file=sys.stderr)
        return 1

    failures = 0
    for path in notebooks:
        print(f"== executing {path.name}", flush=True)
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=args.timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
            allow_errors=False,
        )
        try:
            client.execute()
        except Exception as exc:
            failures += 1
            print(f"FAILED {path.name}: {exc}", file=sys.stderr)
            continue
        nbformat.write(notebook, path)
        print(f"OK {path.name}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
