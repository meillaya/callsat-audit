"""Small command-line entry point for reproducible data and summary steps."""

from __future__ import annotations

import argparse

from .generate import build_datasets, load_datasets, summary_table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="callsat")
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data", help="generate or refresh the synthetic datasets")
    data.add_argument("--force", action="store_true", help="regenerate even if files exist")

    sub.add_parser("summary", help="print dataset sizes and label means")

    args = parser.parse_args(argv)
    if args.command == "data":
        datasets = build_datasets(force=args.force)
        print(summary_table(datasets))
    elif args.command == "summary":
        print(summary_table(load_datasets()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
