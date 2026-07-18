"""Inference-only benchmark entrypoint.

This module exists so that ``python -m mv_phase1_bulk.benchmark`` invokes the
CLI ``benchmark`` command with the same flags.
"""

from __future__ import annotations

import sys

from mv_phase1_bulk.cli import app


def main() -> None:
    sys.argv = ["mv-phase1-bulk", "benchmark", *sys.argv[1:]]
    app()


if __name__ == "__main__":
    main()
