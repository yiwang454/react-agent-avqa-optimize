"""Compatibility module for the modularized DSPy AVQA implementation."""

from __future__ import annotations

import time

from dspy_avqa import *  # noqa: F401,F403


if __name__ == "__main__":
    started = time.perf_counter()
    run_batch()
    elapsed = time.perf_counter() - started
    print(f"Total elapsed time: {elapsed:.2f}s")
