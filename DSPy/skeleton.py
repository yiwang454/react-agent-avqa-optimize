"""Compatibility entrypoint for DSPy AVQA implementation.

Use `DSPy/avqa_dspy_impl.py` as the primary implementation module.
"""

from avqa_dspy_impl import *  # noqa: F401,F403


if __name__ == "__main__":
    run_batch()
