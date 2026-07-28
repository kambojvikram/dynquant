# Present so that `tests` is a real package, which the DDP reduction tests need.
#
# `test_signals_reduce.py` hands `_worker` to `torch.multiprocessing.spawn`, and the
# spawn start method pickles it by (module, qualname) -- the child then performs a
# genuine `import tests.test_signals_reduce`. Under `--import-mode=importlib` pytest
# registers test modules in `sys.modules` under that dotted name without requiring a
# package, so the parent is happy and the child is not: it raises
# `ModuleNotFoundError: No module named 'tests.test_signals_reduce'` before any test
# code runs, and the failure surfaces as a bare `ProcessExitedException` with the
# real cause buried in the child's stderr. This file plus `pythonpath = ["."]` in
# pyproject.toml is what makes the child's import resolve.
