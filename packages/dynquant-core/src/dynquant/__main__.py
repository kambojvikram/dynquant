"""``python -m dynquant`` -- the CLI reached through the interpreter that runs it.

The console script in ``[project.scripts]`` is the documented entry point, but it is
not always the *reachable* one: it is written into whichever ``bin/`` the install
chose, and a driver that shells out has no guarantee that directory is on ``PATH``.
``sys.executable -m dynquant`` has that guarantee by construction -- it is the same
interpreter, therefore the same environment, therefore the same package -- which is
why the phase-3 drivers build their subprocesses that way.

This file exists because one of them already does. ``scripts/run_s1_headroom.py``
assembles ``[sys.executable, "-m", "dynquant", "eval", ...]`` and would have died on
every cell with *"No module named dynquant.__main__"*; the screen that produced the
committed S1 records ran through ``-m dynquant.cli`` instead, so the driver's form was
never exercised. Adding the module is the fix rather than rewriting the call sites,
because ``-m dynquant`` is what a user types and expects to work.
"""

from __future__ import annotations

import sys

from dynquant.cli import main

if __name__ == "__main__":
    sys.exit(main())
