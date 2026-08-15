"""The positive control, moved one process away.

`--target-processes application-only` instruments the process compute-sanitizer launches and
nothing it spawns. The parity suite spawns a subprocess, so a clean report under the default
says nothing about the kernels that ran inside it. This makes that concrete: the identical
out-of-bounds gather, once directly and once behind a `subprocess.run`, under both settings.
"""

import subprocess
import sys

sys.exit(subprocess.run([sys.executable, "/workspace/sanpos.py"], check=False).returncode)
