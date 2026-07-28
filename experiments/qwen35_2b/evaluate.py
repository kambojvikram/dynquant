"""Score any checkpoint on the selected task, for measurement points 1 through 4.

The prompt, the decode settings and the scorer live in :func:`common.run_eval`, so
this script only resolves a directory into a loaded model. The only thing that
differs between invocations is where the weights came from.

Usage::

    python evaluate.py --model /workspace/runs/.../finetuned \\
                       --label "fine-tuned" --name stage3_finetuned
"""

from __future__ import annotations

import argparse
import sys

from common import load_model, run_eval, set_seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    set_seed()
    model = load_model(args.model)
    print(f"loaded {args.model}", flush=True)

    run_eval(model, label=args.label, name=args.name, limit=args.limit, extra={"model": args.model})
    return 0


if __name__ == "__main__":
    sys.exit(main())
