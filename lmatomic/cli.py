"""Entry point for `lmatomic` when installed via pip.

Delegates to the original c/lmatomic script which handles all subcommands.
This wrapper exists so `pip install lmatomic` creates a `lmatomic` console
script that works without the user having to add c/ to PATH manually.
"""

import os
import sys
import runpy


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    engine_dir = os.path.join(os.path.dirname(here), "c")
    lma_script = os.path.join(engine_dir, "lmatomic")

    if not os.path.exists(lma_script):
        sys.exit(
            "lmatomic engine directory not found.\n"
            "Install from source: git clone + pip install -e ."
        )

    sys.path.insert(0, engine_dir)
    sys.argv[0] = lma_script
    runpy.run_path(lma_script, run_name="__main__")


if __name__ == "__main__":
    main()
