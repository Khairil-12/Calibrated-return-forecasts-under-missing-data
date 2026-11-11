"""Entry point for the experiment grid runner."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calibmiss.run_experiment import main

if __name__ == "__main__":
    main()
