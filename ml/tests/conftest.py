import sys
from pathlib import Path

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML))

import common  # noqa: E402,F401  (adds service/ to sys.path)
