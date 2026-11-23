"""Web XAS pilot: scientific reduction is delegated to xraylarch."""
import os
from pathlib import Path

# Keep scientific runtime caches in this source checkout, not in a user's global
# Larch configuration. Explicit environment settings still take precedence.
_runtime = Path(__file__).resolve().parents[1] / ".runtime"
os.environ.setdefault("LARCHDIR", str(_runtime / "larch"))
os.environ.setdefault("MPLCONFIGDIR", str(_runtime / "matplotlib"))
os.environ.setdefault("XRAYLARCH_NO_PLOT", "1")

__version__ = "0.5.2rc1"
