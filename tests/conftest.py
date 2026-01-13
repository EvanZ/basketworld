import sys
from pathlib import Path

# Ensure repository root is on sys.path so `import app.backend...` works in tests
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
