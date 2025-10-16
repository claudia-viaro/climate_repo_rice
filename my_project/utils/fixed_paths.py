from pathlib import Path
import sys

REPO_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_DIR / "my_project" / "configs"
OUTPUTS_DIR = REPO_DIR / "outputs"
REGION_YAMLS_DIR = CONFIG_DIR / "region_yamls"
OTHER_YAMLS_DIR = CONFIG_DIR / "other_yamls"
UTILS_DIR = REPO_DIR / "my_project" / "utils"
SCRIPTS_DIR = REPO_DIR / "scripts"
# Add to sys.path
# Add repo root to sys.path to ensure my_project is found
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

__all__ = [
    "REPO_DIR",
    "CONFIG_DIR",
    "REGION_YAMLS_DIR",
    "OUTPUTS_DIR",
    "UTILS_DIR",
    "SCRIPTS_DIR",
]
