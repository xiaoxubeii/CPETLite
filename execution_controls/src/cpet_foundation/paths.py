from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(os.environ.get("CPET_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
DEFAULT_SOURCE_H5 = (
    PROJECT_ROOT.parent / "cpet_former" / "artifacts" / "dataset" / "processed" / "processed_institutes.h5"
)
DEFAULT_ARTIFACTS = PROJECT_ROOT / "artifacts"
DEFAULT_WEB_DIR = PROJECT_ROOT / "web"
