#!/usr/bin/env python3
"""Train one CPET Foundation configuration without importing API dependencies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cpet_foundation.training.runner import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    result = train_from_config(args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
