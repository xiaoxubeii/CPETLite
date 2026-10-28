#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cpet_foundation.training.distillation_export import export_teacher_only_distillation_package


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit-exams", type=int)
    args = parser.parse_args()
    result = export_teacher_only_distillation_package(
        args.config,
        args.output,
        limit_exams=args.limit_exams,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
