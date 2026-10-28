"""Run the frozen review plan through existing native matrix/evaluation CLIs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

from audit_cpetlite_review_controls import audit, digest, require, verify_inputs
from audit_cpetlite_paper_predictions import audit as audit_predictions


ROOT = Path(__file__).resolve().parents[1]


def execute(script, *args):
    command = [sys.executable, "-u", str(ROOT / "scripts" / script), *map(str, args)]
    print(json.dumps({"native_command": command}), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def verify():
    plan = verify_inputs(ROOT.parent)
    audit(ROOT)
    audit_predictions(ROOT.parent / "assets/dataset.h5", plan["family"], ROOT, 36)
    verify_inputs(ROOT.parent)
    return plan


def run():
    plan = verify_inputs(ROOT.parent)
    for task in plan["tasks"]:
        output = ROOT / task["output"]
        require(not output.exists() or not any(output.iterdir()), f"Refusing existing fit output: {output}")
    for arch in ("direct", "gru"):
        matrix = ROOT / f"configs/{plan['family']}_{arch}_matrix.yaml"
        execute("run_cpetlite_v4_package_comparison.py", "--matrix", matrix,
                "--train", "--workers", plan["workers"])
        execute("evaluate_cpetlite_v4_package_comparison.py", "--matrix", matrix)
    verify()
    exports = Path(os.environ.get("CPET_AUTODL_ARTIFACT_DIR", str(ROOT.parent / "exports")))
    exports.mkdir(parents=True, exist_ok=True)
    archive = exports / "RESULTS.tar.gz"
    require(not archive.exists(), "Refusing existing results archive")
    excluded = {"__pycache__", ".pytest_cache"}
    files = [p for p in ROOT.rglob("*") if p.is_file() and not (excluded & set(p.parts))
             and p.suffix != ".pyc"]
    files += [ROOT.parent / "PLAN.json", ROOT.parent / "DEPLOYMENT_MANIFEST.json",
              ROOT.parent / "assets/teacher.pt"]
    inventory = {str(p.relative_to(ROOT.parent)): {"sha256": digest(p), "bytes": p.stat().st_size}
                 for p in sorted(files)}
    with tarfile.open(archive, "x:gz") as handle:
        for path in sorted(files):
            handle.add(path, arcname=str(path.relative_to(ROOT.parent)), recursive=False)
    (exports / "RESULTS_INVENTORY.json").write_text(json.dumps(inventory, indent=2) + "\n")
    acceptance = {"status": "verified", "family": plan["family"], "student_fits": 18,
                  "evaluations": 36, "teacher_retrained": False,
                  "plan_sha256": digest(ROOT.parent / "PLAN.json"),
                  "archive_sha256": digest(archive), "archive_bytes": archive.stat().st_size,
                  "internal_scope": plan["internal_scope"], "external_scope": plan["external_scope"]}
    (exports / "ACCEPTANCE.json").write_text(json.dumps(acceptance, indent=2) + "\n")
    print(json.dumps(acceptance), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"))
    args = parser.parse_args()
    run() if args.command == "run" else verify()
