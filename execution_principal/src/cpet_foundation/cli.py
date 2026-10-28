from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from cpet_foundation.data.h5_registry import (
    RegistryConfig,
    generate_splits,
    register_dataset,
    resolve_path,
    summarize_dataset,
)
from cpet_foundation.eval.pipeline import evaluate_internal_splits, evaluate_loco_checkpoint
from cpet_foundation.eval.promotion import evaluate_promotion
from cpet_foundation.inference.digital_twin import write_demo_payload
from cpet_foundation.training.runner import reevaluate_wearable_checkpoint, train_from_config


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def register_command(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    raw = _load_yaml(config_path)
    base_dir = config_path.parents[1]
    cfg = RegistryConfig(
        source_h5=resolve_path(raw["source_h5"], base_dir),
        output_dir=resolve_path(raw.get("output_dir", "artifacts/registry"), base_dir),
        split_dir=resolve_path(raw.get("split_dir", "artifacts/splits"), base_dir),
        required_features=tuple(raw.get("required_features", [])),
        target_column=raw.get("target_column", "Time_at_AT"),
        seed=int(raw.get("seed", 42)),
    )
    manifest = register_dataset(cfg)
    splits = generate_splits(cfg.source_h5, cfg.split_dir, cfg.seed, cfg.target_column)
    summary = summarize_dataset(cfg.source_h5, cfg.output_dir, cfg.target_column)
    print(f"registered={manifest['total_exams']} rows={manifest['total_rows']} valid={manifest['schema_valid']}")
    print(f"splits=random:{len(splits['random']['train'])}/{len(splits['random']['validation'])}/{len(splits['random']['test'])}")
    print(f"summary_exams={summary['exam_count']}")


def train_command(args: argparse.Namespace) -> None:
    result = train_from_config(Path(args.config).resolve())
    print(result)


def reevaluate_wearable_command(args: argparse.Namespace) -> None:
    result = reevaluate_wearable_checkpoint(
        Path(args.config).resolve(),
        Path(args.checkpoint).resolve() if args.checkpoint else None,
    )
    print(result)


def demo_command(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    root = config_path.parents[1]
    raw = _load_yaml(config_path)
    checkpoint = root / raw["output_dir"] / "model.pt"
    output = root / "artifacts" / "demo" / "case_payload.json"
    payload = write_demo_payload(checkpoint, output, prefix_fraction=float(args.prefix))
    print(f"demo_payload={output}")
    print(f"recommendation={payload['result']['recommendation']} info_gain={payload['result']['information_gain']:.4f}")


def evaluate_command(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.random_split:
        internal = evaluate_internal_splits(
            checkpoint,
            Path(args.random_split).resolve(),
            output_dir / f"{args.run_name}_internal.json",
            limit_per_split=args.limit_per_split,
            prefix_fractions=tuple(float(item) for item in args.prefix_fractions.split(",")),
        )
        print(f"internal_eval={output_dir / f'{args.run_name}_internal.json'} splits={list(internal['splits'])}")
    if args.loco_split:
        loco = evaluate_loco_checkpoint(
            checkpoint,
            Path(args.loco_split).resolve(),
            output_dir / f"{args.run_name}_loco.json",
            limit_per_fold=args.limit_per_fold,
            prefix_fraction=float(args.loco_prefix),
        )
        print(f"loco_eval={output_dir / f'{args.run_name}_loco.json'} folds={len(loco['folds'])}")


def promote_command(args: argparse.Namespace) -> None:
    model_dir = Path(args.model_dir).resolve()
    decision = evaluate_promotion(
        model_dir,
        output_path=model_dir / "promotion.json",
        internal_eval_path=Path(args.internal_eval).resolve() if args.internal_eval else None,
        loco_eval_path=Path(args.loco_eval).resolve() if args.loco_eval else None,
    )
    print(f"eligible={decision['eligible']} status={decision['status']}")
    if decision["blocked_reasons"]:
        print("blocked_reasons=" + ",".join(decision["blocked_reasons"]))


def serve_command(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        if exc.name != "uvicorn":
            raise
        raise SystemExit(
            "API serving requires uvicorn; install the web-service dependencies to use 'serve'."
        ) from exc
    uvicorn.run("cpet_foundation.api.app:app", host=args.host, port=args.port, reload=False)


def main() -> None:
    parser = argparse.ArgumentParser(prog="cpet-world")
    sub = parser.add_subparsers(required=True)
    register = sub.add_parser("register")
    register.add_argument("--config", default="configs/registry.yaml")
    register.set_defaults(func=register_command)

    train = sub.add_parser("train")
    train.add_argument("--config", default="configs/train_smoke.yaml")
    train.set_defaults(func=train_command)

    reevaluate_wearable = sub.add_parser("reevaluate-wearable")
    reevaluate_wearable.add_argument("--config", required=True)
    reevaluate_wearable.add_argument("--checkpoint")
    reevaluate_wearable.set_defaults(func=reevaluate_wearable_command)

    demo = sub.add_parser("demo")
    demo.add_argument("--config", default="configs/train_smoke.yaml")
    demo.add_argument("--prefix", default=0.6, type=float)
    demo.set_defaults(func=demo_command)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", default="artifacts/models/full/model.pt")
    evaluate.add_argument("--random-split", default="artifacts/splits/split_random_seed42.json")
    evaluate.add_argument("--loco-split", default="artifacts/splits/split_loco_seed42.json")
    evaluate.add_argument("--output-dir", default="artifacts/eval")
    evaluate.add_argument("--run-name", default="full")
    evaluate.add_argument("--limit-per-split", type=int)
    evaluate.add_argument("--limit-per-fold", type=int)
    evaluate.add_argument("--prefix-fractions", default="0.35,0.60,0.85")
    evaluate.add_argument("--loco-prefix", default=0.6, type=float)
    evaluate.set_defaults(func=evaluate_command)

    promote = sub.add_parser("promote")
    promote.add_argument("--model-dir", default="artifacts/models/full")
    promote.add_argument("--internal-eval")
    promote.add_argument("--loco-eval")
    promote.set_defaults(func=promote_command)

    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8765, type=int)
    serve.set_defaults(func=serve_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
