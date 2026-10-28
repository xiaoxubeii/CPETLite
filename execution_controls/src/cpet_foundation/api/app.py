from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from cpet_foundation.data.sequences import (
    CpetSequenceDataset,
    SequenceConfig,
    normalization_options_from_config,
)
from cpet_foundation.inference.digital_twin import SAFETY_NOTICE, infer_prefix, load_checkpoint, load_demo_sample
from cpet_foundation.inference.wearable_twin import (
    action_conditioned_payload,
    load_wearable_checkpoint,
    load_wearable_demo_sample,
    wearable_payload,
    write_wearable_demo_payload,
)
from cpet_foundation.paths import DEFAULT_ARTIFACTS, DEFAULT_WEB_DIR


DEFAULT_WORKBENCH_MODEL_DIR = "protocol_sde_full"
DEFAULT_WEARABLE_MODEL_DIR = "wearable_student_full"


class CorrectionRequest(BaseModel):
    field: str = Field(min_length=1)
    corrected_value: float | str
    user_id: str | None = None
    reason: str = Field(min_length=3)


def _require_workbench_token(token: str | None) -> None:
    expected = os.environ.get("CPET_WORKBENCH_TOKEN", "local-research")
    if token != expected:
        raise HTTPException(status_code=401, detail="Workbench audit token required")


class WorkbenchStore:
    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = artifacts_dir
        self._live_demo_cache: tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]] | None = None
        self._live_model_cache: tuple[Any, dict[str, Any]] | None = None
        self._live_case_index_cache: tuple[list[dict[str, Any]], dict[str, tuple[dict[str, Any], dict[str, Any]]]] | None = None
        self._live_case_index_limit = 0
        self._wearable_model_cache: tuple[Any, dict[str, Any]] | None = None

    @property
    def demo_path(self) -> Path:
        return self.artifacts_dir / "demo" / "case_payload.json"

    @property
    def full_model_dir(self) -> Path:
        raw_model_dir = os.environ.get("CPET_WORKBENCH_MODEL_DIR", DEFAULT_WORKBENCH_MODEL_DIR)
        model_dir = Path(raw_model_dir)
        if model_dir.is_absolute():
            return model_dir
        if model_dir.parts and model_dir.parts[0] == "models":
            return self.artifacts_dir / model_dir
        return self.artifacts_dir / "models" / model_dir

    @property
    def wearable_model_dir(self) -> Path:
        raw_model_dir = os.environ.get("CPET_WEARABLE_MODEL_DIR", DEFAULT_WEARABLE_MODEL_DIR)
        model_dir = Path(raw_model_dir)
        if model_dir.is_absolute():
            return model_dir
        if model_dir.parts and model_dir.parts[0] == "models":
            return self.artifacts_dir / model_dir
        return self.artifacts_dir / "models" / model_dir

    def _read_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Missing artifact: {path.name}")
        return json.loads(path.read_text(encoding="utf-8"))

    def demo_payload(self) -> dict[str, Any]:
        return self._read_json(self.demo_path)

    def _resolve_source_h5(self, checkpoint_path: Path, config: dict[str, Any]) -> Path:
        root = checkpoint_path.resolve().parents[3]
        source_h5 = Path(config["source_h5"])
        if not source_h5.is_absolute():
            source_h5 = (root / source_h5).resolve()
        return source_h5

    def _load_live_case_index(self, limit: int = 60) -> tuple[list[dict[str, Any]], dict[str, tuple[dict[str, Any], dict[str, Any]]]]:
        if self._live_case_index_cache is not None and self._live_case_index_limit >= limit:
            return self._live_case_index_cache
        checkpoint_path = self.full_model_dir / "model.pt"
        if not checkpoint_path.exists():
            raise HTTPException(status_code=404, detail=f"Missing workbench model checkpoint for case index: {checkpoint_path}")
        model, checkpoint = load_checkpoint(checkpoint_path)
        self._live_model_cache = (model, checkpoint)
        config = checkpoint["config"]
        checkpoint_root = checkpoint_path.resolve().parents[3]
        source_h5 = self._resolve_source_h5(checkpoint_path, config)
        dataset = CpetSequenceDataset(
            SequenceConfig(
                source_h5=source_h5,
                feature_columns=tuple(config["feature_columns"]),
                metadata_columns=tuple(config["metadata_columns"]),
                max_len=int(config["max_len"]),
                limit_exams=limit,
                **normalization_options_from_config(config, checkpoint_root),
            )
        )
        cases: list[dict[str, Any]] = []
        samples_by_id: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for index, raw_sample in enumerate(dataset.samples):
            exam_id = str(raw_sample.get("exam_id", f"case-{index}"))
            case = {
                "exam_id": exam_id,
                "display_id": exam_id,
                "source_institute": str(raw_sample.get("institute", "unavailable")),
                "split": "live",
                "identifier_policy": "pseudonymized_exam_id",
                "sequence_completeness": str(raw_sample.get("sequence_completeness", "unknown")),
                "vo2peak_label_provenance": str(raw_sample.get("vo2peak_label_provenance", "unknown")),
            }
            cases.append(case)
            samples_by_id[exam_id] = (
                dataset[index],
                {
                    "feature_columns": list(config["feature_columns"]),
                    "metadata_columns": list(config["metadata_columns"]),
                    "case": case,
                },
            )
        self._live_case_index_cache = (cases, samples_by_id)
        self._live_case_index_limit = limit
        if self._live_demo_cache is None and samples_by_id:
            sample, meta = next(iter(samples_by_id.values()))
            self._live_demo_cache = (model, checkpoint, sample, meta)
        return self._live_case_index_cache

    def live_demo_payload(self, prefix_fraction: float) -> dict[str, Any]:
        if self._live_demo_cache is None:
            checkpoint_path = self.full_model_dir / "model.pt"
            if not checkpoint_path.exists():
                raise HTTPException(status_code=404, detail=f"Missing workbench model checkpoint for live inference: {checkpoint_path}")
            model, checkpoint = load_checkpoint(checkpoint_path)
            self._live_model_cache = (model, checkpoint)
            config = checkpoint["config"]
            source_h5 = self._resolve_source_h5(checkpoint_path, config)
            sample, meta = load_demo_sample(config, source_h5)
            self._live_demo_cache = (model, checkpoint, sample, meta)
        model, checkpoint, sample, meta = self._live_demo_cache
        result = infer_prefix(model, sample, prefix_fraction)
        return {
            "model": "CPET-World",
            "architecture": "variable-aware masked encoder + controlled latent state-space decoder + probabilistic trajectory head",
            "case": meta["case"],
            "features": meta["feature_columns"],
            "result": result,
            "metrics": checkpoint.get("metrics", {}),
            "safety_notice": SAFETY_NOTICE,
            "live_inference": True,
        }

    def live_case_payload(self, exam_id: str, prefix_fraction: float) -> dict[str, Any]:
        checkpoint_path = self.full_model_dir / "model.pt"
        if not checkpoint_path.exists():
            raise HTTPException(status_code=404, detail=f"Missing workbench model checkpoint for live inference: {checkpoint_path}")
        if self._live_model_cache is None:
            model, checkpoint = load_checkpoint(checkpoint_path)
            self._live_model_cache = (model, checkpoint)
        model, checkpoint = self._live_model_cache
        _cases, samples_by_id = self._load_live_case_index()
        if exam_id not in samples_by_id:
            raise HTTPException(status_code=404, detail="Case not present in local workbench artifacts")
        sample, meta = samples_by_id[exam_id]
        result = infer_prefix(model, sample, prefix_fraction)
        return {
            "model": "CPET-World",
            "architecture": "variable-aware masked encoder + controlled latent state-space decoder + probabilistic trajectory head",
            "case": meta["case"],
            "features": meta["feature_columns"],
            "result": result,
            "metrics": checkpoint.get("metrics", {}),
            "safety_notice": SAFETY_NOTICE,
            "live_inference": True,
        }

    def model_status(self) -> dict[str, Any]:
        manifest_path = self.full_model_dir / "manifest.json"
        metrics_path = self.full_model_dir / "metrics.json"
        promotion_path = self.full_model_dir / "promotion.json"
        return {
            "manifest": json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {},
            "metrics": json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {},
            "promotion": json.loads(promotion_path.read_text(encoding="utf-8")) if promotion_path.exists() else {"eligible": False},
        }

    def wearable_model_status(self) -> dict[str, Any]:
        manifest_path = self.wearable_model_dir / "manifest.json"
        metrics_path = self.wearable_model_dir / "metrics.json"
        return {
            "manifest": json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {},
            "metrics": json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {},
        }

    def live_wearable_demo_payload(self, prefix_fraction: float) -> dict[str, Any]:
        checkpoint_path = self.wearable_model_dir / "model.pt"
        if not checkpoint_path.exists():
            raise HTTPException(status_code=404, detail=f"Missing wearable model checkpoint for live inference: {checkpoint_path}")
        output_path = self.artifacts_dir / "demo" / "wearable_case_payload.json"
        return write_wearable_demo_payload(checkpoint_path, output_path, prefix_fraction)

    def live_wearable_action_demo_payload(self, prefix_fraction: float) -> dict[str, Any]:
        checkpoint_path = self.wearable_model_dir / "model.pt"
        if not checkpoint_path.exists():
            raise HTTPException(status_code=404, detail=f"Missing wearable model checkpoint for action inference: {checkpoint_path}")
        if self._wearable_model_cache is None:
            self._wearable_model_cache = load_wearable_checkpoint(checkpoint_path)
        model, checkpoint = self._wearable_model_cache
        config = checkpoint["config"]
        source_h5 = self._resolve_source_h5(checkpoint_path, config)
        sample, meta = load_wearable_demo_sample(config, source_h5)
        return action_conditioned_payload(model, checkpoint, sample, meta, prefix_fraction)

    def live_wearable_case_payload(self, exam_id: str, prefix_fraction: float) -> dict[str, Any]:
        checkpoint_path = self.wearable_model_dir / "model.pt"
        if not checkpoint_path.exists():
            raise HTTPException(status_code=404, detail=f"Missing wearable model checkpoint for live inference: {checkpoint_path}")
        if self._wearable_model_cache is None:
            self._wearable_model_cache = load_wearable_checkpoint(checkpoint_path)
        model, checkpoint = self._wearable_model_cache
        _cases, samples_by_id = self._load_live_case_index()
        if exam_id not in samples_by_id:
            raise HTTPException(status_code=404, detail="Case not present in local workbench artifacts")
        sample, meta = samples_by_id[exam_id]
        return wearable_payload(model, checkpoint, sample, meta, prefix_fraction)

    def live_wearable_action_case_payload(self, exam_id: str, prefix_fraction: float) -> dict[str, Any]:
        checkpoint_path = self.wearable_model_dir / "model.pt"
        if not checkpoint_path.exists():
            raise HTTPException(status_code=404, detail=f"Missing wearable model checkpoint for action inference: {checkpoint_path}")
        if self._wearable_model_cache is None:
            self._wearable_model_cache = load_wearable_checkpoint(checkpoint_path)
        model, checkpoint = self._wearable_model_cache
        _cases, samples_by_id = self._load_live_case_index()
        if exam_id not in samples_by_id:
            raise HTTPException(status_code=404, detail="Case not present in local workbench artifacts")
        sample, meta = samples_by_id[exam_id]
        return action_conditioned_payload(model, checkpoint, sample, meta, prefix_fraction)

    def list_cases(self, query: str = "", center: str = "", limit: int = 40) -> list[dict[str, Any]]:
        cases, _samples = self._load_live_case_index(limit=max(limit, 1))
        query_clean = query.strip().lower()
        center_clean = center.strip().lower()
        filtered = [
            case
            for case in cases
            if (not query_clean or query_clean in str(case.get("exam_id", "")).lower())
            and (not center_clean or center_clean == str(case.get("source_institute", "")).lower())
        ]
        return filtered[:limit]

    def curve_payload(self, exam_id: str) -> dict[str, Any]:
        payload = self.demo_payload()
        case = payload.get("case", {})
        expected_id = str(case.get("exam_id", "demo-case"))
        if exam_id != expected_id:
            raise HTTPException(status_code=404, detail="Case not present in local workbench artifacts")
        result = payload.get("result", {})
        return {
            "case": {"exam_id": expected_id, **case},
            "features": payload.get("features", []),
            "observed": result.get("observed", []),
            "trajectory_mu": result.get("trajectory_mu", []),
            "trajectory_sigma": result.get("trajectory_sigma", []),
            "phase_boundaries": result.get("phase_boundaries", []),
            "prefix_steps": result.get("prefix_steps"),
        }

    def inference_payload(self, exam_id: str) -> dict[str, Any]:
        payload = self.demo_payload()
        case = payload.get("case", {})
        expected_id = str(case.get("exam_id", "demo-case"))
        if exam_id != expected_id:
            raise HTTPException(status_code=404, detail="Case not present in local workbench artifacts")
        return {
            "case": {"exam_id": expected_id, **case},
            "model": payload.get("model", "CPET-World"),
            "result": payload.get("result", {}),
            "metrics": payload.get("metrics", {}),
            "model_status": self.model_status().get("promotion", {}),
            "safety_notice": payload.get("safety_notice", SAFETY_NOTICE),
        }

    def residual_payload(self, exam_id: str) -> dict[str, Any]:
        inference = self.inference_payload(exam_id)
        result = inference["result"]
        return {
            "case": inference["case"],
            "features": self.demo_payload().get("features", []),
            "residual": result.get("residual", []),
            "uncertainty": result.get("uncertainty", []),
        }

    def _correction_path(self, exam_id: str) -> Path:
        safe_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in exam_id)
        return self.artifacts_dir / "corrections" / f"{safe_id}.jsonl"

    def list_corrections(self, exam_id: str) -> list[dict[str, Any]]:
        path = self._correction_path(exam_id)
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def save_correction(self, exam_id: str, request: CorrectionRequest, reviewer_id: str) -> dict[str, Any]:
        result = self.inference_payload(exam_id)["result"]
        audit = {
            "audit_id": str(uuid4()),
            "exam_id": exam_id,
            "field": request.field,
            "original_value": result.get(request.field),
            "corrected_value": request.corrected_value,
            "user_id": reviewer_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "reason": request.reason,
        }
        path = self._correction_path(exam_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(audit, ensure_ascii=False) + "\n")
        return audit

    def report_payload(self, exam_id: str) -> dict[str, Any]:
        inference = self.inference_payload(exam_id)
        result = inference["result"]
        corrections = self.list_corrections(exam_id)
        status = inference["model_status"]
        correction_lines = "\n".join(
            f"- {item['field']}: {item['original_value']} -> {item['corrected_value']} "
            f"by {item['user_id']} ({item['reason']})"
            for item in corrections
        ) or "- No clinician corrections recorded."
        markdown = (
            "# CPET-World Research Report\n\n"
            f"Case: `{exam_id}`\n\n"
            f"Model status: `{status.get('status', 'blocked')}`; eligible: `{status.get('eligible', False)}`\n\n"
            "## Endpoint Estimates\n\n"
            f"- AT normalized time: {result.get('at_norm')}\n"
            f"- VO2peak: {result.get('vo2peak')}\n"
            f"- Recommendation: {result.get('recommendation')}\n"
            f"- Information gain: {result.get('information_gain')}\n\n"
            "## Clinician Corrections\n\n"
            f"{correction_lines}\n\n"
            "## Safety Label\n\n"
            f"{inference.get('safety_notice', SAFETY_NOTICE)}\n"
        )
        return {
            "case": inference["case"],
            "markdown": markdown,
            "corrections": corrections,
            "model_status": status,
            "safety_notice": inference.get("safety_notice", SAFETY_NOTICE),
        }


def create_app(artifacts_dir: Path | None = None, web_dir: Path | None = DEFAULT_WEB_DIR) -> FastAPI:
    app = FastAPI(title="CPET-World", version="0.1.0")
    store = WorkbenchStore(artifacts_dir or DEFAULT_ARTIFACTS)

    @app.middleware("http")
    async def prevent_workbench_asset_cache(request: Any, call_next: Any) -> Any:
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store, max-age=0"
            response.headers["Pragma"] = "no-cache"
        return response

    @app.get("/api/demo")
    def demo(prefix: float | None = Query(default=None, ge=0.05, le=0.95)) -> JSONResponse:
        if prefix is not None:
            return JSONResponse(store.live_demo_payload(prefix))
        if store.demo_path.exists():
            return JSONResponse(store.demo_payload())
        return JSONResponse(
            {
                "model": "CPET-World",
                "error": "Demo payload not generated. Run `python -m cpet_foundation.cli demo --config configs/train_smoke.yaml`.",
            },
            status_code=404,
        )

    @app.get("/api/model/status")
    def model_status() -> dict[str, Any]:
        return store.model_status()

    @app.get("/api/wearable/model/status")
    def wearable_model_status() -> dict[str, Any]:
        return store.wearable_model_status()

    @app.get("/api/wearable/demo")
    def wearable_demo(prefix: float = Query(default=0.6, ge=0.05, le=0.95)) -> JSONResponse:
        return JSONResponse(store.live_wearable_demo_payload(prefix))

    @app.get("/api/wearable/actions")
    def wearable_actions(prefix: float = Query(default=0.6, ge=0.05, le=0.95)) -> JSONResponse:
        return JSONResponse(store.live_wearable_action_demo_payload(prefix))

    @app.get("/api/cases/{exam_id}/wearable/demo")
    def wearable_case_demo(exam_id: str, prefix: float = Query(default=0.6, ge=0.05, le=0.95)) -> JSONResponse:
        return JSONResponse(store.live_wearable_case_payload(exam_id, prefix))

    @app.get("/api/cases/{exam_id}/wearable/actions")
    def wearable_case_actions(exam_id: str, prefix: float = Query(default=0.6, ge=0.05, le=0.95)) -> JSONResponse:
        return JSONResponse(store.live_wearable_action_case_payload(exam_id, prefix))

    @app.get("/api/cases")
    def cases(
        query: str = Query(default=""),
        center: str = Query(default=""),
        limit: int = Query(default=40, ge=1, le=100),
    ) -> dict[str, Any]:
        return {"cases": store.list_cases(query=query, center=center, limit=limit)}

    @app.get("/api/cases/{exam_id}/demo")
    def case_demo(exam_id: str, prefix: float = Query(default=0.6, ge=0.05, le=0.95)) -> JSONResponse:
        return JSONResponse(store.live_case_payload(exam_id, prefix))

    @app.get("/api/cases/{exam_id}/curve")
    def curve(exam_id: str) -> dict[str, Any]:
        return store.curve_payload(exam_id)

    @app.get("/api/cases/{exam_id}/inference")
    def inference(exam_id: str) -> dict[str, Any]:
        return store.inference_payload(exam_id)

    @app.get("/api/cases/{exam_id}/residuals")
    def residuals(exam_id: str) -> dict[str, Any]:
        return store.residual_payload(exam_id)

    @app.get("/api/cases/{exam_id}/corrections")
    def corrections(exam_id: str, x_workbench_token: str | None = Header(default=None, alias="X-Workbench-Token")) -> dict[str, Any]:
        _require_workbench_token(x_workbench_token)
        return {"corrections": store.list_corrections(exam_id)}

    @app.post("/api/cases/{exam_id}/corrections")
    def save_correction(
        exam_id: str,
        request: CorrectionRequest,
        x_workbench_token: str | None = Header(default=None, alias="X-Workbench-Token"),
        x_reviewer_id: str | None = Header(default=None, alias="X-Reviewer-Id"),
    ) -> dict[str, Any]:
        _require_workbench_token(x_workbench_token)
        reviewer_id = x_reviewer_id or "authenticated-reviewer"
        return store.save_correction(exam_id, request, reviewer_id)

    @app.get("/api/cases/{exam_id}/report")
    def report(exam_id: str, x_workbench_token: str | None = Header(default=None, alias="X-Workbench-Token")) -> dict[str, Any]:
        _require_workbench_token(x_workbench_token)
        return store.report_payload(exam_id)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        html = (web_dir or DEFAULT_WEB_DIR) / "index.html"
        return html.read_text(encoding="utf-8")

    if web_dir is not None and web_dir.exists():
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")
    return app


app = create_app()
