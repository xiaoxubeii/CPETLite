# CPETLite v1.1.0

CPETLite provides a teacher–student framework for reconstructing CPET curves
and estimating endpoints from limited physiological signals. This release adds
portable configuration and a unified command interface to the archived v1.0.0
principal/control implementations. Scientific model and loss code is unchanged.
See [PROVENANCE.md](PROVENANCE.md), [CHANGELOG.md](CHANGELOG.md) and
[CODE_AVAILABILITY.md](CODE_AVAILABILITY.md).

## What you can run without clinical data

Use an existing compatible Python environment; the upstream review dependencies
are in `requirements-review.txt`. A clean environment can be prepared separately
with `python -m pip install -r requirements-review.txt`. This dependency file is
for reporting/synthetic CPU checks, not a guarantee of GPU training equivalence.
Historical observed environments are retained in `environments/`.

From this directory:

```sh
python scripts/cpetlite_release.py check
python scripts/cpetlite_release.py check --synthetic
python -m pytest -q tests/test_experiment_cli.py
python scripts/run_experiment.py list
```

These verify source integrity, synthetic implementation behaviour and the portable interface. They do not reproduce clinical performance,
retrain models or reconstruct examination-level intervals from summary CSVs.

## 1. Set your local paths once

Edit `configs/paths.yaml`. Its distributed input values are null: no original
server directories or clinical files are embedded. Fill only the paths needed
for your selected operation. Every relative path is interpreted relative to
this YAML file, regardless of your shell's working directory.
This is the designated user-editable file: its distributed default hash is
recorded separately, and editing it does not invalidate the source integrity
check. The central scientific configuration and code remain covered by that check.

| Group/key | Purpose |
| --- | --- |
| `principal.source_h5` | Approved clinical HDF5 for teacher/direct-measured training and principal scoring |
| `principal.manifest` | Approved registry manifest used by the teacher configuration |
| `principal.supervision` | Frozen teacher export for principal student supervision and the original matched-input measured baseline |
| `principal.teacher_checkpoint` | Teacher weights for export and principal scoring |
| `controls.source_h5` | Approved control-study HDF5 for scoring |
| `controls.supervision` | Frozen control-study teacher export |
| `controls.teacher_checkpoint` | Teacher weights for control-study scoring |
| `output_root` | Directory for generated configurations, models, logs and evaluation outputs |

Use the fixed split/scaling information embedded in the approved derivative and
teacher export. Original null split/scaler fields remain null; do not generate
new splits or refit scalers on validation/test data. Principal and control-study
files must correspond to their respective original experiment lineage.

## 2. Choose an experiment and seeds

The distributed source contains exactly four YAML files:

```text
configs/paths.yaml
configs/experiments.yaml
preprocessing/standard_columns.yaml
preprocessing/standard_field_rules.yaml
```

`configs/experiments.yaml` contains the catalogue, shared scientific parameters,
and explicit differences for each experiment. There are no separate template
files. A parameter set selects one shared base and supplies complete top-level
overrides; nested dictionaries such as loss weights are replaced as whole values,
not implicitly merged. The resolved settings are verified against the original
experiments. Users normally edit only `configs/paths.yaml`. Supported student seeds
are 17, 42 and 73. Teacher training and export use 42. Seed-expanded copies are generated
only when requested, outside the distributed source by default. These generated
run records are not additional files in the distributed four-YAML configuration.

- Principal comparison: `principal.transformer.teacher`,
  `principal.transformer.measured`, `principal.gru.teacher`,
  `principal.gru.measured`.
- Controls: `controls.transformer.full_package`,
  `controls.transformer.mean_endpoint`, `controls.transformer.protocol_static`,
  and the corresponding `controls.gru.*` identifiers.
- Provider steps: `teacher.train`, `teacher.export`.
- Additional archived diagnostic setting: `auxiliary.transformer.observable`;
  this is separate from the principal comparison and control-study baselines.

Prepare a configuration and inspect its command plan without starting training:

```sh
python scripts/run_experiment.py prepare \
  --experiment principal.transformer.teacher --seeds 17 42 73 \
  --paths configs/paths.yaml
```

The JSON plan reports resolved configuration paths and any input-path settings
still missing. `prepare` can run without clinical files. It preserves null raw
data/checkpoint fields in teacher-supervised students, and separates their
finite-signal inputs from teacher-generated supervision.

The auxiliary `auxiliary.transformer.observable` experiment additionally requires
an enriched teacher-supervision HDF5 containing `teacher_observability_weight`.
The ordinary `teacher.export` output does not supply this field. Use an approved
enriched export with the matching record/split/scaler lineage, supplied through
`principal.supervision` in a separate local paths file. Do not fabricate weights
or treat the ordinary export as sufficient for this auxiliary diagnostic.

## 3. Train or export only when authorised inputs are available

To run the same selected experiment, explicit execution is required:

```sh
python scripts/run_experiment.py run \
  --experiment principal.transformer.teacher --seeds 17 42 73 \
  --paths configs/paths.yaml --execute
```

The adapter invokes the existing snapshot's training CLI; it does not implement
another trainer. Seeds run sequentially. Logs are saved beside resolved configs.
Missing inputs stop execution before training. Existing model/output directories
are never overwritten; select a new `output_root` for a fresh run.

The provider can select `teacher.train` and then `teacher.export` through the
same interface. After teacher training, set `principal.teacher_checkpoint` to
its `model.pt`. Export writes `distillation.h5` under the export run's output
directory; set `principal.supervision` to that file before training students.
Approved existing exports/weights can be supplied instead. Control-study exports
must retain that study's original lineage; they are not automatically regenerated
or substituted by the principal export.

## 4. Evaluate frozen students separately

```sh
python scripts/run_experiment.py evaluate \
  --experiment principal.gru.teacher --seeds 17 \
  --paths configs/paths.yaml --split test
```

This is also a dry-run unless `--execute` is added. By default it expects
`model.pt` in the selected run's model directory. For one seed, an independently
supplied model can be selected with `--checkpoint /your/path/model.pt`.
Evaluation requires approved raw HDF5 and teacher weights. The original principal
evaluation settings, including their fixed scoring seed 42, are retained independently
of the selected student's training seed. Control evaluation uses its original
minimal secure-data configuration. `test` selects the external split;
`validation` selects internal validation. Choose checkpoints on internal validation
only; external results must not select configurations or checkpoints.
Internal and external evaluation use separate output/configuration/log paths.

The principal measured-supervision baseline also uses the original export's
matched-input information through `matched_input_package`. Its targets remain
measured values; this input-alignment dependency is retained rather than silently
removed. Supply both `principal.source_h5` and `principal.supervision` for it.

To replay the original fixed-prediction intervals, use
`python scripts/cpetlite_release.py replay-intervals --help` and supply the
separately approved archives/reference summaries matching the required hashes.
This retains the upstream numerical replay contract; no clinical replay was
performed when reorganising this archive.

## Verification and aggregate reporting

The same scientific code and central configuration were exercised in a bounded
real-data smoke test covering all 13 catalogue entries. See
[SMOKE_TEST.md](SMOKE_TEST.md) for the protocol, input requirements and limits.
No clinical data or smoke checkpoints are included in this public release.

Aggregate CSVs, the source workbook and `reporting/verify_summary.py` remain in
the paper's Supplementary Material 2 rather than this public code-only tree.
Use that material for aggregate arithmetic checks. The interval-replay command
above still requires separately authorised, hash-matched inputs.

## Scope

No clinical time-series records, aggregate clinical source tables, examination-level
predictions, teacher targets or model weights are included. A reader cannot reproduce the complete clinical
study from this code-only material. Source/check/synthetic equivalence has been
tested; clean-room GPU retraining and a fresh dependency installation have not.
The v1.0.0 tag and its Zenodo archive remain unchanged. Cite v1.1.0 using
<https://doi.org/10.5281/zenodo.24201819>. Citation-only updates on main do not
change the fixed v1.1.0 tag or ZIP.
