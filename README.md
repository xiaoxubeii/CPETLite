# CPETLite

Code for offline reconstruction of cardiopulmonary exercise responses using
teacher-student learning. Public release **v1.0.0**, corresponding to the
finite-metadata principal experiment and its separate paired-control study.

## Quick check (no clinical data)

Use Python3.12.13 and the pinned reporting environment in requirements-review.txt.
For an existing compatible environment:

```sh
python scripts/cpetlite_release.py check
python scripts/cpetlite_release.py check --synthetic
```

The first command checks file hashes with only the Python standard library.
The second uses PyTorch on CPU and runs a loss/gradient/access-contract smoke
check and four zero/missingness/padding/endpoint tests separately against each
frozen snapshot. These commands do not download data, use a GPU or train models.
To install dependencies in a new environment, run
`python -m pip install -r requirements-review.txt`; installation can require network
access. The observed synthetic/reporting Torch build is recorded in environments/.

## Implementations

- execution_principal/:74 exact accepted Python/YAML files, including the teacher,
  Transformer/GRU students, export, fitting, evaluation and principal configurations.
- execution_controls/:78 exact Python/YAML files from the separately accepted
  control experiment. Keep the snapshots separate; do not mix their runtime claims.
- scripts/:integrity checks, shared-examination bootstrap and authorized interval replay.
- tests/:patient-free examples against real frozen simulator/loss interfaces.
- environments/:observed principal training/reporting environments and the explicit
  gap in the control runtime version inventory. No clean-room retraining is claimed.

The frozen snapshots retain historical nonsecret input/output paths. Edit local
working copies of configurations before use; do not assume those paths exist.

## Authorized clinical replay

No clinical or aggregate source data are supplied. Approved users must obtain the
two accepted result archives and their authorized reference-summary directory:
principal_results.json and control_results.json. Their exact hashes are recorded
in RELEASE_MANIFEST.json, without publishing their contents.

```sh
python scripts/cpetlite_release.py replay-intervals \
  --principal-results /approved/principal/RESULTS.tar.gz \
  --control-results /approved/controls/RESULTS.tar.gz \
  --reference-dir /approved/reference-summaries \
  --output /new/directory/outside-this-repository
```

All inputs must match the accepted hashes before arrays are read or outputs are
created. The replay recomputes36 fixed-prediction intervals using2000 shared
examination draws (RNG42 internal/142 external), retaining fitted seeds17/42/73.
It excludes smoke outputs by complete experiment name and performs no training.
It does not establish patient independence, clinical label validity or uncertainty
from teacher retraining/centre sampling. Reference summaries can be supplied from
the approved author-review reproduction attachment; they are not included here.

## Authorized training

Training requires separately approved Standard5.5.2 records, frozen scalers and
split definitions; package-student fitting requires the frozen teacher export.
From the applicable execution snapshot, the common training CLI is:

```sh
PYTHONPATH=src python -m cpet_foundation.cli train --config /path/to/approved/local.yaml
```

Inspect the included teacher and per-seed student YAMLs and resolve input/package/
output paths first. The provider-side export interface is
`scripts/export_teacher_only_package.py --help` inside each snapshot. This public
release has not been used for new clinical fitting or clean-room GPU retraining.

## Citation, licence and scope

See CITATION.cff and CODE_AVAILABILITY.md. Zenodo archival status is reported by
the actual linked record after publication, never by an invented DOI.
No open-source reuse licence has been granted; code is provided for inspection.
This research software performs offline reconstruction and is not a validated
clinical decision tool. Source metadata and code publication do not authorize
access to hospital records or patient-linked teacher targets/weights.
