# Bounded real-data smoke test

On 13 November 2026, the four-YAML submission code underlying this release was
tested with authorised real CPET records in an existing environment. The public
release has identical algorithm source and scientific configuration. This was
a functional smoke test, not a reproduction of the paper's full training or
external performance analysis.

## Coverage and protocol

- All 13 catalogue entries completed: one teacher fit, one teacher export,
  four principal student fits, six control-study fits and one auxiliary fit.
- Each of the 12 fits used four training examinations and one internal-validation
  examination, one epoch, batch size two and seed 42. No external examinations
  were used for training or checkpoint selection.
- The delivered `prepare` CLI generated native commands. Smoke budgets were
  changed only in local generated configurations before invoking those commands;
  the complete default `run --execute` training schedule was not run.
- Architecture, losses and precision were retained. Producer input scalers and
  split membership were retained; native decoder statistics were computed on
  the selected smoke training subset.
- The existing runtime used Python 3.12.13, PyTorch 2.4.1+cu121, h5py 3.14.0,
  NumPy 2.5.3 and an NVIDIA RTX 4090 D. No fresh environment was installed.

## Checks and input requirements

Training produced finite losses and saved weights. Checkpoints reloaded with
matching state keys and values; trajectory mean/variance and endpoint predictions
were finite and repeated inference agreed within 1e-6. Statistics that were
undefined because a reference or comparison was absent were not treated as
prediction failures. Teacher-only access manifests confirmed that raw measured
training targets were not opened; measured-supervision runs did not load a teacher.

The auxiliary observable-weight setting requires enriched supervision containing
`teacher_observability_weight`. An ordinary teacher export was correctly rejected
for that setting. An unchanged subset of an approved enriched real-data export
was used instead; weights were not fabricated. Other student branches used the
new smoke teacher export. Shared smoke inputs test control-branch execution;
formal control-study reproduction requires that study's original inputs.

## Limits

The smoke test supports loading, backpropagation, export, checkpoint reload and
finite inference for the tested small subset. It does not establish full-schedule
convergence, large-cohort scaling, seed robustness, clinical accuracy, fresh-install
compatibility or exact reproduction of study results. Private records, exports,
logs, weights and record-level predictions are excluded from this release.
