# CPETLite v1.1.0 provenance

This code-only release is derived from the portable four-YAML submission source,
which in turn derives from CPETLite v1.0.0. The immutable v1.0.0 archive is at
<https://doi.org/10.5281/zenodo.23734812>; its ZIP SHA256 is
`cdd4d281bcfc998429898d0b370bcd538797a2138275f88100a0c21289c1778f`.
The v1.0.0 tag and archive have not been replaced.

Both execution snapshots' complete `src/` trees (43 files), retained native
entry scripts, two preprocessing rules and central scientific parameters match
the tested submission source byte-for-byte. The submission expansion audit
compared 33 student seed configurations, teacher training/export and four
evaluation settings against the frozen originals. Only configured data/output
paths and non-scientific run labels change during expansion.

Historical configuration directories, redundant seed copies, nested source ZIP
and obsolete orchestration are replaced by four YAML files and the unified
`scripts/run_experiment.py` interface. Principal and control implementations
remain separate. An auxiliary observable-weight experiment is labelled as such.

The reporting helper no longer inserts a personal search directory. Interval
replay reads the identical metric source from the unpacked principal snapshot
instead of the removed nested ZIP. Metric and bootstrap formulas are unchanged.
The release manifest records preserved-file hashes and these adaptations.

Public-release differences from the tested submission are documentation,
citation/version metadata and the omission of submission-only aggregate CSVs
and their reporting verifier. Those remain in Supplementary Material 2. No
clinical records, teacher supervision, weights, individual predictions or private
smoke artifacts have been added to GitHub.

The v1.1.0 source tag is commit
`88afd0e4183a73135a5f3339976a0a3c871d6b1f`, archived at
<https://doi.org/10.5281/zenodo.24201819>. This DOI was verified after publication
and added to main in a citation-only follow-up. The fixed tag and its code ZIP
remain unchanged. The v1.0.0 DOI above identifies the earlier upstream release.

Four bilingual `label_zh` interface strings remain in each original
`wearable_twin.py` snapshot to preserve source bytes. User documentation and
configuration instructions are in English.
