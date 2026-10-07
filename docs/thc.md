# THC evaluation modes

Build against the pinned coordinated THC green-symmetry revision. Enable GPU
with the coordinated green-gpu revision. Existing no-flag DF remains default.
The standalone Python exporter is documented in green-mbtools/docs/thc.md.

Explicit flags: `--interaction_representation thc --thc_mode reconstruct|native`.
`thc_factor_memory_mb` bounds resident X/cores; `thc_workspace_mb` bounds the
native one-q tau/frequency arrays. Additional projection, linear solve and
transform scratch exists. Both default to 512 MiB. Archive input fingerprints
must match the exact one-body input file; retain original NQ.

Reconstruction uses original-Q slices with existing HF/GW/GF2 contractions.
It preserves the raw representative-buffer ordering used by GF2 corrections.
GF2 with kernel GPU means GPU HF and CPU GF2 correlation.

Native scalar full-BZ double-precision HF/GW projects G/density into point
space and backprojects there, without reconstructing V. HF preserves spin
factors and the separate Madelung term. The bare q0 Hartree uses M M^T to
preserve GREEN's unconjugated auxiliary ordering. GW uses the legacy response
q sign (opposite the source pair q), transposed point-space bubble, existing
tau/frequency transforms, and solves `(1-Z chi) Wc=Z chi Z` by pivoted LU.
It never inserts a constant-tau bare Z or inverts a singular Z. Native q tasks
are scheduled independently of Gaussian-Q partitioning. Initially one node
leader handles each node's assigned q tasks.

Unsupported modes reject explicitly: native GF2; native IBZ/TR/spatial
reduction; spinors; native single precision; extrapolation/AqQ. Native momentum
sums remain direct. Production FFT and point-space symmetry require separate
validation. Reconstruction covariance is checked before export.

The MPI probes in test/ validate CPU/GPU-host original-Q slices and frozen
one-body G across all supported methods/modes. They refuse output reuse.
`thc_frozen_probe` uses GREEN's physical tau sample values directly. Use fresh
run directories for self-consistency; two iterations establish agreement only.
