# THC evaluation modes

Build with the pinned coordinated contraction-order revisions of green-symmetry and
green-gpu. The standalone exporter is documented in green-mbtools/docs/thc.md.
DF remains default, and is also selected by `--interaction_representation df`.
Select THC explicitly with
`--interaction_representation thc --thc_mode reconstruct|native`.
No automatic DF/THC performance crossover is assumed.

Reconstruction supplies original-Q slices to existing HF/GW/GF2 contractions.
It preserves raw representative-buffer ordering for GF2 corrections. GPU GF2
selection means GPU HF and CPU GF2 correlation. Archive input fingerprints
must match the exact one-body input file; retain original NQ.

Native scalar full-BZ double HF/GW projects density/G into interpolation space
and backprojects without reconstructing V. HF preserves spin factors, separate
Madelung correction and unconjugated q0 Hartree ordering `M M^T`. GW retains the
legacy response q sign, transposed point-space bubble and original IR transforms.
It computes correlation-only Wc without inverting a potentially singular Z or
adding constant-tau bare Z.

`--thc_gw_screening auto|point|auxiliary` defaults to auto. Point space solves
`(I-Z chi) Wc=Z chi Z`, with `Z=M M^H`. Auxiliary space solves
`P=M^H chi M; (I-P) C=P; Wc=M C M^H`. The latter is an exact algebraic identity,
including semidefinite Z. Auto selects the smaller dimension, Q or interpolation
rank. Explicit alternatives allow numerical and timing comparisons. These
screening flags require native THC GW.

Auxiliary mode compresses the summed-spin half-tau bubble before the forward
IR transform. Frequency screening and the inverse transform remain in Q space.
It expands the correlation core only for the current tau and reuses it across
spins. This reorders time-independent M through linear IR transforms without
changing the factors, fit tolerance, or screening equation. The q-dependent M
cannot be moved through momentum FFTs, whose fields remain in point space.

Native GPU auxiliary compression/expansion additionally supports the opt-in
`--thc_cuda_aux_gemm3m true` complex-double backend. It defaults to false and
leaves HF, projections, IR transforms and LU/residual checks on the standard
backend. This alternative reorders real operations within complex GEMMs;
measure the full result and runtime for the intended shape and GPU.

`--thc_gw_k_contraction direct|fft` defaults to direct. Direct supports full-BZ
inputs with validated transfer maps. FFT requires a complete Cartesian
commensurate mesh; irregular/reduced inputs reject the explicit FFT option.
The shared mapper preserves shifted cosets, shuffled ordering, actual Bloch
phases and negative-index complex correlation without a second-field conjugate.
CPU uses cached batched FFTW plans when available, with Eigen fallback.
`-DGREEN_THC_USE_FFTW=OFF` forces that portable fallback. Native GPU uses cuFFT
and keeps projection, transforms, screening and backprojection on the device.

`thc_factor_memory_mb` (default 512 MiB) bounds resident host X/cores. The separate
`thc_workspace_mb` (default 512 MiB) estimates CPU working matrices and caps owned
GPU buffers. CPU direct caches projected G across q when the declared budget
allows, otherwise streams it. Auxiliary direct retains owned-q Q-space histories
when possible and accumulates point Sigma over q before one orbital
backprojection per (tau,spin,k). Bounded alternatives stream one q, then one
(spin,k) Sigma field. FFT retains all-q histories in the selected screening
space and expands current-tau slices. GPU direct uses bounded
q tiles and FFT requires all q to fit. CPU library scratch, CUDA contexts and
library-owned allocations are additional; report process memory separately.
When histories fit but full projected G does not, CPU direct shares both
projected half-tau slices across owned q. This reuses projections with bounded
per-time storage instead of requiring a full point-space G history.
Low GPU memory trims idle buffers at stage boundaries while retaining within-stage
reuse. An allocation check leaves 512 MiB of physical GPU headroom.

Native q tasks use node leaders independently of Gaussian-Q partitioning.
FFT executes on the first node leader and reduces Sigma. Same-host MPI correctness
does not establish multi-node scaling. Native GF2, IBZ/TR/spatial reduction,
spinors, single precision and extrapolation/AqQ reject explicitly. HF stays
direct; point-space symmetry and distributed FFTs remain separate work.

The test probes validate original-Q slices and frozen physical G, and refuse
output reuse. `thc_frozen_probe` records HF and GW stage times. Use fresh directories
for self-consistency; two iterations establish agreement, not convergence.
Benchmark DF and THC at the needed accuracy on the intended basis and mesh.


## Sigma contraction and execution controls

The `feature/thc-sigma-optimization` branch adds exact complex-double execution
alternatives for native scalar, full-BZ GW. DF remains the default interaction
representation. These controls do not change the fitted factors or fit tolerances.

- `--thc_gw_sigma point|orbital|auto` defaults to `point`. `orbital` requires
  auxiliary screening and direct momentum sums. It constructs fixed vertices
  `V[k,kp,A,a,i] = sum_p conj(X[k,p,a])*X[kp,p,i]*M[q,p,A]`, then contracts
  `Sigma[k] -= sum_(kp,A,B) V_A G[kp] V_B^H C[q,A,B]/Nk`. The screened core `C`
  stays in Q space; Sigma never expands it into a point matrix. The CPU and GPU
  layouts differ internally and are tested against independent point expressions.
  The cache scales as `Nk*Nq*n*n*Q` complex values and can be reused across solver
  iterations. Cold-run timings include its construction. `auto` uses a conservative
  cached-point operation estimate with a 10% margin, not a timing guarantee.
  In unsupported momentum/screening modes it selects `point`. GPU auto also
  selects point when vertex workspace does not fit or a tau batch is requested.
- `--thc_fft_reuse_screening true|false` defaults to `true`. In momentum FFT mode
  it transforms the screened interaction once per tau and shares it across spins.
  The original non-conjugating correlation, negative-frequency mapping and shifted
  mesh values are preserved. Two-spin Sigma uses five FFT calls per tau instead
  of six. The complete Nt=110 example uses 880 rather than 990 calls, including
  the unchanged bubble stage. A one-point mesh needs no transforms.
- `--thc_cpu_threads N` defaults to 1, allows 1..64, and bounds parallel Sigma
  tau workers by the declared workspace. It applies to auxiliary/direct CPU GW
  with retained owned-q histories. Streaming falls back to one worker. Each worker
  writes disjoint tau slices; q accumulation order is preserved. Use one BLAS
  thread when testing these workers to avoid nested CPU oversubscription.
- `--thc_cuda_sigma_batch N` defaults to 0. Values 1..32 batch point-space
  projection/backprojection over spins and up to N adjacent tau slices. A final
  partial tile is supported. Extra projected/point fields consume workspace;
  this option is incompatible with explicit orbital Sigma. Zero preserves the
  original per-spin point path. Benchmark the size for the intended dimensions.

MBPT enables OpenMP when available with `GREEN_THC_OPENMP=ON` (default). Set it
OFF for a serial build; requesting CPU workers above one then produces an error.
A pre-existing GF2 tau loop was rewritten with one induction variable to make it
valid OpenMP syntax, preserving all original tau indices. Rebuild the coordinated
symmetry, GPU and MBPT revisions together after changing resident APIs.
