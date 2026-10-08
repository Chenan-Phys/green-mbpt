# THC evaluation modes

Build with the pinned coordinated performance revisions of green-symmetry and
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
allows, otherwise streams it; FFT retains all-q Wc. GPU direct uses bounded
q tiles and FFT requires all q to fit. CPU library scratch, CUDA contexts and
library-owned allocations are additional; report process memory separately.
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
