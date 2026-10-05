# Opt-in space-group DF integrals

This branch reads `green.df.space_group.v1` archives through the shared
`green-symmetry` host implementation. Requested factors are reconstructed in
the producer's captured auxiliary frame. The three tensor legs, affine Bloch
phases, TR/exchange chain and optional square orbital basis maps are applied
before contractions. The one-body and q-star weights retain their existing
meaning. Legacy files and default dispatch continue to work without new flags.

Use coordinated producer/consumer branches: the default symmetry dependency
is `Chenan-Phys/green-symmetry` at
`82b5c67a5c2f6d53dd0fbf80178eb1a234879041`. When requesting the `green-gpu`
custom kernel, its default repository/revision is `Chenan-Phys/green-gpu` at
`0110ef3037e4fd92f58a3d2fdba9f876549b38c9`. CMake cache variables
`GREEN_SYMMETRY_GIT_REPOSITORY`, `GREEN_SYMMETRY_GIT_TAG`,
`GREEN_GPU_GIT_REPOSITORY` and `GREEN_GPU_GIT_TAG` allow explicit overrides.
Both producer and consumers must understand the schema; old executables cannot
read it. Archives intentionally omit legacy chunk aliases.

From a fresh result directory, use the same one-body input and numerical
settings as the matching legacy calculation, adding:

```bash
mbpt.exe --input_file /path/to/input.h5 \
  --dfintegral_hf_file /path/to/hf_sg \
  --dfintegral_file /path/to/correlation_sg \
  --integral_symmetry space_group \
  --scf_type GW --kernel CPU --BETA 10 \
  --grid_file /path/to/ir/1e4.h5 --itermax 2 --restart false \
  --cuda_low_cpu_memory true --cuda_low_gpu_memory true \
  --P_sp false --Sigma_sp false --results_file result.h5
```

CPU HF/GW/GF2 and GPU HF/GW are covered. `GF2 --kernel GPU` uses GPU HF and
CPU GF2 correlation. No GPU GF2 correlation implementation is added.
The existing GPU precision behavior uses single precision when both `P_sp`
and `Sigma_sp` are true. Mixed flag combinations retain existing behavior.

Each reader has a byte-bounded transform/metric cache (64 MiB by default;
GF2 owns six readers). Set `--integral_symmetry_cache_bytes N` to change it.
`--integral_symmetry_preload_bytes N` bounds GPU representative host preload
per node (1 GiB by default), before allocation. Only representatives are
preloaded. Shared buffers remain read-only during owned reconstruction;
target-Q slices sum all source-Q rows. Individual factor/chunk buffers have
a 128 MiB bound. Descriptor memory and active output tensors are additional.

The initial profile requires scalar spherical 3D cells, positive full-rank
Cholesky metrics, and square orbital maps. Truncated/negative/ED gauges,
X2C/spinors, range-separated physical interactions, `df_ewald.h5`/`AqQ.h5`
special sidecars, special finite-size flags, coarse graining and GW q=0
extrapolation are rejected explicitly for SG. Reduced-q GW input must identify
its actual correlation gauge. Ordinary HF and default GREEN legacy Ewald
q=0 correlation sets have separate captured frames.

With `BUILD_TESTING=ON`, `test/frozen_integral_probe` runs production kernels
once on an externally supplied arbitrary complex `G_tau` tensor in
interleaved float64 `(tau,spin,k,i,j,2)` layout. It requires unreduced one-body
k points and checks that G is unchanged. Use the solver flags above plus
`--frozen_G_file /path/to/frozen_G.h5 --probe_output fresh-result.h5`.
Compare `Sigma1` and `Selfenergy` with a separately generated complete reference.
This probe does not iterate self-consistency. Shared library CTests register
format, reconstruction, ownership and preload checks; optional physical
fixtures use `GREEN_SG_VALIDATION_ROOT`.
