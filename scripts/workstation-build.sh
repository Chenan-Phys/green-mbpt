#!/usr/bin/env bash
set -euo pipefail
source "$HOME/green/env.sh"
unset PYTHONPATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source_dir="$HOME/green/qed-research"
build_dir=/data/cwei/green_runs/qed-validation/build
prefix="$HOME/green/install-qed/cavity-general"
mkdir -p "$build_dir"
args=(-S "$source_dir" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_INSTALL_PREFIX="$prefix" -DCUSTOM_KERNELS= -DBuild_Tests=ON)
for dep in green-utils green-ndarray green-h5pp green-params green-grids green-symmetry green-sc green-impurity green-opt catch2 magic_enum; do
    cached="$HOME/green/build/_deps/${dep}-src"
    if [[ -d "$cached" ]]; then
        upper=${dep^^}
        args+=("-DFETCHCONTENT_SOURCE_DIR_${upper}=$cached")
    fi
done
cmake "${args[@]}"
cmake --build "$build_dir" -j 2
cmake --install "$build_dir"
"$prefix/bin/mbpt.exe" --version
