#include <green/gpu/thc_gpu_resident.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuComplex.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using gpu = green::gpu::thc_gpu_resident;
using matrix = gpu::matrix;
using colmatrix = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using z = cuDoubleComplex;

void check(int status, const char* operation) {
  if (status) throw std::runtime_error(std::string(operation) + " status " + std::to_string(status));
}

template<class T> struct device_buffer {
  T* pointer = nullptr;
  size_t count = 0;
  explicit device_buffer(size_t n) : count(n) {
    check(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)), "cudaMalloc");
  }
  ~device_buffer() { if (pointer) cudaFree(pointer); }
  device_buffer(const device_buffer&) = delete;
  device_buffer& operator=(const device_buffer&) = delete;
  void upload(const T* source) {
    check(cudaMemcpy(pointer, source, count * sizeof(T), cudaMemcpyHostToDevice), "upload");
  }
  std::vector<T> download() const {
    std::vector<T> result(count);
    check(cudaMemcpy(result.data(), pointer, count * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return result;
  }
};

struct raw_handles {
  cublasHandle_t blas = nullptr;
  cusolverDnHandle_t solver = nullptr;
  raw_handles() {
    check(cublasCreate(&blas), "cublasCreate");
    check(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
    check(cusolverDnCreate(&solver), "cusolverDnCreate");
  }
  ~raw_handles() {
    cudaDeviceSynchronize();
    if (solver) cusolverDnDestroy(solver);
    if (blas) cublasDestroy(blas);
  }
};

matrix values(size_t rows, size_t cols, unsigned seed, double scale = .02) {
  std::mt19937 generator(seed);
  std::uniform_real_distribution<double> uniform(-scale, scale);
  matrix result(rows, cols);
  for (size_t i = 0; i < rows; ++i)
    for (size_t j = 0; j < cols; ++j) result(i, j) = {uniform(generator), uniform(generator)};
  return result;
}

std::vector<z> pack(const std::vector<matrix>& input) {
  std::vector<z> result;
  for (const auto& value : input) {
    colmatrix columns = value;
    for (Eigen::Index i = 0; i < columns.size(); ++i)
      result.push_back(make_cuDoubleComplex(columns.data()[i].real(), columns.data()[i].imag()));
  }
  return result;
}

matrix unpack(const std::vector<z>& input, size_t rows, size_t cols, size_t batch = 0) {
  matrix result(rows, cols);
  for (size_t j = 0; j < cols; ++j)
    for (size_t i = 0; i < rows; ++i) {
      z value = input[batch * rows * cols + j * rows + i];
      result(i, j) = {value.x, value.y};
    }
  return result;
}

struct error_summary {
  double maximum = 0, relative = 0;
  void compare(const matrix& actual, const matrix& reference) {
    if (actual.rows() != reference.rows() || actual.cols() != reference.cols())
      throw std::runtime_error("oracle dimensions differ");
    if (!actual.allFinite() || !reference.allFinite())
      throw std::runtime_error("nonfinite complex-double oracle input/output");
    maximum = std::max(maximum, (actual - reference).cwiseAbs().maxCoeff());
    relative = std::max(relative, (actual - reference).norm() / std::max(1e-30, reference.norm()));
  }
  void require() const {
    if (!std::isfinite(maximum) || !std::isfinite(relative) || maximum > 1e-10 || relative > 1e-11)
      throw std::runtime_error("complex-double CPU oracle comparison failed");
  }
};

struct timings { double gpu_ms, wall_ms, gpu_min_ms, gpu_max_ms; };

// Uploads, CPU oracles, and downloads are outside these measurements. For LU,
// identical device-to-device resets are included in both backend measurements.
template<class Function> timings measure(Function function, int repeats = 3) {
  for (int i = 0; i < 3; ++i) function();
  check(cudaDeviceSynchronize(), "warmup synchronize");
  cudaEvent_t start = nullptr, stop = nullptr;
  check(cudaEventCreate(&start), "start event");
  check(cudaEventCreate(&stop), "stop event");
  std::vector<double> gpu_times, wall_times;
  for (int trial = 0; trial < 5; ++trial) {
    check(cudaDeviceSynchronize(), "pretrial synchronize");
    auto begin = std::chrono::steady_clock::now();
    check(cudaEventRecord(start), "start record");
    for (int i = 0; i < repeats; ++i) function();
    check(cudaEventRecord(stop), "stop record");
    check(cudaEventSynchronize(stop), "stop synchronize");
    float elapsed = 0;
    check(cudaEventElapsedTime(&elapsed, start, stop), "event elapsed");
    gpu_times.push_back(double(elapsed) / repeats);
    wall_times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count() / repeats);
  }
  check(cudaEventDestroy(start), "start destroy");
  check(cudaEventDestroy(stop), "stop destroy");
  std::sort(gpu_times.begin(), gpu_times.end());
  std::sort(wall_times.begin(), wall_times.end());
  return {gpu_times[2], wall_times[2], gpu_times.front(), gpu_times.back()};
}

void emit(const std::string& name, size_t rank, size_t batch, const timings& time,
          const error_summary& errors, double residual = 0) {
  errors.require();
  if (!std::isfinite(residual) || residual > 1e-10) throw std::runtime_error("LU residual failed");
  std::cout << "{\"operation\":\"" << name << "\",\"rank\":" << rank
            << ",\"batch\":" << batch << ",\"gpu_median_ms\":" << time.gpu_ms
            << ",\"wall_median_ms\":" << time.wall_ms << ",\"gpu_min_ms\":" << time.gpu_min_ms
            << ",\"gpu_max_ms\":" << time.gpu_max_ms << ",\"max_absolute\":" << errors.maximum
            << ",\"relative_frobenius\":" << errors.relative << ",\"residual\":" << residual << "}" << std::endl;
}

template<class Function> void operation(gpu& ops, const std::string& name, size_t r, size_t batch,
                                       const std::vector<matrix>& reference, Function function) {
  gpu::field output = function();
  ops.synchronize();
  auto time = measure([&] { output = function(); });
  auto actual = ops.download(output);
  error_summary errors;
  for (size_t b = 0; b < batch; ++b) errors.compare(actual[b], reference[b]);
  emit(name, r, batch, time, errors);
}

void contraction_orders(size_t r, size_t batch) {
  const size_t Q = 150, n = 26;
  gpu ops(false, 768ul * 1024 * 1024);
  matrix m = values(r, Q, 100);
  std::vector<matrix> response, core, x, g;
  std::vector<matrix> compression_reference, expansion_reference, projection_reference, backprojection_reference;
  for (size_t b = 0; b < batch; ++b) {
    response.push_back(values(r, r, 200 + b)); // Deliberately non-Hermitian.
    core.push_back(values(Q, Q, 300 + b));
    x.push_back(values(r, n, 400 + b));
    g.push_back(values(n, n, 500 + b));
    compression_reference.emplace_back((m.adjoint() * response.back()).eval() * m);
    expansion_reference.emplace_back((m * core.back()).eval() * m.adjoint());
    projection_reference.emplace_back((x.back() * g.back()).eval() * x.back().adjoint());
    backprojection_reference.emplace_back((x.back().adjoint() * response.back()).eval() * x.back());
  }
  auto dm = ops.upload(m), dp = ops.upload(response), dc = ops.upload(core), dx = ops.upload(x), dg = ops.upload(g);
  operation(ops, "compress_left", r, batch, compression_reference, [&] {
    auto first = ops.multiply(dm, dp, 'C', 'N'); return ops.multiply(first, dm);
  });
  operation(ops, "compress_right", r, batch, compression_reference, [&] {
    auto first = ops.multiply(dp, dm); return ops.multiply(dm, first, 'C', 'N');
  });
  operation(ops, "expand_left", r, batch, expansion_reference, [&] {
    auto first = ops.multiply(dm, dc); return ops.multiply(first, dm, 'N', 'C');
  });
  operation(ops, "expand_right", r, batch, expansion_reference, [&] {
    auto first = ops.multiply(dc, dm, 'N', 'C'); return ops.multiply(dm, first);
  });
  operation(ops, "project_left", r, batch, projection_reference, [&] { return ops.project(dx, dg); });
  operation(ops, "project_right", r, batch, projection_reference, [&] {
    auto first = ops.multiply(dg, dx, 'N', 'C'); return ops.multiply(dx, first);
  });
  operation(ops, "backproject_left", r, batch, backprojection_reference, [&] { return ops.backproject(dx, dp); });
  operation(ops, "backproject_right", r, batch, backprojection_reference, [&] {
    auto first = ops.multiply(dp, dx); return ops.multiply(dx, first, 'C', 'N');
  });
}

void single_gemm_dispatch(raw_handles& handles, int rows, int columns, int inner, const std::string& name) {
  matrix a = values(rows, inner, 700), b = values(inner, columns, 701);
  matrix reference = a * b;
  auto pa = pack({a}), pb = pack({b});
  device_buffer<z> da(pa.size()), db(pb.size()), dc(size_t(rows) * columns);
  da.upload(pa.data()); db.upload(pb.data());
  const z alpha = make_cuDoubleComplex(1, 0), beta = make_cuDoubleComplex(0, 0);
  for (bool strided : {true, false}) {
    auto function = [&] {
      if (strided)
        check(cublasZgemmStridedBatched(handles.blas, CUBLAS_OP_N, CUBLAS_OP_N, rows, columns, inner,
              &alpha, da.pointer, rows, size_t(rows) * inner, db.pointer, inner, size_t(inner) * columns,
              &beta, dc.pointer, rows, size_t(rows) * columns, 1), "single strided GEMM");
      else
        check(cublasZgemm(handles.blas, CUBLAS_OP_N, CUBLAS_OP_N, rows, columns, inner,
              &alpha, da.pointer, rows, db.pointer, inner, &beta, dc.pointer, rows), "single ordinary GEMM");
    };
    auto time = measure(function);
    error_summary errors;
    errors.compare(unpack(dc.download(), rows, columns), reference);
    emit(name + (strided ? "_strided" : "_ordinary"), rows, 1, time, errors);
  }
}

void lu_backends(raw_handles& handles, int n) {
  const int batch = 107;
  const size_t stride = size_t(n) * n;
  std::vector<matrix> response, dielectric, reference;
  for (int b = 0; b < batch; ++b) {
    matrix seed = values(n, 12, 900 + b, .03);
    matrix p = -(seed * seed.adjoint()); // I-P is positive definite and safely conditioned.
    matrix a = matrix::Identity(n, n) - p;
    response.push_back(p); dielectric.push_back(a);
    reference.emplace_back(a.partialPivLu().solve(p));
  }
  auto host_a = pack(dielectric), host_b = pack(response);
  device_buffer<z> source_a(host_a.size()), source_b(host_b.size()), lu(host_a.size()), solution(host_b.size());
  source_a.upload(host_a.data()); source_b.upload(host_b.data());
  device_buffer<int> pivots(size_t(n) * batch), statuses(2 * batch);
  device_buffer<z*> aa(batch), bb(batch);
  std::vector<z*> host_aa(batch), host_bb(batch);
  for (int b = 0; b < batch; ++b) { host_aa[b] = lu.pointer + b * stride; host_bb[b] = solution.pointer + b * stride; }
  aa.upload(host_aa.data()); bb.upload(host_bb.data());
  int work_count = 0;
  check(cusolverDnZgetrf_bufferSize(handles.solver, n, n, lu.pointer, n, &work_count), "LU buffer size");
  device_buffer<z> work(std::max(1, work_count));
  for (bool batched : {false, true}) {
    int argument_status = 0;
    auto function = [&] {
      check(cudaMemcpyAsync(lu.pointer, source_a.pointer, lu.count * sizeof(z), cudaMemcpyDeviceToDevice), "reset LU");
      check(cudaMemcpyAsync(solution.pointer, source_b.pointer, solution.count * sizeof(z), cudaMemcpyDeviceToDevice), "reset RHS");
      check(cudaMemsetAsync(statuses.pointer, 0, statuses.count * sizeof(int)), "reset LU status");
      if (batched) {
        check(cublasZgetrfBatched(handles.blas, n, aa.pointer, n, pivots.pointer, statuses.pointer, batch), "batched getrf");
        argument_status = 0;
        check(cublasZgetrsBatched(handles.blas, CUBLAS_OP_N, n, n,
              reinterpret_cast<const z* const*>(aa.pointer), n, pivots.pointer, bb.pointer, n,
              &argument_status, batch), "batched getrs");
        if (argument_status) throw std::runtime_error("batched getrs argument status " + std::to_string(argument_status));
      } else {
        for (int b = 0; b < batch; ++b) {
          check(cusolverDnZgetrf(handles.solver, n, n, lu.pointer + b * stride, n, work.pointer,
                pivots.pointer + b * n, statuses.pointer + b), "loop getrf");
          // Separate solve status preserves factorization singular-status flags.
          check(cusolverDnZgetrs(handles.solver, CUBLAS_OP_N, n, n, lu.pointer + b * stride, n,
                pivots.pointer + b * n, solution.pointer + b * stride, n, statuses.pointer + batch + b), "loop getrs");
        }
      }
    };
    auto time = measure(function, 1);
    for (int status : statuses.download())
      if (status) throw std::runtime_error("factorization/solve status " + std::to_string(status));
    auto output = solution.download();
    error_summary errors;
    double residual = 0;
    for (int b = 0; b < batch; ++b) {
      matrix actual = unpack(output, n, n, b);
      errors.compare(actual, reference[b]);
      double current_residual = (dielectric[b] * actual - response[b]).norm() / std::max(1., response[b].norm());
      if (!std::isfinite(current_residual)) throw std::runtime_error("nonfinite LU residual");
      residual = std::max(residual, current_residual);
    }
    emit(batched ? "lu_batched_with_reset" : "lu_cusolver_loop_with_reset", n, batch, time, errors, residual);
  }
}
} // namespace

int main() {
  try {
    std::cout << std::setprecision(15);
    check(cudaSetDevice(0), "device");
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, 0), "device properties");
    raw_handles handles;
    int blas_version = 0, runtime_version = 0;
    check(cublasGetVersion(handles.blas, &blas_version), "cuBLAS version");
    check(cudaRuntimeGetVersion(&runtime_version), "CUDA runtime version");
    std::cout << "{\"device\":\"" << properties.name << "\",\"cublas_version\":" << blas_version
              << ",\"cuda_runtime_version\":" << runtime_version
              << ",\"precision\":\"complex_double_default_math\",\"warmups\":3,\"trials\":5}" << std::endl;
    for (size_t r : {512ul, 768ul})
      for (size_t batch : {1ul, 4ul}) contraction_orders(r, batch);
    // Exact compact auxiliary IR transform shapes after the contraction reorder.
    single_gemm_dispatch(handles, 150 * 150, 107, 110, "aux_forward");
    single_gemm_dispatch(handles, 150 * 150, 110, 107, "aux_backward");
    for (int n : {64, 150, 192, 256}) lu_backends(handles, n);
    std::cout << "{\"passed\":true}" << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 2;
  }
}
