// Optional workstation-only experiment. No production backend is changed.
// cuBLAS 13.2 documents Zgemm3m, but only Cgemm3mStridedBatched.
// Consequently both ordinary Zgemm and Zgemm3m are called in batch loops;
// the current strided-batched Zgemm schedule is also measured.
#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using matrix = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using z = cuDoubleComplex;

void check(int status, const char* operation) {
  if (status) throw std::runtime_error(std::string(operation) + " status " + std::to_string(status));
}

template<class T> struct device_buffer {
  T* data = nullptr;
  size_t count;
  explicit device_buffer(size_t size) : count(size) {
    check(cudaMalloc(reinterpret_cast<void**>(&data), size * sizeof(T)), "cudaMalloc");
  }
  ~device_buffer() { if (data) cudaFree(data); }
  device_buffer(const device_buffer&) = delete;
  device_buffer& operator=(const device_buffer&) = delete;
  void upload(const std::vector<T>& input) {
    if (input.size() != count) throw std::runtime_error("upload extent mismatch");
    check(cudaMemcpy(data, input.data(), count * sizeof(T), cudaMemcpyHostToDevice), "upload");
  }
  std::vector<T> download() const {
    std::vector<T> output(count);
    check(cudaMemcpy(output.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return output;
  }
};

struct handles {
  cublasHandle_t blas = nullptr;
  handles() {
    check(cublasCreate(&blas), "cublasCreate");
    check(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH), "default math");
    check(cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_HOST), "host pointer mode");
  }
  ~handles() {
    cudaDeviceSynchronize();
    if (blas) cublasDestroy(blas);
  }
};

matrix values(size_t rows, size_t cols, unsigned seed, bool cancellation = false) {
  std::mt19937 generator(seed);
  std::uniform_real_distribution<double> uniform(-.02, .02);
  matrix output(rows, cols);
  for (Eigen::Index j = 0; j < output.cols(); ++j)
    for (Eigen::Index i = 0; i < output.rows(); ++i) {
      const double real = uniform(generator);
      output(i, j) = {real, cancellation ? real * (1. + 1e-10) : uniform(generator)};
    }
  return output;
}

std::vector<z> pack(const std::vector<matrix>& input) {
  std::vector<z> output;
  for (const auto& value : input)
    for (Eigen::Index i = 0; i < value.size(); ++i)
      output.push_back(make_cuDoubleComplex(value.data()[i].real(), value.data()[i].imag()));
  return output;
}

matrix unpack(const std::vector<z>& input, size_t rows, size_t cols, size_t batch) {
  matrix output(rows, cols);
  for (size_t j = 0; j < cols; ++j)
    for (size_t i = 0; i < rows; ++i) {
      const z value = input[batch * rows * cols + j * rows + i];
      output(i, j) = {value.x, value.y};
    }
  return output;
}

struct error_summary {
  double maximum = 0, relative = 0;
  bool finite = true;
  void compare(const matrix& actual, const matrix& reference) {
    if (actual.rows() != reference.rows() || actual.cols() != reference.cols())
      throw std::runtime_error("oracle dimensions differ");
    finite = finite && actual.allFinite() && reference.allFinite();
    if (!finite) return;
    maximum = std::max(maximum, (actual - reference).cwiseAbs().maxCoeff());
    relative = std::max(relative, (actual - reference).norm() / std::max(1e-30, reference.norm()));
  }
  bool passed() const {
    return finite && std::isfinite(maximum) && std::isfinite(relative) && maximum <= 1e-10 && relative <= 1e-11;
  }
};

struct timings { double gpu_ms, wall_ms, gpu_min_ms, gpu_max_ms; };

template<class Function> timings measure(Function function) {
  constexpr int warmups = 3, trials = 5, repeats = 3;
  for (int i = 0; i < warmups; ++i) function();
  check(cudaDeviceSynchronize(), "warmup synchronize");
  cudaEvent_t start = nullptr, stop = nullptr;
  check(cudaEventCreate(&start), "start event");
  check(cudaEventCreate(&stop), "stop event");
  std::vector<double> gpu_times, wall_times;
  for (int trial = 0; trial < trials; ++trial) {
    check(cudaDeviceSynchronize(), "pretrial synchronize");
    const auto begin = std::chrono::steady_clock::now();
    check(cudaEventRecord(start), "start record");
    for (int i = 0; i < repeats; ++i) function();
    check(cudaEventRecord(stop), "stop record");
    check(cudaEventSynchronize(stop), "stop synchronize");
    float elapsed = 0;
    check(cudaEventElapsedTime(&elapsed, start, stop), "elapsed");
    gpu_times.push_back(double(elapsed) / repeats);
    wall_times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count() / repeats);
  }
  check(cudaEventDestroy(start), "start destroy");
  check(cudaEventDestroy(stop), "stop destroy");
  std::sort(gpu_times.begin(), gpu_times.end());
  std::sort(wall_times.begin(), wall_times.end());
  return {gpu_times[2], wall_times[2], gpu_times.front(), gpu_times.back()};
}

enum class backend { strided, ordinary, gauss3m };
const char* backend_name(backend value) {
  if (value == backend::strided) return "zgemm_strided_batched";
  if (value == backend::ordinary) return "zgemm_batch_loop";
  return "zgemm3m_batch_loop";
}

// All strides are in complex elements, not bytes. M is independently allocated
// for every batch item: no unsupported zero-stride broadcasting is assumed.
void multiply(handles& h, backend which, cublasOperation_t transa, cublasOperation_t transb,
              int rows, int cols, int inner, const z* a, int lda, long long stride_a,
              const z* b, int ldb, long long stride_b, z* c, int ldc, long long stride_c, int batch) {
  const z alpha = make_cuDoubleComplex(1, 0), beta = make_cuDoubleComplex(0, 0);
  if (which == backend::strided) {
    check(cublasZgemmStridedBatched(h.blas, transa, transb, rows, cols, inner,
          &alpha, a, lda, stride_a, b, ldb, stride_b, &beta, c, ldc, stride_c, batch), "strided GEMM");
    return;
  }
  for (int item = 0; item < batch; ++item) {
    const z* ai = a + item * stride_a;
    const z* bi = b + item * stride_b;
    z* ci = c + item * stride_c;
    if (which == backend::ordinary)
      check(cublasZgemm(h.blas, transa, transb, rows, cols, inner,
            &alpha, ai, lda, bi, ldb, &beta, ci, ldc), "ordinary GEMM");
    else
      check(cublasZgemm3m(h.blas, transa, transb, rows, cols, inner,
            &alpha, ai, lda, bi, ldb, &beta, ci, ldc), "Gauss 3m GEMM");
  }
}

bool contraction_case(handles& h, int rank, int batch, const std::string& variety) {
  constexpr int Q = 150;
  const long long point_stride = static_cast<long long>(rank) * rank;
  const long long core_stride = static_cast<long long>(Q) * Q;
  const long long intermediate_stride = static_cast<long long>(rank) * Q;
  std::vector<matrix> factors, responses, cores, compression_reference, expansion_reference;
  for (int item = 0; item < batch; ++item) {
    const bool cancellation = variety == "cancellation";
    factors.emplace_back(values(rank, Q, 110 + item, cancellation));
    matrix response = values(rank, rank, 210 + item, cancellation);
    matrix core = values(Q, Q, 310 + item, cancellation);
    if (variety == "hermitian") {
      response = ((response + response.adjoint()) * .5).eval();
      core = ((core + core.adjoint()) * .5).eval();
    }
    responses.emplace_back(std::move(response));
    cores.emplace_back(std::move(core));
    // Eigen CPU oracles are evaluated before any GPU timing. The equivalent
    // right-associated schedule is checked against this independent reference.
    compression_reference.emplace_back((factors.back().adjoint() * responses.back()).eval() * factors.back());
    expansion_reference.emplace_back((factors.back() * cores.back()).eval() * factors.back().adjoint());
  }
  device_buffer<z> dm(size_t(intermediate_stride) * batch), dr(size_t(point_stride) * batch),
                   dc(size_t(core_stride) * batch), work(size_t(intermediate_stride) * batch),
                   compressed(size_t(core_stride) * batch), expanded(size_t(point_stride) * batch);
  dm.upload(pack(factors)); dr.upload(pack(responses)); dc.upload(pack(cores));
  bool all_passed = true;
  for (const std::string operation : {"compress_left", "compress_right", "expand_left", "expand_right"}) {
    const bool compression = operation.find("compress") == 0;
    const bool left = operation.find("left") != std::string::npos;
    for (backend which : {backend::strided, backend::ordinary, backend::gauss3m}) {
      auto function = [&] {
        if (compression && left) {
          multiply(h, which, CUBLAS_OP_C, CUBLAS_OP_N, Q, rank, rank,
                   dm.data, rank, intermediate_stride, dr.data, rank, point_stride,
                   work.data, Q, intermediate_stride, batch);
          multiply(h, which, CUBLAS_OP_N, CUBLAS_OP_N, Q, Q, rank,
                   work.data, Q, intermediate_stride, dm.data, rank, intermediate_stride,
                   compressed.data, Q, core_stride, batch);
        } else if (compression) {
          multiply(h, which, CUBLAS_OP_N, CUBLAS_OP_N, rank, Q, rank,
                   dr.data, rank, point_stride, dm.data, rank, intermediate_stride,
                   work.data, rank, intermediate_stride, batch);
          multiply(h, which, CUBLAS_OP_C, CUBLAS_OP_N, Q, Q, rank,
                   dm.data, rank, intermediate_stride, work.data, rank, intermediate_stride,
                   compressed.data, Q, core_stride, batch);
        } else if (left) {
          multiply(h, which, CUBLAS_OP_N, CUBLAS_OP_N, rank, Q, Q,
                   dm.data, rank, intermediate_stride, dc.data, Q, core_stride,
                   work.data, rank, intermediate_stride, batch);
          multiply(h, which, CUBLAS_OP_N, CUBLAS_OP_C, rank, rank, Q,
                   work.data, rank, intermediate_stride, dm.data, rank, intermediate_stride,
                   expanded.data, rank, point_stride, batch);
        } else {
          multiply(h, which, CUBLAS_OP_N, CUBLAS_OP_C, Q, rank, Q,
                   dc.data, Q, core_stride, dm.data, rank, intermediate_stride,
                   work.data, Q, intermediate_stride, batch);
          multiply(h, which, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, Q,
                   dm.data, rank, intermediate_stride, work.data, Q, intermediate_stride,
                   expanded.data, rank, point_stride, batch);
        }
      };
      const auto time = measure(function);
      const auto actual = compression ? compressed.download() : expanded.download();
      const auto& reference = compression ? compression_reference : expansion_reference;
      error_summary errors;
      for (int item = 0; item < batch; ++item)
        errors.compare(unpack(actual, compression ? Q : rank, compression ? Q : rank, item), reference[item]);
      all_passed = all_passed && errors.passed();
      std::cout << "{\"operation\":\"" << operation << "\",\"backend\":\"" << backend_name(which)
                << "\",\"variety\":\"" << variety << "\",\"rank\":" << rank << ",\"auxiliary_rank\":" << Q
                << ",\"batch\":" << batch << ",\"gpu_median_ms\":" << time.gpu_ms
                << ",\"wall_median_ms\":" << time.wall_ms << ",\"gpu_min_ms\":" << time.gpu_min_ms
                << ",\"gpu_max_ms\":" << time.gpu_max_ms << ",\"max_absolute\":" << errors.maximum
                << ",\"relative_frobenius\":" << errors.relative << ",\"finite\":" << (errors.finite ? "true" : "false")
                << ",\"passed\":" << (errors.passed() ? "true" : "false") << "}" << std::endl;
    }
  }
  return all_passed;
}
} // namespace

int main() {
  try {
    std::cout << std::setprecision(15);
    check(cudaSetDevice(0), "device");
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, 0), "device properties");
    handles h;
    int blas_version = 0, runtime_version = 0;
    check(cublasGetVersion(h.blas, &blas_version), "cuBLAS version");
    check(cudaRuntimeGetVersion(&runtime_version), "CUDA runtime version");
    if (properties.major < 5) throw std::runtime_error("gemm3m requires compute capability >= 5.0");
    std::cout << "{\"device\":\"" << properties.name << "\",\"cublas_version\":" << blas_version
              << ",\"cuda_runtime_version\":" << runtime_version
              << ",\"precision\":\"complex_double_default_math\",\"warmups\":3,\"trials\":5,\"repeats\":3"
              << ",\"timing_excludes\":\"uploads_downloads_allocations_cpu_oracle\",\"max_absolute_tolerance\":1e-10"
              << ",\"relative_frobenius_tolerance\":1e-11}" << std::endl;
    bool passed = true;
    for (int rank : {512, 768})
      for (int batch : {1, 4})
        for (const std::string variety : {"general", "hermitian", "cancellation"})
          passed = contraction_case(h, rank, batch, variety) && passed;
    std::cout << "{\"passed\":" << (passed ? "true" : "false") << "}" << std::endl;
    return passed ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 2;
  }
}
