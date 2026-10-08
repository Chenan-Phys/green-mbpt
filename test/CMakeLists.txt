project(_test)

if(CUSTOM_KERNELS)
    add_executable(thc_reader_probe thc_reader_probe.cpp)
    target_link_libraries(thc_reader_probe PRIVATE GREEN::MBPT)
endif()
add_executable(thc_frozen_probe thc_frozen_probe.cpp)
target_link_libraries(thc_frozen_probe PRIVATE GREEN::MBPT)
add_executable(thc_fft_probe thc_fft_probe.cpp)
target_link_libraries(thc_fft_probe PRIVATE GREEN::MBPT)
add_test(NAME THC_full_mesh_complex_FFT COMMAND thc_fft_probe)
if(TARGET GREEN::GPU)
    find_package(CUDAToolkit REQUIRED)
    add_executable(thc_cuda_probe thc_cuda_probe.cpp)
    target_link_libraries(thc_cuda_probe PRIVATE GREEN::MBPT GREEN::GPU)
    add_executable(thc_cuda_order_probe thc_cuda_order_probe.cpp)
    target_link_libraries(thc_cuda_order_probe PRIVATE GREEN::MBPT GREEN::GPU CUDA::cublas CUDA::cusolver CUDA::cudart)
    add_executable(thc_cuda_gemm3m_probe thc_cuda_gemm3m_probe.cpp)
    target_link_libraries(thc_cuda_gemm3m_probe PRIVATE GREEN::MBPT GREEN::GPU CUDA::cublas CUDA::cudart)
    option(GREEN_THC_RUN_GPU_TESTS "Register device-dependent THC tests on a CUDA workstation" OFF)
    if(GREEN_THC_RUN_GPU_TESTS)
        add_test(NAME THC_resident_complex_CUDA COMMAND thc_cuda_probe)
    endif()
endif()
add_test(NAME THC_coordinated_GPU_dependency COMMAND ${CMAKE_COMMAND}
    -DGREEN_SOURCE_DIR=${CMAKE_SOURCE_DIR} -DGREEN_TEST_BINARY_DIR=${CMAKE_BINARY_DIR}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/thc_cmake_kernel_test.cmake)

Include(FetchContent)

FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0 # or a later release_
)

FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

add_executable(solvers_test solvers_test.cpp)
target_compile_definitions(solvers_test PRIVATE TEST_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data"
        GRID_PATH="${grids_SOURCE_DIR}/data")
target_link_libraries(solvers_test
        PRIVATE
        Catch2::Catch2
        GREEN::MBPT GREEN::SC GREEN::IMPURITY BLAS::BLAS transform)

add_executable(decomp_test decomp_test.cpp)
target_compile_definitions(decomp_test PRIVATE TEST_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data"
        GRID_PATH="${grids_SOURCE_DIR}/data")
target_link_libraries(decomp_test
        PRIVATE
        Catch2::Catch2WithMain
        GREEN::MBPT GREEN::SC BLAS::BLAS transform)

include(CTest)
include(Catch)
catch_discover_tests(solvers_test)
catch_discover_tests(decomp_test)
