# Compile the minimal workflow. 
add_executable(minimal minimal.cpp)
target_link_libraries(minimal scran)

include(FetchContent)
FetchContent_Declare(
  tatami_mtx
  GIT_REPOSITORY https://github.com/tatami-inc/tatami_mtx
  GIT_TAG ecc2654dce7bbb87fbaea60efe9cbe57a7972d42
)
FetchContent_MakeAvailable(tatami_mtx)
target_link_libraries(minimal tatami_mtx)

target_compile_options(minimal PUBLIC -O2)
find_package(OpenMP)
target_link_libraries(minimal OpenMP::OpenMP_CXX)

# Compile the PCA speed test.
add_executable(pca_test pca_test.cpp)
target_link_libraries(pca_test scran)
target_compile_definitions(pca_test PRIVATE TEST_SCRAN_CUSTOM_SPARSE_MATRIX NDEBUG)

target_compile_options(pca_test PUBLIC -O2)
find_package(OpenMP)
target_link_libraries(pca_test OpenMP::OpenMP_CXX)
