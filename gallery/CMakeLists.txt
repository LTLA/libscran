# Compile the minimal workflow. 
add_executable(minimal minimal.cpp)
target_link_libraries(minimal scran)

# Throwing in some optimizations and parallelizations.
target_compile_options(minimal PUBLIC -O2)
find_package(OpenMP)
target_link_libraries(minimal OpenMP::OpenMP_CXX)

# Compile the PCA speed test.
add_executable(pca_test pca_test.cpp)
target_link_libraries(pca_test scran)
target_compile_definitions(pca_test PRIVATE TEST_SCRAN_CUSTOM_SPARSE_MATRIX NDEBUG)

# Throwing in some optimizations and parallelizations.
target_compile_options(pca_test PUBLIC -O2)
find_package(OpenMP)
target_link_libraries(pca_test OpenMP::OpenMP_CXX)
