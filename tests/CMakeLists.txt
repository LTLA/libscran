include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
    libtest 

    src/quality_control/PerCellRnaQcMetrics.cpp
    src/quality_control/PerCellRnaQcFilters.cpp
    src/quality_control/PerCellAdtQcFilters.cpp
    src/quality_control/IsOutlier.cpp
    src/quality_control/FilterCells.cpp

    src/normalization/LogNormCounts.cpp
    src/normalization/CenterSizeFactors.cpp
    src/normalization/MedianSizeFactors.cpp
    src/normalization/GroupedSizeFactors.cpp
    src/feature_selection/ModelGeneVar.cpp
    src/feature_selection/FitTrendVar.cpp
    src/feature_selection/ChooseHVGs.cpp

    src/dimensionality_reduction/CustomSparseMatrix.cpp
    src/dimensionality_reduction/RunPCA.cpp
    src/dimensionality_reduction/BlockedPCA.cpp
    src/dimensionality_reduction/MultiBatchPCA.cpp
    src/dimensionality_reduction/ScaleByNeighbors.cpp

    src/clustering/BuildSNNGraph.cpp
    src/clustering/ClusterSNNGraph.cpp
    src/differential_analysis/summarize_comparisons.cpp
    src/differential_analysis/auc.cpp
    src/differential_analysis/cohens_d.cpp
    src/differential_analysis/lfc.cpp
    src/differential_analysis/delta_detected.cpp
    src/differential_analysis/ScoreMarkers.cpp
    src/aggregation/AggregateAcrossCells.cpp
    src/utils/average_vectors.cpp
    src/utils/subset_vector.cpp
    src/data/data_sparse.cpp
)

target_link_libraries(
    libtest
    gtest_main
    scran
)

set(USE_OPENMP OFF CACHE BOOL "Compile with OpenMP support")
if (USE_OPENMP)
    find_package(OpenMP)
    target_link_libraries(libtest OpenMP::OpenMP_CXX)
endif()

set(CODE_COVERAGE OFF CACHE BOOL "Enable coverage testing")
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(libtest PRIVATE -O0 -g --coverage)
    target_link_options(libtest PRIVATE --coverage)
endif()

# Check that the custom parallelization schemes are properly set up.
add_executable(
    custom_parallel
    src/clustering/BuildSNNGraph.cpp
    src/differential_analysis/summarize_comparisons.cpp
    src/dimensionality_reduction/ScaleByNeighbors.cpp
)

set_target_properties(custom_parallel PROPERTIES COMPILE_DEFINITIONS "TEST_SCRAN_CUSTOM_PARALLEL")

target_link_libraries(
    custom_parallel
    gtest_main
    scran
)

include(GoogleTest)
gtest_discover_tests(libtest)
gtest_discover_tests(custom_parallel)
