include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/afd902e992b720d1b3e106bc5e425a5768872265.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
    libtest 

    src/quality_control/PerCellQcMetrics.cpp
    src/quality_control/PerCellRnaQcMetrics.cpp
    src/quality_control/PerCellAdtQcMetrics.cpp
    src/quality_control/PerCellCrisprQcMetrics.cpp
    src/quality_control/ComputeMedianMad.cpp
    src/quality_control/ChooseOutlierFilters.cpp
    src/quality_control/SuggestRnaQcFilters.cpp
    src/quality_control/SuggestAdtQcFilters.cpp
    src/quality_control/SuggestCrisprQcFilters.cpp
    src/quality_control/FilterCells.cpp

    src/normalization/LogNormCounts.cpp
    src/normalization/CenterSizeFactors.cpp
    src/normalization/MedianSizeFactors.cpp
    src/normalization/GroupedSizeFactors.cpp
    src/normalization/ChoosePseudoCount.cpp

    src/feature_selection/ModelGeneVariances.cpp
    src/feature_selection/FitVarianceTrend.cpp
    src/feature_selection/ChooseHvgs.cpp

    src/dimensionality_reduction/utils.cpp
    src/dimensionality_reduction/convert.cpp
    src/dimensionality_reduction/SimplePca.cpp
    src/dimensionality_reduction/ResidualPca.cpp
    src/dimensionality_reduction/MultiBatchPca.cpp
    src/dimensionality_reduction/ScaleByNeighbors.cpp

    src/clustering/BuildSnnGraph.cpp
    src/clustering/ClusterSnnGraph.cpp

    src/differential_analysis/summarize_comparisons.cpp
    src/differential_analysis/auc.cpp
    src/differential_analysis/cohens_d.cpp
    src/differential_analysis/simple_diff.cpp
    src/differential_analysis/ScoreMarkers.cpp
    src/differential_analysis/PairwiseEffects.cpp
    src/differential_analysis/MatrixCalculator.cpp

    src/aggregation/AggregateAcrossCells.cpp
    src/aggregation/DownsampleByNeighbors.cpp

    src/feature_set_enrichment/ScoreFeatureSet.cpp
    src/feature_set_enrichment/HypergeometricTail.cpp

    src/utils/average_vectors.cpp
    src/utils/subset_vector.cpp
    src/utils/blocking.cpp
    src/data/data_sparse.cpp
)

target_link_libraries(
    libtest
    gtest_main
    scran
)


# Handling the igraph requirements.
set(IGRAPH_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  igraph
  URL https://github.com/igraph/igraph/releases/download/0.10.6/igraph-0.10.6.tar.gz
  URL_HASH MD5=1074f1fde392d1b7924677ad8cfb1dea
)
FetchContent_MakeAvailable(igraph)
target_link_libraries(libtest igraph)

target_compile_definitions(libtest PRIVATE TEST_SCRAN_CUSTOM_SPARSE_MATRIX)

set(USE_OPENMP OFF CACHE BOOL "Compile with OpenMP support")
set(USE_CUSTOM_PARALLEL OFF CACHE BOOL "Compile with custom parallelization")

if (USE_OPENMP)
    find_package(OpenMP)
    target_link_libraries(libtest OpenMP::OpenMP_CXX)
elseif(USE_CUSTOM_PARALLEL)
    target_compile_definitions(libtest PRIVATE TEST_CUSTOM_PARALLEL)
endif()

set(CODE_COVERAGE OFF CACHE BOOL "Enable coverage testing")
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(libtest PRIVATE -O0 -g --coverage)
    target_link_options(libtest PRIVATE --coverage)
endif()

include(GoogleTest)
gtest_discover_tests(libtest)
