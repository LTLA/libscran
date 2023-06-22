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

    src/feature_selection/ModelGeneVar.cpp
    src/feature_selection/FitTrendVar.cpp
    src/feature_selection/ChooseHVGs.cpp

    src/dimensionality_reduction/pca_utils.cpp
    src/dimensionality_reduction/RunPCA.cpp
    src/dimensionality_reduction/BlockedPCA.cpp
    src/dimensionality_reduction/MultiBatchPCA.cpp
    src/dimensionality_reduction/ScaleByNeighbors.cpp

    src/clustering/BuildSNNGraph.cpp
    src/clustering/ClusterSNNGraph.cpp

    #    src/differential_analysis/summarize_comparisons.cpp
    #    src/differential_analysis/auc.cpp
    #    src/differential_analysis/cohens_d.cpp
    #    src/differential_analysis/simple_diff.cpp
    #    src/differential_analysis/ScoreMarkers.cpp
    #    src/differential_analysis/PairwiseEffects.cpp
    #    src/differential_analysis/Factory.cpp

    src/aggregation/AggregateAcrossCells.cpp
    src/aggregation/DownsampleByNeighbors.cpp

    #   src/feature_set_enrichment/ScoreFeatureSet.cpp
    src/feature_set_enrichment/HypergeometricTail.cpp

    src/utils/average_vectors.cpp
    src/utils/subset_vector.cpp
    src/data/data_sparse.cpp
)

target_link_libraries(
    libtest
    gtest_main
    scran
)

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
