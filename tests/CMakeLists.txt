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
    src/quality_control/PerCellQCMetrics.cpp
    src/quality_control/PerCellQCFilters.cpp
    src/quality_control/IsOutlier.cpp
    src/quality_control/FilterCells.cpp
    src/normalization/LogNormCounts.cpp
    src/feature_selection/ModelGeneVar.cpp
    src/feature_selection/FitTrendVar.cpp
    src/dimensionality_reduction/RunPCA.cpp
    src/clustering/BuildSNNGraph.cpp
    src/clustering/ClusterSNNGraph.cpp
    src/differential_analysis/CohensD.cpp
    src/data/data_sparse.cpp
)

target_link_libraries(
    libtest
    gtest_main
    scran
)

include(GoogleTest)
gtest_discover_tests(libtest)

add_test(NAME scran_tests COMMAND libtest)
