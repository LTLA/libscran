cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

include(FetchContent)

FetchContent_Declare(
  tatami
  GIT_REPOSITORY https://github.com/LTLA/tatami
  GIT_TAG master
)

FetchContent_MakeAvailable(tatami)

FetchContent_Declare(
  WeightedLowess
  GIT_REPOSITORY https://github.com/LTLA/CppWeightedLowess
  GIT_TAG b9cf104772d899d318cc90a1e6913be353fc9a60
)

FetchContent_MakeAvailable(WeightedLowess)

FetchContent_Declare(
  irlba 
  GIT_REPOSITORY https://github.com/LTLA/CppIrlba
  GIT_TAG master
)

FetchContent_MakeAvailable(irlba)

FetchContent_Declare(
  kmeans 
  GIT_REPOSITORY https://github.com/LTLA/CppKmeans
  GIT_TAG 3d5f21df81014e23f83301132a15abf662ad2536
)
FetchContent_MakeAvailable(kmeans)

FetchContent_Declare(
  knncolle 
  GIT_REPOSITORY https://github.com/LTLA/knncolle
  GIT_TAG master
)

FetchContent_MakeAvailable(knncolle)

if(NOT SCRAN_EXCLUDE_IGRAPH)
    FetchContent_Declare(
      igraph
      URL https://github.com/igraph/igraph/releases/download/0.9.4/igraph-0.9.4.tar.gz
      URL_HASH MD5=ea8d7791579cfbc590060570e0597f6b
    )

    FetchContent_MakeAvailable(igraph)
endif()
