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
  GIT_TAG v1.0.0
)

FetchContent_MakeAvailable(WeightedLowess)

FetchContent_Declare(
  irlba 
  GIT_REPOSITORY https://github.com/LTLA/CppIrlba
  GIT_TAG master
)

FetchContent_MakeAvailable(irlba)

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