cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

include(FetchContent)

FetchContent_Declare(
  tatami
  GIT_REPOSITORY https://github.com/LTLA/tatami
  GIT_TAG 1a97da57ab7874510f9f7feaaf9114921331c62d
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
  GIT_TAG 7ac95138475ea042b087e8fbbfafc006bc6d7c29
)

FetchContent_MakeAvailable(irlba)

FetchContent_Declare(
  kmeans 
  GIT_REPOSITORY https://github.com/LTLA/CppKmeans
  GIT_TAG aed1b7ad1c4eddaf80d851fc24fb81333337bf57
)
FetchContent_MakeAvailable(kmeans)

FetchContent_Declare(
  knncolle 
  GIT_REPOSITORY https://github.com/LTLA/knncolle
  GIT_TAG c5a1776ebf10641d9bf8715e89cb2d965b06e899
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
