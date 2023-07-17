cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

include(FetchContent)

FetchContent_Declare(
  tatami
  GIT_REPOSITORY https://github.com/LTLA/tatami
  GIT_TAG 09770b07db2c71bde6a515b47748bcf5d3557a4c
)

FetchContent_MakeAvailable(tatami)

FetchContent_Declare(
  WeightedLowess
  GIT_REPOSITORY https://github.com/LTLA/CppWeightedLowess
  GIT_TAG 1f6ca2447fdcd38f4d33b84e1d59364f4cc7b6a6
)

FetchContent_MakeAvailable(WeightedLowess)

FetchContent_Declare(
  irlba 
  GIT_REPOSITORY https://github.com/LTLA/CppIrlba
  GIT_TAG 15d6399fc2c9c2ed4755fa0baef373c086fdedd1
)

FetchContent_MakeAvailable(irlba)

FetchContent_Declare(
  kmeans 
  GIT_REPOSITORY https://github.com/LTLA/CppKmeans
  GIT_TAG 4a6f1257fe5f3c7f1675f8b92251c9950134efea 
)
FetchContent_MakeAvailable(kmeans)

FetchContent_Declare(
  knncolle 
  GIT_REPOSITORY https://github.com/LTLA/knncolle
  GIT_TAG 9e4183326cf92600c18fe2a325054c2315bb5c6d
)

FetchContent_MakeAvailable(knncolle)
