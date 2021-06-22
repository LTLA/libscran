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
  GIT_TAG master
)

FetchContent_MakeAvailable(WeightedLowess)
