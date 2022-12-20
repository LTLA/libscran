cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

include(FetchContent)

FetchContent_Declare(
  tatami
  GIT_REPOSITORY https://github.com/LTLA/tatami
  GIT_TAG 52a4f542a93704f3f236cba9e592b9997a97cf93 
)

FetchContent_MakeAvailable(tatami)

FetchContent_Declare(
  WeightedLowess
  GIT_REPOSITORY https://github.com/LTLA/CppWeightedLowess
  GIT_TAG 100007eacb1d6359bfd22bdb0be470de6f5c06a4
)

FetchContent_MakeAvailable(WeightedLowess)

FetchContent_Declare(
  irlba 
  GIT_REPOSITORY https://github.com/LTLA/CppIrlba
  GIT_TAG 0afbe46f7f37f67a4797c9025696d19dcdf13f29
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

if(NOT SCRAN_EXCLUDE_IGRAPH)
    # Disable the errors to future-proof against later (more stringent) compiler versions.
    # At this point, we don't really have a choice, so we might as well let it pass.
    set(IGRAPH_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
      igraph
      URL https://github.com/igraph/igraph/releases/download/0.9.4/igraph-0.9.4.tar.gz
      URL_HASH MD5=ea8d7791579cfbc590060570e0597f6b
    )

    FetchContent_MakeAvailable(igraph)
endif()
