cmake_minimum_required(VERSION 3.14)

project(scran
    VERSION 1.0.0
    DESCRIPTION "C++ methods for basic single-cell analyses"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_library(scran INTERFACE)

target_include_directories(scran INTERFACE include/)

add_subdirectory(extern)

target_link_libraries(scran INTERFACE tatami WeightedLowess irlba)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
    if(BUILD_TESTING)
        add_subdirectory(tests)
    endif()
endif()
