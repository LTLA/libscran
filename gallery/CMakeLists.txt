# Compile the minimal workflow. 
add_executable(minimal minimal.cpp)
target_link_libraries(minimal scran)

# Throwing in some optimizations and parallelizations.
target_compile_options(minimal PUBLIC -O2)
find_package(OpenMP)
target_link_libraries(minimal OpenMP::OpenMP_CXX)
