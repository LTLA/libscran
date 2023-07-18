#ifndef SCRAN_MACROS_HPP
#define SCRAN_MACROS_HPP

/**
 * @file macros.hpp
 *
 * @brief Set common macros used through **libscran**.
 *
 * @details
 * The `SCRAN_CUSTOM_PARALLEL` macro can be set to a function that specifies a custom parallelization scheme.
 * This function should be a template that accept three arguments:
 *
 * - `njobs`, an integer specifying the number of jobs.
 * - `fun`, a lambda that accepts two arguments, `start` and `end`.
 * - `nthreads`, an integer specifying the number of threads to use.
 *
 * The function should split `[0, njobs)` into any number of contiguous, non-overlapping intervals, and call `fun` on each interval, possibly in different threads.
 * The details of the splitting and evaluation are left to the discretion of the developer defining the macro. 
 * The function should only return once all evaluations of `fun` are complete.
 *
 * If `SCRAN_CUSTOM_PARALLEL` is set, the following macros are also set (if they are not already defined):
 *
 * - `TATAMI_CUSTOM_PARALLEL`, from the [**tatami**](https://ltla.github.io/tatami) library.
 * - `IRLBA_CUSTOM_PARALLEL`, from the [**irlba**](https://ltla.github.io/CppIrlba) library.
 *
 * This ensures that any custom parallelization scheme is propagated to all of **libscran**'s dependencies.
 * If these libraries are used outside of **libscran**, some care is required to ensure that the macros are consistently defined through the client library/application;
 * otherwise, developers may observe ODR compilation errors. 
 */

// Synchronizing all parallelization schemes.
#ifdef SCRAN_CUSTOM_PARALLEL

#ifndef TATAMI_CUSTOM_PARALLEL
#define TATAMI_CUSTOM_PARALLEL SCRAN_CUSTOM_PARALLEL
#endif

#ifndef IRLBA_CUSTOM_PARALLEL
namespace scran {

template<class Function>
void irlba_parallelize_(int nthreads, Function fun) {
    SCRAN_CUSTOM_PARALLEL([&](size_t, size_t f, size_t l) -> void {
        // This loop should be trivial if f + 1== l when nthreads == njobs.
        // Nonetheless, we still have a loop just in case the arbitrary
        // scheduling does wacky things. 
        for (size_t i = 0; i < l; ++i) {
            fun(f + i);
        }
    }, nthreads, nthreads);
}

}

#define IRLBA_CUSTOM_PARALLEL scran::irlba_parallelize_
#endif

#endif

#endif
