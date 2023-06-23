#ifndef CUSTOM_PARALLEL_H
#define CUSTOM_PARALLEL_H

#include <cmath>
#include <vector>
#include <thread>

template<class Function>
void custom_parallelize(Function f, size_t n, size_t nthreads) {
    size_t jobs_per_worker = std::ceil(static_cast<double>(n) / nthreads);
    size_t start = 0;
    std::vector<std::thread> jobs;

    for (size_t w = 0; w < nthreads; ++w) {
        size_t end = std::min(n, start + jobs_per_worker);
        if (start >= end) {
            break;
        }
        jobs.emplace_back(f, w, start, end - start);
        start += jobs_per_worker;
    }

    for (auto& job : jobs) {
        job.join();
    }
}

#define SCRAN_CUSTOM_PARALLEL custom_parallelize
#endif
