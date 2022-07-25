#include "scran/scran.hpp"
#include <iostream>
#include <random>
#include <chrono>

/**
 * This is a little script for speed testing the PCA implementation with or
 * without Eigen and with or without multiple threads. It can be run with:
 *
 * ./build/gallery/pca_test 4 1000
 *
 * which will time both the custom sparse multiplication and Eigen's version
 * on 4 OpenMP threads, using a seed of 1000 for the randomized sparse matrix. 
 */

int main(int argc, char * argv[]) {
    if (argc < 3) {
        std::cerr << "COMMAND [Threads] [seed]" << std::endl;
        return 1;
    }

    size_t nr = 20000;
    size_t nc = 10000;

    // Parsing the values.
    int nthreads = std::stoi(std::string(argv[1]));
    int seed = std::stoi(std::string(argv[2]));

    // Mock up a sparse matrix.
    std::vector<double> values;
    std::vector<int> indices;
    std::vector<size_t> ptrs(nc + 1);

    std::mt19937_64 rng(seed);
    std::normal_distribution ndist;
    std::uniform_real_distribution udist;

    for (size_t c = 0; c < nc; ++c) {
        for (size_t r = 0; r < nr; ++r) {
            if (udist(rng) < 0.2) {
                values.push_back(ndist(rng));
                indices.push_back(r);
                ++(ptrs[c+1]);
            }
        }
        ptrs[c+1] += ptrs[c];
    }

    tatami::CompressedSparseColumnMatrix<double, int> mat(nr, nc, std::move(values), std::move(indices), std::move(ptrs));

    // Running the PCA, with and without Eigen.
    scran::RunPCA runner;
    runner.set_num_threads(nthreads);

    {
        auto start = std::chrono::high_resolution_clock::now();
        auto outcustom = runner.run(&mat);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Custom: " << duration.count() * 1.0 / 1000.0 << " (first var = " << outcustom.variance_explained[0] << ")" << std::endl;
    }

    runner.set_use_eigen(true);
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto outeigen = runner.run(&mat);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Eigen: " << duration.count() * 1.0 / 1000.0 << " (first var = " << outeigen.variance_explained[0] << ")" << std::endl;
    }

    return 0;
}
