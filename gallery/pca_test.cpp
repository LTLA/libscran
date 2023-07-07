#include "scran/scran.hpp"
#include "Eigen/Sparse"
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

    // Running the PCA using the default irlba::ParallelSparseMatrix.
    {
        scran::SimplePca runner;
        runner.set_num_threads(nthreads);

        auto start = std::chrono::high_resolution_clock::now();
        auto outcustom = runner.run(&mat);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Custom: " << duration.count() * 1.0 / 1000.0 << " (first var = " << outcustom.variance_explained[0] << ")" << std::endl;
    }

    // Doing it with Eigen, including the centering and the extraction/transformation time for consistency.
    {   
        irlba::Irlba runner2;
        runner2.set_number(scran::SimplePca::Defaults::rank);

        auto start = std::chrono::high_resolution_clock::now();
        auto extracted = scran::pca_utils::extract_sparse_for_pca(&mat, nthreads); // row-major extraction.
        auto& ptrs = extracted.ptrs;
        auto& values = extracted.values;
        auto& indices = extracted.indices;

        Eigen::VectorXd center_v(nr), scale_v(nr);
        tatami::parallelize([&](size_t, size_t start, size_t length) -> void {
            size_t ncells = nc;
            for (int r = start, end = start + length; r < end; ++r) {
                auto offset = ptrs[r];

                tatami::SparseRange<double, int> range;
                range.number = ptrs[r + 1] - offset;
                range.value = values.data() + offset;
                range.index = indices.data() + offset;

                auto results = tatami::stats::variances::compute_direct(range, ncells);
                center_v.coeffRef(r) = results.first;
                scale_v.coeffRef(r) = results.second;
            }
        }, nr, nthreads);

        Eigen::SparseMatrix<double> spmat(nc, nr); // transposed (features in the columns).
        {
            std::vector<int> column_nonzeros(nr); // transposed
            for (size_t c = 0; c < nr; ++c) {
                column_nonzeros[c] = ptrs[c+1] - ptrs[c];
            }
            spmat.reserve(column_nonzeros);

            auto xIt = values.begin();
            auto iIt = indices.begin();
            for (size_t c = 0; c < nr; ++c) { // transposed
                size_t n = column_nonzeros[c];
                for (size_t i = 0; i < n; ++i, ++xIt, ++iIt) {
                    spmat.insert(*iIt, c) = *xIt;
                }
            }
            spmat.makeCompressed();
        }

        irlba::EigenThreadScope scope(nthreads);
        irlba::Centered centered(&spmat, &center_v); 
        auto outeigen = runner2.run(centered);

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

        std::cout << "Eigen: " << duration.count() * 1.0 / 1000.0 << " (first D = " << outeigen.D[0] * outeigen.D[0] / (nc - 1) << ")" << std::endl;
    }

    return 0;
}
