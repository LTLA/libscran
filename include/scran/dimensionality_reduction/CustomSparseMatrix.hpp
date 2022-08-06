#ifndef SCRAN_CUSTOM_SPARSE_MATRIX_HPP
#define SCRAN_CUSTOM_SPARSE_MATRIX_HPP

#include "../utils/macros.hpp"

#include <vector>
#include "Eigen/Dense"
#include "irlba/parallel.hpp"
#include "pca_utils.hpp"

#ifdef TEST_SCRAN_CUSTOM_SPARSE_MATRIX
#include "Eigen/Sparse"
#endif

namespace scran {

namespace pca_utils {

class CustomSparseMatrix {
public:
    CustomSparseMatrix(size_t nr, size_t nc, int threads) : nrow(nr), ncol(nc), nthreads(threads) {}

    auto rows() const { return nrow; }

    auto cols() const { return ncol; }

private:
    typedef irlba::ParallelSparseMatrix<> InternalMatrix;
    InternalMatrix data;

#ifdef TEST_SCRAN_CUSTOM_SPARSE_MATRIX
public:
    void use_eigen() {
        spmat = Eigen::SparseMatrix<double>(nrow, ncol);
        eigen = true;
        return;
    }

private:
    bool eigen = false;
    Eigen::SparseMatrix<double> spmat;
#endif

public:
    void fill_direct(std::vector<double> x, std::vector<int> i, std::vector<size_t> p) {
#ifdef TEST_SCRAN_CUSTOM_SPARSE_MATRIX
        if (eigen) {
            std::vector<int> column_nonzeros(ncol);
            for (size_t c = 0; c < ncol; ++c) {
                column_nonzeros[c] = p[c+1] - p[c];
            }
            spmat.reserve(column_nonzeros);

            auto xIt = x.begin();
            auto iIt = i.begin();
            for (size_t c = 0; c < ncol; ++c) {
                size_t n = column_nonzeros[c];
                for (size_t i = 0; i < n; ++i, ++xIt, ++iIt) {
                    spmat.insert(*iIt, c) = *xIt;
                }
            }

            spmat.makeCompressed();
            return;
        }
#endif

        data = InternalMatrix(nrow, ncol, std::move(x), std::move(i), std::move(p), nthreads);
    }

public:
    template<class Right>
    void multiply(const Right& rhs, Eigen::VectorXd& output) const {
#ifdef TEST_SCRAN_CUSTOM_SPARSE_MATRIX
        if (eigen) {
            output.noalias() = spmat * rhs;
            return;
        }
#endif

        data.multiply(rhs, output);
        return;
    }

    template<class Right>
    void adjoint_multiply(const Right& rhs, Eigen::VectorXd& output) const {
#ifdef TEST_SCRAN_CUSTOM_SPARSE_MATRIX
        if (eigen) {
            output.noalias() = spmat.adjoint() * rhs;
            return;
        }
#endif

        data.adjoint_multiply(rhs, output);
        return;
    }

    Eigen::MatrixXd realize() const {
#ifdef TEST_SCRAN_CUSTOM_SPARSE_MATRIX
        if (eigen) {
            return Eigen::MatrixXd(spmat);
        }
#endif

        return data.realize();
    }

public:
    void multiply(const Eigen::MatrixXd& rhs, Eigen::MatrixXd& output) const {
#ifdef TEST_SCRAN_CUSTOM_SPARSE_MATRIX
        if (eigen) {
            output.noalias() = spmat * rhs;
            return;
        }
#endif

        size_t nvec = output.cols();
        Eigen::MatrixXd transposed(nvec, nrow); // using a transposed version for more cache efficiency.
        transposed.setZero();

        const auto& x = data.get_values();
        const auto& i = data.get_indices();
        const auto& p = data.get_pointers();

        if (nthreads == 1) {
            Eigen::VectorXd multipliers(nvec);
            for (size_t c = 0; c < ncol; ++c) {
                multipliers.noalias() = rhs.row(c);
                auto start = p[c], end = p[c + 1];
                for (size_t s = start; s < end; ++s) {
                    transposed.col(i[s]).noalias() += x[s] * multipliers;
                }
            }

            output = transposed.adjoint();
            return;
        }


        const auto& row_nonzero_starts = data.get_secondary_nonzero_starts();

#ifndef SCRAN_CUSTOM_PARALLEL
        #pragma omp parallel for num_threads(nthreads)
        for (int t = 0; t < nthreads; ++t) {
#else
        SCRAN_CUSTOM_PARALLEL(nthreads, [&](int first, int last) -> void {
        for (int t = first; t < last; ++t) {
#endif

            const auto& starts = row_nonzero_starts[t];
            const auto& ends = row_nonzero_starts[t + 1];
            Eigen::VectorXd multipliers(nvec);
            for (size_t c = 0; c < ncol; ++c) {
                multipliers.noalias() = rhs.row(c);
                auto start = starts[c], end = ends[c];
                for (size_t s = start; s < end; ++s) {
                    transposed.col(i[s]).noalias() += x[s] * multipliers;
                }
            }

#ifndef SCRAN_CUSTOM_PARALLEL
        }
#else
        }
        }, nthreads);
#endif

        output = transposed.adjoint();
        return;
    }

private:
    size_t nrow, ncol;
    int nthreads;
};

}

}

#endif
