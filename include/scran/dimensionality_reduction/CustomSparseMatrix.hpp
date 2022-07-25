#ifndef SCRAN_CUSTOM_SPARSE_MATRIX_HPP
#define SCRAN_CUSTOM_SPARSE_MATRIX_HPP

#include <vector>
#include "Eigen/Dense"
#include "irlba/wrappers.hpp"
#include "pca_utils.hpp"

#ifdef TEST_SCRAN_CUSTOM_SPARSE_MATRIX
#include "Eigen/Sparse"
#endif

namespace scran {

namespace pca_utils {

class CustomSparseMatrix {
public:
    CustomSparseMatrix(size_t nr, size_t nc, int threads) : nrow(nr), ncol(nc), nthreads(threads), p(nc + 1) {}

    auto rows() const { return nrow; }

    auto cols() const { return ncol; }

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
    void fill_columns(const std::vector<std::vector<double> >& values, const std::vector<std::vector<int> >& indices) {
#ifdef TEST_SCRAN_CUSTOM_SPARSE_MATRIX
        if (eigen) {
            std::vector<int> nnzeros;
            nnzeros.reserve(values.size());
            for (const auto& v : values) {
                nnzeros.push_back(v.size());
            }

            spmat.reserve(nnzeros);
            for (size_t z = 0; z < values.size(); ++z) {
                const auto& curi = indices[z];
                const auto& curv = values[z];
                for (size_t i = 0; i < curi.size(); ++i) {
                    spmat.insert(curi[i], z) = curv[i];
                }
            }
            spmat.makeCompressed();
            return;
        }
#endif

        size_t nnzeros = 0;
        auto pIt = p.begin() + 1;
        for (const auto& v : values) {
            nnzeros += v.size();
            *pIt = nnzeros;
            ++pIt;
        }

        x.reserve(nnzeros);
        for (const auto& v : values) {
            x.insert(x.end(), v.begin(), v.end());
        }

        i.reserve(nnzeros);
        for (const auto& v : indices) {
            i.insert(i.end(), v.begin(), v.end());
        }

        if (nthreads > 1) {
            fragment_threads();
        }
    }

    void fill_rows(const std::vector<std::vector<double> >& values, const std::vector<std::vector<int> >& indices, const std::vector<int>& column_nonzeros) {
#ifdef TEST_SCRAN_CUSTOM_SPARSE_MATRIX
        if (eigen) {
            spmat.reserve(column_nonzeros);
            for (size_t z = 0; z < values.size(); ++z) {
                const auto& curi = indices[z];
                const auto& curv = values[z];
                for (size_t i = 0; i < curi.size(); ++i) {
                    spmat.insert(z, curi[i]) = curv[i];
                }
            }
            spmat.makeCompressed();
            return;
        }
#endif

        size_t nnzeros = 0;
        auto pIt = p.begin() + 1;
        for (auto nz : column_nonzeros) {
            *pIt = nnzeros;
            ++pIt;
            nnzeros += nz;
        }

        x.resize(nnzeros);
        i.resize(nnzeros);

        for (size_t j = 0, endj = values.size(); j < endj; ++j) {
            const auto& curv = values[j];
            const auto& curi = indices[j];

            for (size_t k = 0, endk = curv.size(); k < endk; ++k) {
                auto& pos = p[curi[k] + 1];
                x[pos] = curv[k];
                i[pos] = j;
                ++pos;
            }
        }

        if (nthreads > 1) {
            fragment_threads();
        }
    }

private:
    std::vector<size_t> col_starts, col_ends;
    std::vector<std::vector<size_t> > row_nonzero_starts;

    void fragment_threads() {
        double total_nzeros = p.back();
        size_t per_thread = std::ceil(total_nzeros / nthreads);
        
        // Splitting columns across threads so each thread processes the same number of nonzero elements.
        col_starts.resize(nthreads);
        col_ends.resize(nthreads);
        {
            size_t col_counter = 0;
            size_t sofar = per_thread;
            for (int t = 0; t < nthreads; ++t) {
                col_starts[t] = col_counter;
                while (col_counter < ncol && p[col_counter + 1] <= sofar) {
                    ++col_counter;
                }
                col_ends[t] = col_counter;
                sofar += per_thread;
            }
        }

        // Splitting rows across threads so each thread processes the same number of nonzero elements.
        row_nonzero_starts.resize(nthreads + 1, std::vector<size_t>(ncol));        
        {
            std::vector<size_t> row_nonzeros(nrow);
            for (auto i_ : i) {
                ++(row_nonzeros[i_]);
            }
            
            std::vector<size_t> row_starts(nthreads), row_ends(nthreads);
            size_t row_counter = 0;
            size_t sofar = per_thread;
            size_t cum_rows = 0;

            for (int t = 0; t < nthreads; ++t) {
                row_starts[t] = row_counter;
                while (row_counter < nrow && cum_rows <= sofar) {
                    cum_rows += row_nonzeros[row_counter];
                    ++row_counter;
                }
                row_ends[t] = row_counter;
                sofar += per_thread;
            }

            for (size_t c = 0; c < ncol; ++c) {
                size_t col_start = p[c], col_end = p[c + 1];
                row_nonzero_starts[0][c] = col_start;

                size_t s = col_start;
                for (int thread = 0; thread < nthreads; ++thread) {
                    while (s < col_end && i[s] < row_ends[thread]) { 
                        ++s; 
                    }
                    row_nonzero_starts[thread + 1][c] = s;
                }
            }
        }
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

        output.setZero();

        if (nthreads == 1) {
            for (size_t c = 0; c < ncol; ++c) {
                column_sum_product(p[c], p[c + 1], rhs.coeff(c), output); 
            }
            return;
        }

#ifndef SCRAN_CUSTOM_PARALLEL
        #pragma omp parallel for num_threads(nthreads);
        for (int t = 0; t < nthreads; ++t) {
#else
        SCRAN_CUSTOM_PARALLEL(nthreads, [&](int first, int last) -> void {
        for (int t = first; t < last; ++t) {
#endif

            const auto& starts = row_nonzero_starts[t];
            const auto& ends = row_nonzero_starts[t + 1];
            for (size_t c = 0; c < ncol; ++c) {
                column_sum_product(starts[c], ends[c], rhs.coeff(c), output);
            }

#ifndef SCRAN_CUSTOM_PARALLEL
        }
#else
        }
        }, nthreads);
#endif

        return;
    }

public:
    template<class Right>
    void adjoint_multiply(const Right& rhs, Eigen::VectorXd& output) const {
#ifdef TEST_SCRAN_CUSTOM_SPARSE_MATRIX
        if (eigen) {
            output.noalias() = spmat.adjoint() * rhs;
            return;
        }
#endif

        if (nthreads == 1) {
            for (size_t c = 0; c < ncol; ++c) {
                output.coeffRef(c) = column_dot_product(c, rhs);
            }
            return;
        }

#ifndef SCRAN_CUSTOM_PARALLEL
        #pragma omp parallel for num_threads(nthreads);
        for (int t = 0; t < nthreads; ++t) {
#else
        SCRAN_CUSTOM_PARALLEL(nthreads, [&](int first, int last) -> void {
        for (int t = first; t < last; ++t) {
#endif

            auto curstart = col_starts[t];
            auto curend = col_ends[t];
            for (size_t c = curstart; c < curend; ++c) {
                output.coeffRef(c) = column_dot_product(c, rhs);
            }

#ifndef SCRAN_CUSTOM_PARALLEL
        }
#else
        }
        }, nthreads);
#endif

        return;
    }

private:
    template<class Right>
    double column_dot_product(size_t c, const Right& rhs) const {
        size_t col_start = p[c], col_end = p[c + 1];
        double dot = 0;
        for (size_t s = col_start; s < col_end; ++s) {
            dot += x[s] * rhs.coeff(i[s]);
        }
        return dot;
    }

    void column_sum_product(size_t start, size_t end, double val, Eigen::VectorXd& output) const {
        for (size_t s = start; s < end; ++s) {
            output.coeffRef(i[s]) += x[s] * val;
        }
    }

    void column_sum_product(size_t start, size_t end, const Eigen::VectorXd& rhs, Eigen::MatrixXd& transposed) const {
        Eigen::Index nc = rhs.size();
        for (size_t s = start; s < end; ++s) {
            transposed.col(i[s]).noalias() += x[s] * rhs;
        }
    }

public:
    Eigen::MatrixXd realize() const {
#ifdef TEST_SCRAN_CUSTOM_SPARSE_MATRIX
        if (eigen) {
            return Eigen::MatrixXd(spmat);
        }
#endif

        Eigen::MatrixXd output(nrow, ncol);
        output.setZero();

        for (size_t c = 0; c < ncol; ++c) {
            size_t col_start = p[c], col_end = p[c + 1];
            for (size_t s = col_start; s < col_end; ++s) {
                output.coeffRef(i[s], c) = x[s];
            }
        }

        return output;
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

        if (nthreads == 1) {
            Eigen::VectorXd multipliers(nvec);
            for (size_t c = 0; c < ncol; ++c) {
                multipliers.noalias() = rhs.row(c);
                column_sum_product(p[c], p[c + 1], multipliers, transposed);
            }

            output = transposed.adjoint();
            return;
        }

#ifndef SCRAN_CUSTOM_PARALLEL
        #pragma omp parallel for num_threads(nthreads);
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
                column_sum_product(starts[c], ends[c], multipliers, transposed);
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
    
    std::vector<double> x;
    std::vector<int> i;
    std::vector<size_t> p;
};

}

}

#endif
