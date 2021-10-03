#ifndef SCRAN_RUN_PCA
#define SCRAN_RUN_PCA

#include "tatami/stats/variances.hpp"
#include "tatami/base/DelayedSubset.hpp"

#include "irlba/irlba.hpp"
#include "Eigen/Dense"
#include "Eigen/Sparse"

#include <vector>
#include <cmath>

namespace scran {

class RunPCA {
public:
    struct Defaults {
        static constexpr int rank = 10;

        static constexpr bool scale = 10;
    };
private:
    int rank = Defaults::rank;
    bool scale = Defaults::scale;
    irlba::Irlba irb;

public:
    RunPCA& set_rank(int r = Defaults::rank) {
        rank = r;
        return *this;
    }

    RunPCA& set_scale(bool s = Defaults::scale) {
        scale = s;
        return *this;
    }

private:
    template<typename T, typename IDX>
    void run(const tatami::Matrix<T, IDX>* mat, Eigen::MatrixXd& pcs, Eigen::VectorXd& variance_explained, double& total_var) {
        irlba::NormalSampler norm(42);
        irb.set_number(rank);

        if (mat->sparse()) {
            Eigen::VectorXd center_v(mat->nrow()), scale_v(mat->nrow());
            auto emat = create_eigen_matrix_sparse(mat, center_v, scale_v, total_var);

#ifdef SCRAN_LOGGER
            SCRAN_LOGGER("scran::RunPCA", "Running the IRLBA algorithm");
#endif

            auto result = irb.run(emat, center_v, scale_v, norm);
            clean_up(mat->ncol(), result.U, result.D, pcs, variance_explained);
        } else {
            auto emat = create_eigen_matrix_dense(mat, total_var);

#ifdef SCRAN_LOGGER
            SCRAN_LOGGER("scran::RunPCA", "Running the IRLBA algorithm");
#endif

            auto result = irb.run(emat, norm); // already centered and scaled, if relevant.
            clean_up(mat->ncol(), result.U, result.D, pcs, variance_explained);
        }

        return;
    }

public:
    struct Results {
        Eigen::MatrixXd pcs;
        Eigen::VectorXd variance_explained;
        double total_variance = 0;
    };

    template<typename T, typename IDX>
    Results run(const tatami::Matrix<T, IDX>* mat) {
        Results output;
        run(mat, output.pcs, output.variance_explained, output.total_variance);
        return output;
    }

    template<typename T, typename IDX, typename X>
    Results run(const tatami::Matrix<T, IDX>* mat, const X* features) {
        Results output;

        if (!features) {
            run(mat, output.pcs, output.variance_explained, output.total_variance);
        } else {
#ifdef SCRAN_LOGGER
            SCRAN_LOGGER("RunPCA", "Subsetting to features of interest");
#endif
            std::vector<int> subset;
            subset.reserve(mat->nrow());
            for (size_t r = 0; r < mat->nrow(); ++r) {
                if (features[r]) {
                    subset.push_back(r);
                }
            }

            // Using a no-op deleter in a shared pointer to get it to work with
            // the DelayedSubset. This hacky shared pointer dies once we move
            // out of this block, and the matrix will continue to exist
            // outside, so we shouldn't have any problems with lifetimes. 
            std::shared_ptr<const tatami::Matrix<T, IDX> > ptr(mat, [](const tatami::Matrix<T, IDX>*){});

            auto subsetted = tatami::make_DelayedSubset<0>(std::move(ptr), std::move(subset));
            run(subsetted.get(), output.pcs, output.variance_explained, output.total_variance);
        }

        return output;
    }

private:
    void clean_up(size_t NC, const Eigen::MatrixXd& U, const Eigen::VectorXd& D, Eigen::MatrixXd& pcs, Eigen::VectorXd& variance_explained) {
#ifdef SCRAN_LOGGER
       SCRAN_LOGGER("scran::RunPCA", "Reformatting the output PCs");
#endif

        pcs = U;
        for (int i = 0; i < U.cols(); ++i) {
            for (size_t j = 0; j < NC; ++j) {
                pcs(j, i) *= D[i];
            }
        }

        variance_explained.resize(D.size());
        for (int i = 0; i < D.size(); ++i) {
            variance_explained[i] = D[i] * D[i] / static_cast<double>(NC - 1);
        }

        return;
    }

private:
    template<typename T, typename IDX> 
    Eigen::SparseMatrix<double> create_eigen_matrix_sparse(const tatami::Matrix<T, IDX>* mat, Eigen::VectorXd& center_v, Eigen::VectorXd& scale_v, double& total_var) {
        size_t NR = mat->nrow(), NC = mat->ncol();
        total_var = 0;

        Eigen::SparseMatrix<double> A(NC, NR); // transposed; we want genes in the columns.
        std::vector<std::vector<double> > values;
        std::vector<std::vector<int> > indices;

#ifdef SCRAN_LOGGER
        SCRAN_LOGGER("scran::RunPCA", "Preparing the input matrix");
#endif

        if (mat->prefer_rows()) {
            std::vector<double> xbuffer(NC);
            std::vector<int> ibuffer(NC);
            std::vector<int> nnzeros(NR);
            values.reserve(NR);
            indices.reserve(NR);

            for (size_t r = 0; r < NR; ++r) {
                auto range = mat->sparse_row(r, xbuffer.data(), ibuffer.data());

                auto stats = tatami::stats::variances::compute_direct(range, NC);
                center_v[r] = stats.first;
                if (scale) {
                    ++total_var;
                    scale_v[r] = std::sqrt(stats.second);
                } else {
                    total_var += stats.second;
                    scale_v[r] = 1;
                }

                nnzeros[r] = range.number;
                values.emplace_back(range.value, range.value + range.number);
                indices.emplace_back(range.index, range.index + range.number);
            }

            // Actually filling the matrix. Note the implicit transposition.
            A.reserve(nnzeros);
            for (size_t r = 0; r < NR; ++r) {
                const auto& curi = indices[r];
                const auto& curv = values[r];
                for (size_t i = 0; i < curi.size(); ++i) {
                    A.insert(curi[i], r) = curv[i];
                }
            }

        } else {
            std::vector<double> xbuffer(NR);
            std::vector<int> ibuffer(NR);
            values.reserve(NC);
            indices.reserve(NC);

            center_v.setZero();
            scale_v.setZero();
            std::vector<int> nonzeros(NR);
            int count = 0;

            // First pass to compute variances and extract non-zero values.
            for (size_t c = 0; c < NC; ++c) {
                auto range = mat->sparse_column(c, xbuffer.data(), ibuffer.data());
                tatami::stats::variances::compute_running(range, center_v.data(), scale_v.data(), nonzeros.data(), count);
                values.emplace_back(range.value, range.value + range.number);
                indices.emplace_back(range.index, range.index + range.number);
            }

            tatami::stats::variances::finish_running(NR, center_v.data(), scale_v.data(), nonzeros.data(), count);

            if (scale) {
                for (auto& s : scale_v) { s = std::sqrt(s); }
                total_var = NR;
            } else {
                total_var = std::accumulate(scale_v.begin(), scale_v.end(), 0.0);
                std::fill(scale_v.begin(), scale_v.end(), 1);
            }

            // Actually filling the matrix. Note the implicit transposition.
            A.reserve(nonzeros);
            for (size_t c = 0; c < NC; ++c) {
                const auto& curi = indices[c];
                const auto& curv = values[c];
                for (size_t i = 0; i < curi.size(); ++i) {
                    A.insert(c, curi[i]) = curv[i];
                }
            }
        }

        A.makeCompressed();
        return A;
    }

private:
    template<typename T, typename IDX> 
    Eigen::MatrixXd create_eigen_matrix_dense(const tatami::Matrix<T, IDX>* mat, double& total_var) {
        size_t NR = mat->nrow(), NC = mat->ncol();
        total_var = 0;

        Eigen::MatrixXd output(NC, NR); // transposed.
        std::vector<double> xbuffer(NC);
        double* outIt = output.data();

#ifdef SCRAN_LOGGER
        SCRAN_LOGGER("scran::RunPCA", "Preparing the input matrix");
#endif

        for (size_t r = 0; r < NR; ++r, outIt += NC) {
            auto ptr = mat->row_copy(r, outIt);
            auto stats = tatami::stats::variances::compute_direct(outIt, NC);

            auto copy = outIt;
            for (size_t c = 0; c < NC; ++c, ++copy) {
                *copy -= stats.first;
            }

            if (scale) {
                double sd = std::sqrt(stats.second);
                auto copy = outIt;
                for (size_t c = 0; c < NC; ++c, ++copy) {
                    *copy /= sd;
                }
                ++total_var;
            } else {
                total_var += stats.second;
            }
        }

        return output;
    }

};

}

#endif
