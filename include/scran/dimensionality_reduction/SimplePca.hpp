#ifndef SCRAN_SIMPLE_PCA_HPP
#define SCRAN_SIMPLE_PCA_HPP

#include "../utils/macros.hpp"

#include "tatami/tatami.hpp"
#include "irlba/irlba.hpp"
#include "Eigen/Dense"

#include <vector>
#include <cmath>

#include "utils.hpp"

/**
 * @file SimplePca.hpp
 *
 * @brief Perform a simple PCA on a gene-by-cell matrix.
 */

namespace scran {

/**
 * @brief Perform a simple PCA on a gene-cell matrix.
 *
 * Principal components analysis (PCA) is a helpful technique for data compression and denoising.
 * The idea is that the earlier PCs capture most of the systematic biological variation while the later PCs capture random technical noise.
 * Thus, we can reduce the size of the data and eliminate noise by only using the earlier PCs for further analyses.
 * Most practitioners will keep the first 10-50 PCs, though the exact choice is fairly arbitrary.
 * For speed, we use the [**CppIrlba**](https://github.com/LTLA/CppIrlba) package to perform an approximate PCA.
 */
class SimplePca {
public:
    /**
     * @brief Default parameter settings.
     */
    struct Defaults {
        /**
         * See `set_rank()` for more details.
         */
        static constexpr int rank = 10;

        /**
         * See `set_scale()` for more details.
         */
        static constexpr bool scale = false;

        /**
         * See `set_transpose()` for more details.
         */
        static constexpr bool transpose = true;

        /**
         * See `set_num_threads()` for more details.
         */
        static constexpr int num_threads = 1;
    };
private:
    bool scale = Defaults::scale;
    bool transpose = Defaults::transpose;
    int rank = Defaults::rank;
    int nthreads = Defaults::num_threads;

public:
    /**
     * @param r Number of PCs to compute.
     * This should be smaller than the smaller dimension of the input matrix.
     *
     * @return A reference to this `SimplePca` instance.
     */
    SimplePca& set_rank(int r = Defaults::rank) {
        rank = r;
        return *this;
    }

    /**
     * @param s Should genes be scaled to unit variance?
     *
     * @return A reference to this `SimplePca` instance.
     */
    SimplePca& set_scale(bool s = Defaults::scale) {
        scale = s;
        return *this;
    }

    /**
     * @param t Should the PC matrix be transposed on output?
     * If `true`, the output matrix is column-major with cells in the columns, which is compatible with downstream **libscran** steps.
     * 
     * @return A reference to this `SimplePca` instance.
     */
    SimplePca& set_transpose(bool t = Defaults::transpose) {
        transpose = t;
        return *this;
    }

    /**
     * @param n Number of threads to use.
     * @return A reference to this `SimplePca` instance.
     */
    SimplePca& set_num_threads(int n = Defaults::num_threads) {
        nthreads = n;
        return *this;
    }

private:
    template<typename T, typename IDX>
    void run(const tatami::Matrix<T, IDX>* mat, Eigen::MatrixXd& pcs, Eigen::MatrixXd& rotation, Eigen::VectorXd& variance_explained, double& total_var) const {
        irlba::EigenThreadScope t(nthreads);
        irlba::Irlba irb;
        irb.set_number(rank);

        if (mat->sparse()) {
            auto extracted = pca_utils::extract_sparse_for_pca(mat, nthreads); // row-major extraction to create a CSR matrix.
            auto& ptrs = extracted.ptrs;
            auto& values = extracted.values;
            auto& indices = extracted.indices;

            size_t NR = mat->nrow(), NC = mat->ncol();
            Eigen::VectorXd center_v(NR), scale_v(NR);
            tatami::parallelize([&](size_t, size_t start, size_t length) -> void {
                for (int r = start, end = start + length; r < end; ++r) {
                    auto offset = ptrs[r];

                    tatami::SparseRange<double, int> range;
                    range.number = ptrs[r+1] - offset;
                    range.value = values.data() + offset;
                    range.index = indices.data() + offset;

                    auto results = tatami::stats::variances::compute_direct(range, NC);
                    center_v.coeffRef(r) = results.first;
                    scale_v.coeffRef(r) = results.second;
                }
            }, NR, nthreads);

            total_var = pca_utils::process_scale_vector(scale, scale_v);

            // Actually creating a sparse matrix. Again, note that this is transposed; we want genes in the columns.
            pca_utils::SparseMatrix emat(NC, NR, std::move(values), std::move(indices), std::move(ptrs), nthreads);

            irlba::Centered<decltype(emat)> centered(&emat, &center_v);
            if (scale) {
                irlba::Scaled<decltype(centered)> scaled(&centered, &scale_v);
                irb.run(scaled, pcs, rotation, variance_explained);
            } else {
                irb.run(centered, pcs, rotation, variance_explained);
            }

        } else {
            auto emat = pca_utils::extract_dense_for_pca(mat, nthreads); // get a column-major matrix with genes in columns.

            size_t NC = emat.cols();
            Eigen::VectorXd center_v(NC), scale_v(NC);
            tatami::parallelize([&](size_t, size_t start, size_t length) -> void {
                size_t NR = emat.rows();
                const double* ptr = emat.data() + static_cast<size_t>(start) * NR; // enforce size_t to avoid overflow issues.
                for (size_t c = start, end = start + length; c < end; ++c, ptr += NR) {
                    auto results = tatami::stats::variances::compute_direct(ptr, NR);
                    center_v.coeffRef(c) = results.first;
                    scale_v.coeffRef(c) = results.second;
                }
            }, NC, nthreads);

            total_var = pca_utils::process_scale_vector(scale, scale_v);

            // Applying the centering and scaling now so we can do the PCA without any wrappers.
            pca_utils::apply_dense_center_and_scale(emat, center_v, scale, scale_v, nthreads);

            irb.run(emat, pcs, rotation, variance_explained); 
        }

        pca_utils::clean_up(mat->ncol(), pcs, variance_explained);
        if (transpose) {
            pcs.adjointInPlace();
        }

        return;
    }

public:
    /**
     * @brief Container for the PCA results.
     *
     * Instances should be constructed by the `SimplePca::run()` methods.
     */
    struct Results {
        /**
         * Matrix of principal components.
         * By default, each row corresponds to a PC while each column corresponds to a cell in the input matrix.
         * If `set_transpose()` is set to `false`, rows are cells instead.
         * The number of PCs is determined by `set_rank()`.
         */
        Eigen::MatrixXd pcs;

        /**
         * Variance explained by each PC.
         * Each entry corresponds to a column in `pcs` and is in decreasing order.
         */
        Eigen::VectorXd variance_explained;

        /**
         * Total variance of the dataset (possibly after scaling, if `set_scale()` is set to `true`).
         * This can be used to divide `variance_explained` to obtain the percentage of variance explained.
         */
        double total_variance = 0;

        /**
         * Rotation matrix.
         * Each row corresponds to a feature while each column corresponds to a PC.
         * The number of PCs is determined by `set_rank()`.
         * If feature filtering was performed, the number of rows is equal to the number of features remaining after filtering.
         */
        Eigen::MatrixXd rotation;
    };

    /**
     * Run PCA on an input gene-by-cell matrix.
     *
     * @tparam T Floating point type for the data.
     * @tparam IDX Integer type for the indices.
     *
     * @param[in] mat Pointer to the input matrix.
     * Columns should contain cells while rows should contain genes.
     *
     * @return A `Results` object containing the PCs and the variance explained.
     */
    template<typename T, typename IDX>
    Results run(const tatami::Matrix<T, IDX>* mat) const {
        Results output;
        run(mat, output.pcs, output.rotation, output.variance_explained, output.total_variance);
        return output;
    }

    /**
     * Run PCA on an input gene-by-cell matrix after filtering for genes of interest.
     * We typically use the set of highly variable genes from `ChooseHVGs`, 
     * with the aim being to improve computational efficiency and avoid random noise by removing lowly variable genes.
     *
     * @tparam T Floating point type for the data.
     * @tparam IDX Integer type for the indices.
     * @tparam X Integer type for the feature filter.
     *
     * @param[in] mat Pointer to the input matrix.
     * Columns should contain cells while rows should contain genes.
     * @param[in] features Pointer to an array of length equal to the number of genes.
     * Each entry treated as a boolean specifying whether the corresponding genes should be used in the PCA.
     *
     * @return A `Results` object containing the PCs and the variance explained.
     */
    template<typename T, typename IDX, typename X>
    Results run(const tatami::Matrix<T, IDX>* mat, const X* features) const {
        Results output;

        if (!features) {
            run(mat, output.pcs, output.rotation, output.variance_explained, output.total_variance);
        } else {
            auto subsetted = pca_utils::subset_matrix_by_features(mat, features);
            run(subsetted.get(), output.pcs, output.rotation, output.variance_explained, output.total_variance);
        }

        return output;
    }
};

}

#endif
