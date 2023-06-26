#ifndef SCRAN_MODEL_GENE_VAR_H
#define SCRAN_MODEL_GENE_VAR_H

#include "../utils/macros.hpp"

#include "tatami/tatami.hpp"

#include "../utils/vector_to_pointers.hpp"
#include "FitTrendVar.hpp"
#include "blocked_variances.hpp"

#include <algorithm>
#include <vector>
#include <limits>

/**
 * @file ModelGeneVar.hpp
 *
 * @brief Model the per-gene variance from log-count data.
 */

namespace scran {

/**
 * @brief Compute and model the per-gene variances in log-expression data.
 *
 * This scans through a log-transformed normalized expression matrix (e.g., from `LogNormCounts`) and computes per-feature means and variances.
 * It then fits a trend to the variances with respect to the means using `FitTrendVar`.
 * We assume that most genes at any given abundance are not highly variable, such that the fitted value of the trend is interpreted as the "uninteresting" variance - 
 * this is mostly attributed to technical variation like sequencing noise, but can also represent constitutive biological noise like transcriptional bursting.
 * Under this assumption, the residual can be treated as a quantification of biologically interesting variation, and can be used to identify relevant features for downstream analyses.
 */
class ModelGeneVar {
public:
    /**
     * @brief Default parameters for variance modelling.
     */
    struct Defaults {
        /**
         * See `set_num_threads()`.
         */
        static constexpr int num_threads = 1;
    };

private:
    int num_threads = Defaults::num_threads;
    double span = FitTrendVar::Defaults::span;
    double min_mean = FitTrendVar::Defaults::minimum_mean;

    bool use_fixed_width = FitTrendVar::Defaults::use_fixed_width;
    bool fixed_width = FitTrendVar::Defaults::fixed_width;
    int minimum_window_count = FitTrendVar::Defaults::minimum_window_count;

public:
    /** 
     * @param s See `FitTrendVar::set_span()` for more details.
     *
     * @return A reference to this `ModelGeneVar` object.
     */
    ModelGeneVar& set_span(double s = FitTrendVar::Defaults::span) {
        span = s;
        return *this;
    }

    /** 
     * @param m See `FitTrendVar::set_minimum_mean()` for more details.
     *
     * @return A reference to this `ModelGeneVar` object.
     */
    ModelGeneVar& set_minimum_mean(double m = FitTrendVar::Defaults::minimum_mean) {
        min_mean = m;
        return *this;
    }

    /**
     * @param u See `FitTrendVar::set_use_fixed_width()` for more details.
     *
     * @return A reference to this `ModelGeneVar` object.
     */
    ModelGeneVar& set_use_fixed_width(bool u = FitTrendVar::Defaults::use_fixed_width) {
        use_fixed_width = u;
        return *this;
    }

    /**
     * @param f See `FitTrendVar::set_fixed_width()` for more details.
     *
     * @return A reference to this `ModelGeneVar` object.
     */
    ModelGeneVar& set_fixed_width(double f = FitTrendVar::Defaults::fixed_width) {
        fixed_width = f;
        return *this;
    }

    /**
     * @param c See `FitTrendVar::set_minimum_window_count()` for more details.
     *
     * @return A reference to this `ModelGeneVar` object.
     */
    ModelGeneVar& set_minimum_window_count(int c = FitTrendVar::Defaults::minimum_window_count) {
        minimum_window_count = c; 
        return *this;
    }

    /**
     * @param n Number of threads to use. 
     * @return A reference to this `ModelGeneVar` object.
     */
    ModelGeneVar& set_num_threads(int n = Defaults::num_threads) {
        num_threads = n;
        return *this;
    }

private:
    template<bool blocked_, typename Data_, typename Index_, typename Stat_, typename Block_> 
    void compute_dense_row(const tatami::Matrix<Data_, Index_>* mat, std::vector<Stat_*>& means, std::vector<Stat_*>& variances, const Block_* block, const std::vector<Index_>& block_size) const {
        auto nblocks = block_size.size();
        auto NR = mat->nrow(), NC = mat->ncol();

        tatami::parallelize([&](size_t, Index_ start, Index_ length) -> void {
            std::vector<Stat_> tmp_means(nblocks);
            std::vector<Stat_> tmp_vars(nblocks);

            std::vector<Data_> buffer(NC);
            auto ext = tatami::consecutive_extractor<true, false>(mat, start, length);

            for (Index_ r = start, end = start + length; r < end; ++r) {
                auto ptr = ext->fetch(r, buffer.data());
                feature_selection::blocked_variance_with_mean<blocked_>(ptr, NC, block, nblocks, block_size.data(), tmp_means.data(), tmp_vars.data());
                for (size_t b = 0; b < nblocks; ++b) {
                    means[b][r] = tmp_means[b];
                    variances[b][r] = tmp_vars[b];
                }
            }
        }, NR, num_threads);
    }

    template<bool blocked_, typename Data_, typename Index_, typename Stat_, typename Block_> 
    void compute_sparse_row(const tatami::Matrix<Data_, Index_>* mat, std::vector<Stat_*>& means, std::vector<Stat_*>& variances, const Block_* block, const std::vector<Index_>& block_size) const {
        auto nblocks = block_size.size();
        auto NR = mat->nrow(), NC = mat->ncol();

        tatami::parallelize([&](size_t, Index_ start, Index_ length) -> void {
            std::vector<Stat_> tmp_means(nblocks);
            std::vector<Stat_> tmp_vars(nblocks);
            std::vector<Index_> tmp_nzero(nblocks);

            std::vector<Data_> vbuffer(NC);
            std::vector<Index_> ibuffer(NC);
            tatami::Options opt;
            opt.sparse_ordered_index = false;
            auto ext = tatami::consecutive_extractor<true, true>(mat, start, length, opt);

            for (Index_ r = start, end = start + length; r < end; ++r) {
                auto range = ext->fetch(r, vbuffer.data(), ibuffer.data());
                feature_selection::blocked_variance_with_mean<blocked_>(range, block, nblocks, block_size.data(), tmp_means.data(), tmp_vars.data(), tmp_nzero.data());
                for (size_t b = 0; b < nblocks; ++b) {
                    means[b][r] = tmp_means[b];
                    variances[b][r] = tmp_vars[b];
                }
            }
        }, NR, num_threads);
    }

    template<bool blocked_, typename Data_, typename Index_, typename Stat_, typename Block_> 
    void compute_dense_column(const tatami::Matrix<Data_, Index_>* mat, std::vector<Stat_*>& means, std::vector<Stat_*>& variances, const Block_* block, const std::vector<Index_>& block_size) const {
        auto nblocks = block_size.size();
        auto NR = mat->nrow(), NC = mat->ncol();

        tatami::parallelize([&](size_t, Index_ start, Index_ length) -> void {
            std::vector<Data_> buffer(length);
            auto ext = tatami::consecutive_extractor<false, false>(mat, 0, NC, start, length);

            // Shifting pointers to account for the new start point.
            auto mcopy = means;
            auto vcopy = variances;
            for (Index_ b = 0; b < nblocks; ++b) {
                mcopy[b] += start;
                vcopy[b] += start;
            }

            std::vector<Index_> counts(nblocks);
            for (Index_ c = 0; c < NC; ++c) {
                auto ptr = ext->fetch(c, buffer.data());
                auto b = feature_selection::get_block<blocked_>(c, block);
                tatami::stats::variances::compute_running(ptr, length, mcopy[b], vcopy[b], counts[b]);
            }

            for (size_t b = 0; b < nblocks; ++b) {
                tatami::stats::variances::finish_running(length, mcopy[b], vcopy[b], counts[b]);
            }
        }, NR, num_threads);
    }

    template<bool blocked_, typename Data_, typename Index_, typename Stat_, typename Block_> 
    void compute_sparse_column(const tatami::Matrix<Data_, Index_>* mat, std::vector<Stat_*>& means, std::vector<Stat_*>& variances, const Block_* block, const std::vector<Index_>& block_size) const {
        auto nblocks = block_size.size();
        auto NR = mat->nrow(), NC = mat->ncol();
        std::vector<std::vector<Index_> > nonzeros(nblocks, std::vector<Index_>(NR));

        tatami::parallelize([&](size_t, Index_ start, Index_ length) -> void {
            std::vector<Data_> vbuffer(length);
            std::vector<Index_> ibuffer(length);
            tatami::Options opt;
            opt.sparse_ordered_index = false;
            auto ext = tatami::consecutive_extractor<false, true>(mat, 0, NC, start, length, opt);

            std::vector<Index_> counts(nblocks);
            for (Index_ c = 0; c < NC; ++c) {
                auto range = ext->fetch(c, vbuffer.data(), ibuffer.data());;
                auto b = feature_selection::get_block<blocked_>(c, block);
                tatami::stats::variances::compute_running(range, means[b], variances[b], nonzeros[b].data(), counts[b]);
            }

            for (size_t b = 0; b < nblocks; ++b) {
                tatami::stats::variances::finish_running(length, means[b] + start, variances[b] + start, nonzeros[b].data() + start, counts[b]);
            }
        }, NR, num_threads);
    }

private:
    template<bool blocked_, typename Data_, typename Index_, typename Stat_, typename Block_> 
    void compute(const tatami::Matrix<Data_, Index_>* mat, std::vector<Stat_*>& means, std::vector<Stat_*>& variances, const Block_* block, const std::vector<Index_>& block_size) const {
        if (mat->prefer_rows()) {
            if (mat->sparse()) {
                compute_sparse_row<blocked_>(mat, means, variances, block, block_size);
            } else {
                compute_dense_row<blocked_>(mat, means, variances, block, block_size);
            }

        } else {
            // Set everything to zero before computing the running statistics.
            auto NR = mat->nrow();
            for (auto& mptr : means) {
                std::fill(mptr, mptr + NR, 0);
            }
            for (auto& vptr : variances) {
                std::fill(vptr, vptr + NR, 0);
            }

            if (mat->sparse()) {
                compute_sparse_column<blocked_>(mat, means, variances, block, block_size);
            } else {
                compute_dense_column<blocked_>(mat, means, variances, block, block_size);
            }
        }
    }

public:
    /** 
     * Compute and model the per-feature variances from a log-expression matrix.
     *
     * @tparam MAT Type of matrix, usually a `tatami::NumericMatrix`.
     * @tparam Stat Floating-point type for the output statistics.
     *
     * @param mat Pointer to a feature-by-cells **tatami** matrix containing log-expression values.
     * @param[out] means Pointer to an output array of length equal to the number of rows in `mat`, used to store the mean of each feature.
     * @param[out] variances Pointer to an output array of length equal to the number of rows in `mat`, used to store the variance of each feature.
     * @param[out] fitted Pointer to an output array of length equal to the number of rows in `mat`, used to store the fitted value of the trend.
     * @param[out] residuals Pointer to an output array of length equal to the number of rows in `mat`, used to store the residual from the trend for each feature.
     *
     * @return `means`, `variances`, `fitted` and `residuals` are filled with the relevant statistics.
     */
    template<class MAT, typename Stat> 
    void run(const MAT* mat, Stat* means, Stat* variances, Stat* fitted, Stat* residuals) const {
        run_blocked(mat, static_cast<int*>(NULL), std::vector<Stat*>{means}, std::vector<Stat*>{variances}, std::vector<Stat*>{fitted}, std::vector<Stat*>{residuals});
        return;
    }

    /** 
     * Compute and model the per-feature variances from a log-expression matrix with blocking.
     * The mean and variance of each gene is computed separately for all cells in each block, and a separate trend is fitted to each block to obtain residuals.
     * This ensures that sample and batch effects do not confound the variance estimates.
     * We suggest taking the average of the residuals as the effective statistic for feature selection.
     *
     * @tparam MAT Type of matrix, usually a `tatami::NumericMatrix`.
     * @tparam B An integer type, to hold the block IDs.
     * @tparam Stat Floating-point type for the output statistics.
     *
     * @param mat Pointer to a feature-by-cells **tatami** matrix containing log-expression values.
     * @param[in] block Pointer to an array of block identifiers.
     * If provided, the array should be of length equal to `ncells`.
     * Values should be integer IDs in \f$[0, N)\f$ where \f$N\f$ is the number of blocks.
     * This can also be a `nullptr`, in which case all cells are assumed to belong to the same block.
     * @param[out] means Vector of length equal to the number of blocks, containing pointers to output arrays of length equal to the number of rows in `mat`.
     * Each vector stores the mean of each feature in the corresponding block of cells.
     * @param[out] variances Vector of length equal to the number of blocks, containing pointers to output arrays of length equal to the number of rows in `mat`.
     * Each vector stores the variance of each feature in the corresponding block of cells.
     * @param[out] fitted Vector of length equal to the number of blocks, containing pointers to output arrays of length equal to the number of rows in `mat`.
     * Each vector stores the fitted value of the trend for each feature in the corresponding block of cells.
     * @param[out] residuals Vector of length equal to the number of blocks, containing pointers to output arrays of length equal to the number of rows in `mat`.
     * Each vector stores the residual from the trend for each feature in the corresponding block of cells.
     *
     * @return `means`, `variances`, `fitted` and `residuals` are filled with the relevant statistics.
     */
    template<class MAT, typename B, typename Stat>
    void run_blocked(const MAT* mat, const B* block, std::vector<Stat*> means, std::vector<Stat*> variances, std::vector<Stat*> fitted, std::vector<Stat*> residuals) const {
        size_t NR = mat->nrow(), NC = mat->ncol();
        std::vector<typename MAT::index_type> block_size(means.size());

        if (block) {
            auto copy = block;
            for (size_t j = 0; j < NC; ++j, ++copy) {
                ++block_size[*copy];
            }
            compute<true>(mat, means, variances, block, block_size);
        } else {
            block_size[0] = NC;
            compute<false>(mat, means, variances, block, block_size);
        }

        // Applying the trend fit to each block.
        FitTrendVar fit;
        fit.set_span(span);
        fit.set_minimum_mean(min_mean);
        fit.set_use_fixed_width(use_fixed_width);
        fit.set_fixed_width(fixed_width);
        fit.set_minimum_window_count(minimum_window_count);

        for (size_t b = 0; b < block_size.size(); ++b) {
            if (block_size[b] >= 2) {
                fit.run(NR, means[b], variances[b], fitted[b], residuals[b]);
            } else {
                std::fill(fitted[b], fitted[b] + NR, std::numeric_limits<double>::quiet_NaN());
                std::fill(residuals[b], residuals[b] + NR, std::numeric_limits<double>::quiet_NaN());
            }
        }

        return;
    }

public:
    /**
     * @brief Results of the variance modelling.
     *
     * Meaningful instances of this object should generally be constructed by calling the `ModelGeneVar::run()` methods.
     * Empty instances can be default-constructed as placeholders.
     */
    struct Results {
        /**
         * @cond
         */
        Results() {}

        Results(size_t ngenes, int nblocks) : means(nblocks, std::vector<double>(ngenes)),
                                              variances(nblocks, std::vector<double>(ngenes)),
                                              fitted(nblocks, std::vector<double>(ngenes)),
                                              residuals(nblocks, std::vector<double>(ngenes)) {}
        /**
         * @endcond
         */

        /**
         * Vector of length equal to the number of blocks, where each internal vector is of length equal to the number of genes.
         * Each entry contains the mean log-expression for each gene in each block.
         */
        std::vector<std::vector<double> > means;

        /**
         * Vector of vectors of the same dimensions as `means`.
         * Each entry contains the variance for each gene in each block.
         */
        std::vector<std::vector<double> > variances;

        /**
         * Vector of vectors of the same dimensions as `means`.
         * Each entry contains the fitted value for each gene in each block.
         */
        std::vector<std::vector<double> > fitted;

        /**
         * Vector of vectors of the same dimensions as `means`.
         * Each vector contains the residual for each gene in each block.
         */
        std::vector<std::vector<double> > residuals;
    };

    /** 
     * Compute and model the per-feature variances from a log-expression matrix.
     *
     * @tparam MAT Type of matrix, usually a `tatami::NumericMatrix`.
     *
     * @param mat Pointer to a feature-by-cells **tatami** matrix containing log-expression values.
     *
     * @return A `Results` object containing the results of the variance modelling.
     */
    template<class MAT>
    Results run(const MAT* mat) const {
        Results output(mat->nrow(), 1);
        run(mat, output.means[0].data(), output.variances[0].data(), output.fitted[0].data(), output.residuals[0].data());
        return output;
    }

    /** 
     * Compute and model the per-feature variances from a log-expression matrix with blocking, see `run_blocked()` for details.
     *
     * @tparam MAT Type of matrix, usually a `tatami::NumericMatrix`.
     * @tparam B An integer type, to hold the block IDs.
     *
     * @param mat Pointer to a feature-by-cells **tatami** matrix containing log-expression values.
     * @param[in] block Pointer to an array of block identifiers, see `run_blocked()` for details.
     *
     * @return A `Results` object containing the results of the variance modelling.
     */
    template<class MAT, typename B>
    Results run_blocked(const MAT* mat, const B* block) const {
        int nblocks = 1;
        if (block) {
            if (mat->ncol()) {
                nblocks = *std::max_element(block, block + mat->ncol()) + 1;
            }
        }

        Results output(mat->nrow(), nblocks);
        std::vector<double*> mean_ptr, var_ptr, fit_ptr, resid_ptr;
        for (int b = 0; b < nblocks; ++b) {
            mean_ptr.push_back(output.means[b].data());
            var_ptr.push_back(output.variances[b].data());
            fit_ptr.push_back(output.fitted[b].data());
            resid_ptr.push_back(output.residuals[b].data());
        }

        run_blocked(mat, block, std::move(mean_ptr), std::move(var_ptr), std::move(fit_ptr), std::move(resid_ptr));
        return output;
    }
};

}

#endif
