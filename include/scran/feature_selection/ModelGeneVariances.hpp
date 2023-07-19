#ifndef SCRAN_MODEL_GENE_VAR_H
#define SCRAN_MODEL_GENE_VAR_H

#include "../utils/macros.hpp"

#include "tatami/tatami.hpp"

#include "../utils/vector_to_pointers.hpp"
#include "../utils/blocking.hpp"
#include "../utils/average_vectors.hpp"

#include "FitVarianceTrend.hpp"
#include "blocked_variances.hpp"

#include <algorithm>
#include <vector>
#include <limits>

/**
 * @file ModelGeneVariances.hpp
 *
 * @brief Model the per-gene variance from log-count data.
 */

namespace scran {

/**
 * @brief Compute and model the per-gene variances in log-expression data.
 *
 * This scans through a log-transformed normalized expression matrix (e.g., from `LogNormCounts`) and computes per-feature means and variances.
 * It then fits a trend to the variances with respect to the means using `FitVarianceTrend`.
 * We assume that most genes at any given abundance are not highly variable, such that the fitted value of the trend is interpreted as the "uninteresting" variance - 
 * this is mostly attributed to technical variation like sequencing noise, but can also represent constitutive biological noise like transcriptional bursting.
 * Under this assumption, the residual can be treated as a quantification of biologically interesting variation, and can be used to identify relevant features for downstream analyses.
 */
class ModelGeneVariances {
public:
    /**
     * @brief Default parameters for variance modelling.
     */
    struct Defaults {
        /**
         * See `set_block_weight_policy()` for more details.
         */
        static constexpr WeightPolicy block_weight_policy = WeightPolicy::VARIABLE;

        /**
         * See `set_variable_block_weight_parameters()` for more details.
         */
        static constexpr VariableBlockWeightParameters variable_block_weight_parameters = VariableBlockWeightParameters();

        /**
         * See `set_compute_average()`.
         */
        static constexpr bool compute_average = true;

        /**
         * See `set_num_threads()`.
         */
        static constexpr int num_threads = 1;
    };

private:
    WeightPolicy block_weight_policy = Defaults::block_weight_policy;
    VariableBlockWeightParameters variable_block_weight_parameters = Defaults::variable_block_weight_parameters;
    int num_threads = Defaults::num_threads;

    double span = FitVarianceTrend::Defaults::span;
    double min_mean = FitVarianceTrend::Defaults::minimum_mean;

    bool use_fixed_width = FitVarianceTrend::Defaults::use_fixed_width;
    bool fixed_width = FitVarianceTrend::Defaults::fixed_width;
    int minimum_window_count = FitVarianceTrend::Defaults::minimum_window_count;

    bool compute_average = Defaults::compute_average;

public:
    /** 
     * @param s See `FitVarianceTrend::set_span()` for more details.
     *
     * @return A reference to this `ModelGeneVariances` object.
     */
    ModelGeneVariances& set_span(double s = FitVarianceTrend::Defaults::span) {
        span = s;
        return *this;
    }

    /** 
     * @param m See `FitVarianceTrend::set_minimum_mean()` for more details.
     *
     * @return A reference to this `ModelGeneVariances` object.
     */
    ModelGeneVariances& set_minimum_mean(double m = FitVarianceTrend::Defaults::minimum_mean) {
        min_mean = m;
        return *this;
    }

    /**
     * @param u See `FitVarianceTrend::set_use_fixed_width()` for more details.
     *
     * @return A reference to this `ModelGeneVariances` object.
     */
    ModelGeneVariances& set_use_fixed_width(bool u = FitVarianceTrend::Defaults::use_fixed_width) {
        use_fixed_width = u;
        return *this;
    }

    /**
     * @param f See `FitVarianceTrend::set_fixed_width()` for more details.
     *
     * @return A reference to this `ModelGeneVariances` object.
     */
    ModelGeneVariances& set_fixed_width(double f = FitVarianceTrend::Defaults::fixed_width) {
        fixed_width = f;
        return *this;
    }

    /**
     * @param c See `FitVarianceTrend::set_minimum_window_count()` for more details.
     *
     * @return A reference to this `ModelGeneVariances` object.
     */
    ModelGeneVariances& set_minimum_window_count(int c = FitVarianceTrend::Defaults::minimum_window_count) {
        minimum_window_count = c; 
        return *this;
    }

    /**
     * @param w Weighting policy to use for averaging statistics across blocks.
     * 
     * @return A reference to this `ModelGeneVariances` instance.
     */
    ModelGeneVariances& set_block_weight_policy(WeightPolicy w = Defaults::block_weight_policy) {
        block_weight_policy = w;
        return *this;
    }

    /**
     * @param v Parameters for the variable block weights, see `variable_block_weight()` for more details.
     * Only used when the block weight policy is set to `WeightPolicy::VARIABLE`.
     * 
     * @return A reference to this `ModelGeneVariances` instance.
     */
    ModelGeneVariances& set_variable_block_weight_parameters(VariableBlockWeightParameters v = Defaults::variable_block_weight_parameters) {
        variable_block_weight_parameters = v;
        return *this;
    }

    /**
     * @param a Whether to compute the average of each statistic across blocks.
     * Note that this only affects the `run_blocked()` method that returns a `BlockResults` object.
     * @return A reference to this `ModelGeneVariances` object.
     */
    ModelGeneVariances& set_compute_average(bool a = Defaults::compute_average) {
        compute_average = a;
        return *this;
    }

    /**
     * @param n Number of threads to use. 
     * @return A reference to this `ModelGeneVariances` object.
     */
    ModelGeneVariances& set_num_threads(int n = Defaults::num_threads) {
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
     * This returns the mean and variance for each feature, as well as the fitted value and residuals from the mean-variance trend fitted across features.
     *
     * @tparam Value_ Data type of the matrix.
     * @tparam Index_ Integer type for the row/column indices.
     * @tparam Stat_ Floating-point type for the output statistics.
     *
     * @param mat Pointer to a feature-by-cells **tatami** matrix containing log-expression values.
     * @param[out] means Pointer to an output array of length equal to the number of rows in `mat`, used to store the mean of each feature.
     * @param[out] variances Pointer to an output array of length equal to the number of rows in `mat`, used to store the variance of each feature.
     * @param[out] fitted Pointer to an output array of length equal to the number of rows in `mat`, used to store the fitted value of the trend.
     * @param[out] residuals Pointer to an output array of length equal to the number of rows in `mat`, used to store the residual from the trend for each feature.
     */
    template<typename Value_, typename Index_, typename Stat_> 
    void run(const tatami::Matrix<Value_, Index_>* mat, Stat_* means, Stat_* variances, Stat_* fitted, Stat_* residuals) const {
        run_blocked(mat, static_cast<int*>(NULL), std::vector<Stat_*>{means}, std::vector<Stat_*>{variances}, std::vector<Stat_*>{fitted}, std::vector<Stat_*>{residuals});
        return;
    }

    /** 
     * Compute and model the per-feature variances from a log-expression matrix with blocking.
     * The mean and variance of each gene is computed separately for all cells in each block, and a separate trend is fitted to each block to obtain residuals.
     * This ensures that sample and batch effects do not confound the variance estimates.
     *
     * We also compute the average of each statistic across blocks, using the weighting strategy described in `weight_block()`.
     * The average residual is particularly useful for feature selection with `ChooseHVGs`.
     *
     * @tparam Value_ Data type of the matrix.
     * @tparam Index_ Integer type for the row/column indices.
     * @tparam Block_ Integer type to hold the block IDs.
     * @tparam Stat_ Floating-point type for the output statistics.
     *
     * @param mat Pointer to a feature-by-cells **tatami** matrix containing log-expression values.
     * @param[in] block Pointer to an array of length equal to the number of cells, containing a 0-based block ID for each cell - see `tabulate_ids()` for more details.
     * This can also be a `nullptr`, in which case all cells are assumed to belong to the same block.
     * @param[out] means Vector of length equal to the number of blocks, containing pointers to output arrays of length equal to the number of rows in `mat`.
     * Each vector stores the mean of each feature in the corresponding block of cells.
     * @param[out] variances Vector of length equal to the number of blocks, containing pointers to output arrays of length equal to the number of rows in `mat`.
     * Each vector stores the variance of each feature in the corresponding block of cells.
     * @param[out] fitted Vector of length equal to the number of blocks, containing pointers to output arrays of length equal to the number of rows in `mat`.
     * Each vector stores the fitted value of the trend for each feature in the corresponding block of cells.
     * @param[out] residuals Vector of length equal to the number of blocks, containing pointers to output arrays of length equal to the number of rows in `mat`.
     * Each vector stores the residual from the trend for each feature in the corresponding block of cells.
     * @param[out] ave_means Pointer to an array of length equal to the number of rows in `mat`, storing the average mean across blocks for each gene.
     * If `nullptr`, the average calculation is skipped.
     * @param[out] ave_variances Pointer to an array of length equal to the number of rows in `mat`, storing the average variance across blocks for each gene.
     * If `nullptr`, the average calculation is skipped.
     * @param[out] ave_fitted Pointer to an array of length equal to the number of rows in `mat`, storing the average fitted value across blocks for each gene.
     * If `nullptr`, the average calculation is skipped.
     * @param[out] ave_residuals Pointer to an array of length equal to the number of rows in `mat`, storing the average residual across blocks for each gene.
     * If `nullptr`, the average calculation is skipped.
     */
    template<typename Value_, typename Index_, typename Block_, typename Stat_>
    void run_blocked(
        const tatami::Matrix<Value_, Index_>* mat, 
        const Block_* block, 
        std::vector<Stat_*> means, 
        std::vector<Stat_*> variances,
        std::vector<Stat_*> fitted, 
        std::vector<Stat_*> residuals,
        Stat_* ave_means,
        Stat_* ave_variances,
        Stat_* ave_fitted,
        Stat_* ave_residuals)
    const {
        Index_ NR = mat->nrow(), NC = mat->ncol();
        std::vector<Index_> block_size;

        if (block) {
            block_size = tabulate_ids(NC, block);
            compute<true>(mat, means, variances, block, block_size);
        } else {
            block_size.push_back(NC); // everything is one big block.
            compute<false>(mat, means, variances, block, block_size);
        }

        // Applying the trend fit to each block.
        FitVarianceTrend fit;
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

        // Computing averages under different policies.
        if (ave_means || ave_variances || ave_fitted || ave_residuals) {
            std::vector<double> block_weight = compute_block_weights(block_size, block_weight_policy, variable_block_weight_parameters);
            if (ave_means) {
                average_vectors_weighted(NR, means, block_weight.data(), ave_means);
            }
            if (ave_variances) {
                average_vectors_weighted(NR, variances, block_weight.data(), ave_variances);
            }
            if (ave_fitted) {
                average_vectors_weighted(NR, fitted, block_weight.data(), ave_fitted);
            }
            if (ave_residuals) {
                average_vectors_weighted(NR, residuals, block_weight.data(), ave_residuals);
            }
        }

        return;
    }

    /** 
     * Compute and model the per-feature variances from a log-expression matrix with blocking.
     * This overload omits the calculation of the averaged statistics across blocks.
     *
     * @tparam Value_ Data type of the matrix.
     * @tparam Index_ Integer type for the row/column indices.
     * @tparam Block_ Integer type to hold the block IDs.
     * @tparam Stat_ Floating-point type for the output statistics.
     *
     * @param mat Pointer to a feature-by-cells **tatami** matrix containing log-expression values.
     * @param[in] block Pointer to an array of length equal to the number of cells, containing block identifiers - see `tabulate_ids()` for more details.
     * This can also be a `nullptr`, in which case all cells are assumed to belong to the same block.
     * @param[out] means Vector of length equal to the number of blocks, containing pointers to output arrays of length equal to the number of rows in `mat`.
     * Each vector stores the mean of each feature in the corresponding block of cells.
     * @param[out] variances Vector of length equal to the number of blocks, containing pointers to output arrays of length equal to the number of rows in `mat`.
     * Each vector stores the variance of each feature in the corresponding block of cells.
     * @param[out] fitted Vector of length equal to the number of blocks, containing pointers to output arrays of length equal to the number of rows in `mat`.
     * Each vector stores the fitted value of the trend for each feature in the corresponding block of cells.
     * @param[out] residuals Vector of length equal to the number of blocks, containing pointers to output arrays of length equal to the number of rows in `mat`.
     * Each vector stores the residual from the trend for each feature in the corresponding block of cells.
     */
    template<typename Value_, typename Index_, typename Block_, typename Stat_>
    void run_blocked(
        const tatami::Matrix<Value_, Index_>* mat, 
        const Block_* block, 
        std::vector<Stat_*> means, 
        std::vector<Stat_*> variances,
        std::vector<Stat_*> fitted, 
        std::vector<Stat_*> residuals)
    const {
        run_blocked(
            mat, 
            block, 
            std::move(means), 
            std::move(variances), 
            std::move(fitted), 
            std::move(residuals),
            static_cast<Stat_*>(NULL),
            static_cast<Stat_*>(NULL),
            static_cast<Stat_*>(NULL),
            static_cast<Stat_*>(NULL)
        );
    }

public:
    /**
     * @brief Results of variance modelling without blocks.
     *
     * Meaningful instances of this object should generally be constructed by calling the `ModelGeneVariances::run()` methods.
     * Empty instances can be default-constructed as placeholders.
     */
    struct Results {
        /**
         * @cond
         */
        Results() {}

        Results(size_t ngenes) : means(ngenes), variances(ngenes), fitted(ngenes), residuals(ngenes) {}
        /**
         * @endcond
         */

        /**
         * Vector of length equal to the number of genes, containing the mean log-expression for each gene.
         */
        std::vector<double> means;

        /**
         * Vector of length equal to the number of genes, containing the variance in the log-expression for each gene.
         */
        std::vector<double> variances;

        /**
         * Vector of length equal to the number of genes, containing the fitted value of the mean-variance trend for each gene.
         */
        std::vector<double> fitted;

        /**
         * Vector of length equal to the number of genes, containing the residuals of the mean-variance trend for each gene.
         */
        std::vector<double> residuals;
    };

    /** 
     * Compute and model the per-feature variances from a log-expression matrix.
     *
     * @tparam Value_ Data type of the matrix.
     * @tparam Index_ Integer type for the row/column indices.
     *
     * @param mat Pointer to a feature-by-cells **tatami** matrix containing log-expression values.
     *
     * @return A `Results` object containing the results of the variance modelling.
     */
    template<typename Value_, typename Index_>
    Results run(const tatami::Matrix<Value_, Index_>* mat) const {
        Results output(mat->nrow());
        run(mat, output.means.data(), output.variances.data(), output.fitted.data(), output.residuals.data());
        return output;
    }

public:
    /**
     * @brief Results of variance modelling with blocks.
     *
     * Meaningful instances of this object should generally be constructed by calling the `ModelGeneVariances::run_blocked()` method.
     * Empty instances can be default-constructed as placeholders.
     */
    struct BlockResults {
        /**
         * @cond
         */
        BlockResults() {}

        BlockResults(size_t ngenes, int nblocks, bool compute_average) : 
            per_block(nblocks, Results(ngenes)),
            average(compute_average ? ngenes : 0) {}
        /**
         * @endcond
         */

        /**
         * Vector of length equal to the number of blocks, where each entry contains the variance modelling results for a single block.
         */
        std::vector<Results> per_block;

        /**
         * Average across blocks for all statistics in `per_block`.
         */
        Results average;
    };

private:
    template<typename Stat_>
    static void fill_pointers(
        int nblocks,
        BlockResults& output,
        std::vector<Stat_*>& mean_ptr, 
        std::vector<Stat_*>& var_ptr,
        std::vector<Stat_*>& fit_ptr,
        std::vector<Stat_*>& resid_ptr
    ) {
        mean_ptr.reserve(nblocks);
        var_ptr.reserve(nblocks);
        fit_ptr.reserve(nblocks);
        resid_ptr.reserve(nblocks);

        for (int b = 0; b < nblocks; ++b) {
            mean_ptr.push_back(output.per_block[b].means.data());
            var_ptr.push_back(output.per_block[b].variances.data());
            fit_ptr.push_back(output.per_block[b].fitted.data());
            resid_ptr.push_back(output.per_block[b].residuals.data());
        }
    }

public:
    /** 
     * Compute and model the per-feature variances from a log-expression matrix with blocking, see `run_blocked()` for details.
     *
     * @tparam Value_ Data type of the matrix.
     * @tparam Index_ Integer type for the row/column indices.
     * @tparam Block_ Integer type, to hold the block IDs.
     *
     * @param mat Pointer to a feature-by-cells **tatami** matrix containing log-expression values.
     * @param[in] block Pointer to an array of block identifiers, see `run_blocked()` for details.
     *
     * @return A `BlockResults` object containing the results of the variance modelling in each block.
     * An average for each statistic is also computed by default unless `set_compute_average()` is set to `false`.
     */
    template<typename Value_, typename Index_, typename Block_>
    BlockResults run_blocked(const tatami::Matrix<Value_, Index_>* mat, const Block_* block) const {
        int nblocks = (block ? count_ids(mat->ncol(), block) : 1);
        BlockResults output(mat->nrow(), nblocks, compute_average);

        std::vector<double*> mean_ptr, var_ptr, fit_ptr, resid_ptr;
        fill_pointers(nblocks, output, mean_ptr, var_ptr, fit_ptr, resid_ptr);

        if (compute_average) {
            run_blocked(
                mat, 
                block, 
                std::move(mean_ptr), 
                std::move(var_ptr), 
                std::move(fit_ptr), 
                std::move(resid_ptr),
                output.average.means.data(), 
                output.average.variances.data(), 
                output.average.fitted.data(), 
                output.average.residuals.data() 
            );
        } else {
            run_blocked(
                mat, 
                block, 
                std::move(mean_ptr), 
                std::move(var_ptr), 
                std::move(fit_ptr), 
                std::move(resid_ptr)
            );
        }

        return output;
    }
};

}

#endif
