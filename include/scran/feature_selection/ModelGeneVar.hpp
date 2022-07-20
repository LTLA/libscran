#ifndef SCRAN_MODEL_GENE_VAR_H
#define SCRAN_MODEL_GENE_VAR_H

#include "tatami/base/Matrix.hpp"
#include "tatami/stats/apply.hpp"

#include "../utils/vector_to_pointers.hpp"
#include "blocked_variances.hpp"
#include "FitTrendVar.hpp"

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

    /** 
     * Use the default span for the LOWESS smoother, see https://ltla.github.io/CppWeightedLowess for details.
     *
     * @return A reference to this `ModelGeneVar` object.
     */
    ModelGeneVar& set_span() {
        fit.set_span();
        return *this;
    }

    /** 
     * Set the span for `FitTrendVar`.
     *
     * @param s Span for the smoother, as a proportion of the total number of points.
     *
     * @return A reference to this `ModelGeneVar` object.
     */
    ModelGeneVar& set_span(double s) {
        fit.set_span(s);
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
    void run(const MAT* mat, Stat* means, Stat* variances, Stat* fitted, Stat* residuals) {
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
    void run_blocked(const MAT* mat, const B* block, std::vector<Stat*> means, std::vector<Stat*> variances, std::vector<Stat*> fitted, std::vector<Stat*> residuals) {
        size_t NR = mat->nrow(), NC = mat->ncol();
        std::vector<int> block_size(means.size());

#ifdef SCRAN_LOGGER
        SCRAN_LOGGER("scran::ModelGeneVar", "Estimating mean and variance of each gene");
#endif

        if (block) {
            auto copy = block;
            for (size_t j = 0; j < NC; ++j, ++copy) {
                ++block_size[*copy];
            }
            feature_selection::BlockedVarianceFactory<true, Stat, B, decltype(block_size)> fact(NR, NC, means, variances, block, &block_size);
            tatami::apply<0>(mat, fact, num_threads);
        } else {
            block_size[0] = NC;
            feature_selection::BlockedVarianceFactory<false, Stat, B, decltype(block_size)> fact(NR, NC, means, variances, block, &block_size);
            tatami::apply<0>(mat, fact, num_threads);
        }

        // Applying the trend fit to each block.
#ifdef SCRAN_LOGGER
        SCRAN_LOGGER("scran::ModelGeneVar", "Fitting a mean-variance trend across genes");
#endif

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
    Results run(const MAT* mat) {
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
    Results run_blocked(const MAT* mat, const B* block) {
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
    
private:
    FitTrendVar fit;
    int num_threads = Defaults::num_threads;
};

}

#endif
