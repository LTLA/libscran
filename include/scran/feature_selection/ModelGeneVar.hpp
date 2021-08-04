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

private:

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
    void run(const MAT* p, Stat* means, Stat* variances, Stat* fitted, Stat* residuals) {
        run_blocked(p, static_cast<int*>(NULL), std::vector<Stat*>{means}, std::vector<Stat*>{variances}, std::vector<Stat*>{fitted}, std::vector<Stat*>{residuals});
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
    void run_blocked(const MAT* p, const B* block, std::vector<Stat*> means, std::vector<Stat*> variances, std::vector<Stat*> fitted, std::vector<Stat*> residuals) {
        size_t NR = p->nrow(), NC = p->ncol();
        std::vector<int> block_size(means.size());
        if (block) {
            auto copy = block;
            for (size_t j = 0; j < NC; ++j, ++copy) {
                ++block_size[*copy];
            }
            feature_selection::BlockedVarianceFactory<true, decltype(means), B, decltype(block_size)> fact(NR, NC, means, variances, block, block_size);
            tatami::apply<0>(p, fact);
        } else {
            block_size[0] = NC;
            feature_selection::BlockedVarianceFactory<false, decltype(means), B, decltype(block_size)> fact(NR, NC, means, variances, block, block_size);
            tatami::apply<0>(p, fact);
        }

        // Applying the trend fit to each block.
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
     */
    struct Results {
        /**
         * @param ngenes Number of genes.
         * @param nblocks Number of blocks.
         */
        Results(size_t ngenes, int nblocks) : means(nblocks * ngenes),
                                              variances(nblocks * ngenes),
                                              fitted(nblocks * ngenes),
                                              residuals(nblocks * ngenes) {}

        /**
         * Array holding a column-major matrix where each row corresponds to a gene and each column corresponds to a block (defaulting to a single block for `run()`).
         * Each entry contains the mean log-expression for each gene in each block.
         */
        std::vector<double> means;

        /**
         * Array holding a column-major matrix of the same shape as `means`.
         * Each entry contains the variance for each gene in each block.
         */
        std::vector<double> variances;

        /**
         * Array holding a column-major matrix of the same shape as `means`.
         * Each entry contains the fitted value for each gene in each block.
         */
        std::vector<double> fitted;

        /**
         * Array holding a column-major matrix of the same shape as `means`.
         * Each vector contains the residual for each gene in each block.
         */
        std::vector<double> residuals;
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
        run(mat, output.means.data(), output.variances.data(), output.fitted.data(), output.residuals.data());
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
            size_t offset = b * mat->nrow();
            mean_ptr.push_back(output.means.data() + offset);
            var_ptr.push_back(output.variances.data() + offset);
            fit_ptr.push_back(output.fitted.data() + offset);
            resid_ptr.push_back(output.residuals.data() + offset);
        }

        run_blocked(mat, block, std::move(mean_ptr), std::move(var_ptr), std::move(fit_ptr), std::move(resid_ptr));
        return output;
    }

    
private:
    FitTrendVar fit;
};

}

#endif
