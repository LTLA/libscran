#ifndef SCRAN_MODEL_GENE_VAR_H
#define SCRAN_MODEL_GENE_VAR_H

#include "tatami/base/Matrix.hpp"
#include "../utils/vector_to_pointers.hpp"
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

public:
    /**
     * @brief Results of the variance modelling.
     */
    struct Results {
        /**
         * @param ngenes Number of genes.
         * @param nblocks Number of blocks.
         */
        Results(size_t ngenes, int nblocks) : means(nblocks, std::vector<double>(ngenes)),
                                              variances(nblocks, std::vector<double>(ngenes)),
                                              fitted(nblocks, std::vector<double>(ngenes)),
                                              residuals(nblocks, std::vector<double>(ngenes)) {}

        /**
         * Vector of length equal to the number of blocks, containing vectors of length equal to the number of genes.
         * Each vector contains the mean for each gene in each block.
         */
        std::vector<std::vector<double> > means;

        /**
         * Vector of length equal to the number of blocks, containing vectors of length equal to the number of genes.
         * Each vector contains the variance for each gene in each block.
         */
        std::vector<std::vector<double> > variances;

        /**
         * Vector of length equal to the number of blocks, containing vectors of length equal to the number of genes.
         * Each vector contains the fitted value for each gene in each block.
         */
        std::vector<std::vector<double> > fitted;

        /**
         * Vector of length equal to the number of blocks, containing vectors of length equal to the number of genes.
         * Each vector contains the residual for each gene in each block.
         */
        std::vector<std::vector<double> > residuals;
    };

    /** 
     * Compute and model the per-feature variances from a log-expression matrix.
     *
     * @tparam MAT Type of matrix, usually a `tatami::NumericMatrix`.
     * @tparam BPTR Pointer to an integer type, to hold the block IDs.
     *
     * @param mat Pointer to a feature-by-cells **tatami** matrix containing log-expression values.
     * @param[in] block Pointer to an array of block identifiers.
     * If provided, the array should be of length equal to `ncells`.
     * Values should be integer IDs in \f$[0, N)\f$ where \f$N\f$ is the number of blocks.
     * This can also be a `nullptr`, in which case all cells are assumed to belong to the same block.
     *
     * @return A `Results` object containing the results of the variance modelling.
     */
    template<class MAT, typename BPTR>
    Results run(const MAT* mat, BPTR block) {
        int nblocks = 1;
        if constexpr(!std::is_same<BPTR, std::nullptr_t>::value) {
            if (mat->ncol()) {
                nblocks = *std::max_element(block, block + mat->ncol()) + 1;
            }
        }

        Results output(mat->nrow(), nblocks);
        run(mat, 
            block, 
            vector_to_pointers(output.means),
            vector_to_pointers(output.variances),
            vector_to_pointers(output.fitted),
            vector_to_pointers(output.residuals));

        return output;
    }

public:
    /** 
     * Compute and model the per-feature variances from a log-expression matrix.
     *
     * @tparam MAT Type of matrix, usually a `tatami::NumericMatrix`.
     * @tparam BPTR Pointer to an integer type, to hold the block IDs.
     *
     * @param mat Pointer to a feature-by-cells **tatami** matrix containing log-expression values.
     * @param[in] block Pointer to an array of block identifiers, see `run()` for details.
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
    template<class MAT, typename BPTR>
    void run(const MAT* p, 
             BPTR block,
             std::vector<double*> means,
             std::vector<double*> variances,
             std::vector<double*> fitted,
             std::vector<double*> residuals)
    {
        // Estimating the raw values.
        size_t NR = p->nrow(), NC = p->ncol();
        size_t nblocks = means.size();
        std::vector<int> block_size(nblocks);

        if constexpr(!std::is_same<BPTR, std::nullptr_t>::value) {
            auto copy = block;
            for (size_t j = 0; j < NC; ++j, ++copy) {
                ++block_size[*copy];
            }
        } else {
            block_size[0] = NC;
        }

        if (p->prefer_rows()) {
            std::vector<typename MAT::value> obuffer(NC);
            auto wrk = p->new_workspace(true);

            // We use temporary buffers to improve memory locality for frequent
            // write operations, before transferring the result to the actual stores.
            std::vector<double> tmp_means(nblocks), tmp_vars(nblocks);

            auto sum2mean = [&]() -> void {
                for (size_t b = 0; b < nblocks; ++b) {
                    if (block_size[b]) {
                        tmp_means[b] /= block_size[b];
                    } else {
                        tmp_means[b] = std::numeric_limits<double>::quiet_NaN();
                    }
                }
            };

            if (p->sparse()) {
                std::vector<int> tmp_nzero(nblocks);
                std::vector<typename MAT::index> ibuffer(NC);

                for (size_t i = 0; i < NR; ++i) {
                    auto range = p->sparse_row(i, obuffer.data(), ibuffer.data(), wrk.get());
                    std::fill(tmp_means.begin(), tmp_means.end(), 0);
                    std::fill(tmp_vars.begin(), tmp_vars.end(), 0);
                    std::fill(tmp_nzero.begin(), tmp_nzero.end(), 0);

                    auto get_block = [&](size_t j) -> int {
                        if constexpr(std::is_same<BPTR, std::nullptr_t>::value) {
                            return 0;
                        } else {
                            return block[range.index[j]];
                        }
                    };

                    for (size_t j = 0; j < range.number; ++j) {
                        int b = get_block(j);
                        tmp_means[b] += range.value[j];
                        ++tmp_nzero[b];
                    }

                    sum2mean();

                    for (size_t j = 0; j < range.number; ++j) {
                        auto b = get_block(j);
                        tmp_vars[b] += (range.value[j] - tmp_means[b]) * (range.value[j] - tmp_means[b]);
                    }

                    for (size_t b = 0; b < nblocks; ++b) {
                        means[b][i] = tmp_means[b];
                        variances[b][i] = tmp_vars[b] + tmp_means[b] * tmp_means[b] * (block_size[b] - tmp_nzero[b]);
                    }
                }
            } else {
                for (size_t i = 0; i < NR; ++i) {
                    auto ptr = p->row(i, obuffer.data(), wrk.get());
                    std::fill(tmp_means.begin(), tmp_means.end(), 0);
                    std::fill(tmp_vars.begin(), tmp_vars.end(), 0);

                    auto get_block = [&](size_t j) -> int {
                        if constexpr(std::is_same<BPTR, std::nullptr_t>::value) {
                            return 0;
                        } else {
                            return block[j];
                        }
                    };

                    for (size_t j = 0; j < NC; ++j) {
                        auto b = get_block(j);
                        tmp_means[b] += ptr[j];
                    }

                    sum2mean();

                    for (size_t j = 0; j < NC; ++j) {
                        auto b = get_block(j);
                        tmp_vars[b] += (ptr[j] - tmp_means[b]) * (ptr[j] - tmp_means[b]);
                    }

                    for (size_t b = 0; b < nblocks; ++b) {
                        means[b][i] = tmp_means[b];
                        variances[b][i] = tmp_vars[b];
                    }
                }
            }

            // Dividing by the relevant denominators.
            for (size_t b = 0; b < nblocks; ++b) {
                if (block_size[b] < 2) {
                    std::fill(variances[b], variances[b] + NR, std::numeric_limits<double>::quiet_NaN());
                } else {
                    double denominator = block_size[b] - 1;
                    for (size_t i = 0; i < NR; ++i) {
                        variances[b][i] /= denominator;
                    }
                }
            }

        } else {
            std::vector<typename MAT::value> obuffer(NR);
            auto wrk = p->new_workspace(false);

            for (size_t b = 0; b < nblocks; ++b) {
                std::fill(means[b], means[b] + NR, 0);
                std::fill(variances[b], variances[b] + NR, 0);
                std::fill(fitted[b], fitted[b] + NR, 0);
            }

            auto get_block = [&](size_t i) -> int {
                if constexpr(std::is_same<BPTR, std::nullptr_t>::value) {
                    return 0;
                } else {
                    return block[i];
                }
            };

            if (p->sparse()) {
                std::vector<typename MAT::index> ibuffer(NR);
                std::vector<tatami::stats::VarianceHelper::Sparse> running(nblocks, tatami::stats::VarianceHelper::Sparse(NR));

                for (size_t i = 0; i < NC; ++i) {
                    auto range = p->sparse_column(i, obuffer.data(), ibuffer.data(), wrk.get());
                    auto b = get_block(i);
                    running[b].add(range);
                }

                for (size_t b = 0; b < nblocks; ++b) {
                    running[b].finish();
                    const auto& running_vars = running[b].statistics();
                    std::copy(running_vars.begin(), running_vars.end(), variances[b]);
                    const auto& running_means = running[b].means();
                    std::copy(running_means.begin(), running_means.end(), means[b]);
                }

            } else {
                std::vector<tatami::stats::VarianceHelper::Dense> running(nblocks, tatami::stats::VarianceHelper::Dense(NR));

                for (size_t i = 0; i < NC; ++i) {
                    auto ptr = p->column(i, obuffer.data(), wrk.get());
                    auto b = get_block(i);
                    running[b].add(ptr);
                }

                for (size_t b = 0; b < nblocks; ++b) {
                    running[b].finish();
                    const auto& running_vars = running[b].statistics();
                    std::copy(running_vars.begin(), running_vars.end(), variances[b]);
                    const auto& running_means = running[b].means();
                    std::copy(running_means.begin(), running_means.end(), means[b]);
                }
            }
        }

        // Applying the trend fit to each block.
        for (size_t b = 0; b < nblocks; ++b) {
            if (block_size[b] >= 2) {
                fit.run(NR, means[b], variances[b], fitted[b], residuals[b]);
            } else {
                std::fill(fitted[b], fitted[b] + NR, std::numeric_limits<double>::quiet_NaN());
                std::fill(residuals[b], residuals[b] + NR, std::numeric_limits<double>::quiet_NaN());
            }
        }

        return;
    }
    
private:
    FitTrendVar fit;
};

}

#endif
