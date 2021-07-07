#ifndef SCRAN_PER_CELL_QC_METRICS_HPP
#define SCRAN_PER_CELL_QC_METRICS_HPP

#include <vector>
#include <algorithm>
#include <limits>
#include <cstdint>

#include "tatami/base/typed_matrix.hpp"
#include "../utils/vector_to_pointers.hpp"

/**
 * @file PerCellQCMetrics.hpp
 *
 * Compute typical per-cell quality control metrics.
 */

namespace scran {

/**
 * @brief Compute typical per-cell quality control metrics.
 *
 * @tparam X Type of the boolean flag to indicate whether a feature belongs to a feature subset.
 *
 * Given a feature-by-cell count matrix, this class computes several QC metrics:
 * 
 * - The total sum for each cell quantifies the efficiency of library preparation and sequencing.
 * - The number of detected features also quantifies the library preparation efficiency,
 *   but with a greater focus on capturing the transcriptional complexity.
 * - The interpretation of the subset proportions depend on the subsets.
 *   In the most common case of mitochondrial transcripts, higher proportions indicate cell damage.
 *   Spike-in proportions can be interpreted in a similar manner.
 */
template<typename X=uint8_t>
class PerCellQCMetrics {
public:
    /**
     * Default constructor.
     */
    PerCellQCMetrics() {}

    /**
     * @param s A vector of pointers to boolean arrays.
     * Each array represents a feature subset and is of length equal to the number of features.
     * Each value should be non-zero if the corresponding feature belongs to the subset; otherwise, it should be zero.
     */
    PerCellQCMetrics(const std::vector<X*>& s) : subsets(s.begin(), s.end()) {}

    /**
     * @copydoc PerCellQCMetrics(const std::vector<X*>&)
     */
    PerCellQCMetrics(std::vector<const X*> s) : subsets(std::move(s)) {}

public:
    /**
     * Define feature subsets of interest.
     * This is most commonly used to define mitochondrial transcripts or ribosomal genes.
     *
     * @param s A vector of pointers to boolean arrays indicating which features belong in which feature subsets.
     * Each array represents a feature subset and is of length equal to the number of features.
     * Each value should be non-zero if the corresponding feature belongs to the subset; otherwise, it should be zero.
     *
     * @return A reference to this `PerCellQCMetrics` object.
     */
    PerCellQCMetrics& set_subsets(std::vector<const X*> s) {
        subsets = s;
        return *this;
    }

public:
    /**
     * @brief Result store for QC metric calculations.
     */
    struct Results {
        /**
         * @param ncells Number of cells, i.e., the number of columns in the input matrix.
         * @param nsubsets Number of feature subsets of interest.
         */
        Results(size_t ncells, size_t nsubsets) : sums(ncells), detected(ncells), subset_proportions(nsubsets, std::vector<double>(ncells)) {}

        /**
         * Sum of counts for each cell.
         */
        std::vector<double> sums;

        /**
         * Number of detected features in each cell.
         */
        std::vector<int> detected;

        /**
         * Proportion of counts in each feature subset in each cell.
         * Each inner vector corresponds to a feature subset and is of length equal to the number of cells.
         */
        std::vector<std::vector<double> > subset_proportions;
    };

public:
    /**
     * Compute the QC metrics from an input matrix and return the results.
     *
     * @param mat Pointer to a feature-by-cells matrix.
     *
     * @return A `PerCellQCMetrics::Results` object containing the QC metrics.
     * Subset proportions are returned depending on the subsets defined at construction or by `set_subsets()`.
     */
    template<typename T, typename IDX>
    Results run(const tatami::typed_matrix<T, IDX>* mat) {
        Results output(mat->ncol(), subsets.size());
        run(mat, output.sums.data(), output.detected.data(), vector_to_pointers(output.subset_proportions));
        return output;
    }

    /**
     * Compute the QC metrics from an input matrix and return the results.
     *
     * @param mat Pointer to a feature-by-cells matrix.
     * @param[out] sums Pointer to a (typically double-precision) array of length equal to the number of columns in `mat`.
     * This is used to store the computed sums for all cells.
     * @param[out] detected Pointer to a (typically integer) array of length equal to the number of columns in `mat`.
     * This is used to store the number of detected features for all cells.
     * @param[out] subset_proportions Vector of pointers (typically double-precision) arrays of length equal to the number of columns in `mat`.
     * Each array corresponds to a feature subset and is used to store the proportion of counts in that subset across all cells.
     *
     * @return `sums`, `detected`, and each array in `subset_proportions` is filled with the relevant statistics.
     */
    template<typename T, typename IDX, typename S, typename D, typename P>
    void run(const tatami::typed_matrix<T, IDX>* mat, S* sums, D* detected, std::vector<P*> subset_proportions) {
        size_t nr = mat->nrow(), nc = mat->ncol();

        if (mat->prefer_rows()) {
            std::vector<T> buffer(nc);
            std::fill(sums, sums + nc, static_cast<S>(0));
            std::fill(detected, detected + nc, static_cast<D>(0));
            for (size_t s = 0; s < subsets.size(); ++s) {
                std::fill(subset_proportions[s], subset_proportions[s] + nc, static_cast<P>(0));
            }
            auto wrk = mat->new_workspace(false);

            if (mat->sparse()) {
                std::vector<IDX> ibuffer(nc);
                for (size_t r = 0; r < nr; ++r) {
                    auto range = mat->sparse_row(r, buffer.data(), ibuffer.data(), wrk.get());

                    for (size_t i = 0; i < range.number; ++i) {
                        sums[range.index[i]] += range.value[i];
                        detected[range.index[i]] += static_cast<D>(range.value[i] > 0);
                    }

                    for (size_t s = 0; s < subsets.size(); ++s) {
                        if (subsets[s][r]) {
                            for (size_t i = 0; i < range.number; ++i) {
                                subset_proportions[s][range.index[i]] += range.value[i];
                            }
                        }
                    }
                }
                
            } else {
                for (size_t r = 0; r < nr; ++r) {
                    auto ptr = mat->row(r, buffer.data(), wrk.get());

                    for (size_t c = 0; c < nc; ++c) {
                        sums[c] += ptr[c];
                        detected[c] += static_cast<D>(ptr[c] > 0);
                    }

                    for (size_t s = 0; s < subsets.size(); ++s) {
                        if (subsets[s][r]) {
                            auto& sub = subset_proportions[s];
                            for (size_t c = 0; c < nc; ++c) {
                                sub[c] += ptr[c];
                            }
                        }
                    }
                }
            }

        } else {
            std::vector<T> buffer(nr);
            auto wrk = mat->new_workspace(true);

            if (mat->sparse()) {
                std::vector<IDX> ibuffer(nr);
                for (size_t c = 0; c < nc; ++c) {
                    auto range = mat->sparse_column(c, buffer.data(), ibuffer.data(), wrk.get());

                    sums[c] = 0;
                    detected[c] = 0;
                    for (size_t r = 0; r < range.number; ++r) {
                        sums[c] += range.value[r];
                        detected[c] += static_cast<D>(range.value[r] > 0);
                    }

                    for (size_t s = 0; s < subsets.size(); ++s) {
                        const auto& sub = subsets[s];
                        auto& prop = subset_proportions[s][c] = 0;
                        for (size_t i = 0; i < range.number; ++i) {
                            prop += sub[range.index[i]] * range.value[i];
                        }
                    }
                }

            } else {
                for (size_t c = 0; c < nc; ++c) {
                    auto ptr = mat->column(c, buffer.data(), wrk.get());

                    sums[c] = 0;
                    detected[c] = 0;
                    for (size_t r = 0; r < nr; ++r) {
                        sums[c] += ptr[r];
                        detected[c] += static_cast<D>(ptr[r] > 0);
                    }

                    for (size_t s = 0; s < subsets.size(); ++s) {
                        const auto& sub = subsets[s];
                        auto& prop = subset_proportions[s][c] = 0;
                        for (size_t r = 0; r < nr; ++r) {
                            prop += sub[r] * ptr[r];
                        }
                    }
                }
            }
        }

        for (size_t s = 0; s < subsets.size(); ++s) {
            auto& prop = subset_proportions[s];
            for (size_t c = 0; c < nc; ++c) {
                if (sums[c]!=0) {
                    prop[c] /= sums[c];
                } else {
                    prop[c] = std::numeric_limits<double>::quiet_NaN();
                }
            }
        }

        return;
    }

private:
    std::vector<const X*> subsets;
};

}

#endif
