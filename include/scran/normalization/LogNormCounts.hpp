#ifndef SCRAN_LOG_NORM_COUNTS_H
#define SCRAN_LOG_NORM_COUNTS_H

#include <algorithm>
#include <vector>
#include <numeric>

#include "tatami/base/DelayedIsometricOp.hpp"

#include "../utils/block_indices.hpp"

/**
 * @file LogNormCounts.hpp
 *
 * Compute log-normalized expression values.
 */

namespace scran {

/**
 * @brief Compute log-normalized expression values.
 *
 * Given a count matrix and a set of size factors, compute log-transformed normalized expression values.
 * Each cell's counts are divided by the cell's size factor, to account for differences in capture efficiency and sequencing depth across cells.
 * The normalized values are then log-transformed so that downstream analyses focus on the relative rather than absolute differences in expression;
 * this process also provides some measure of variance stabilization.
 * These operations are done in a delayed manner using the `DelayedIsometricOp` class from the **tatami** library.
 */
class LogNormCounts {
public:
    /** 
     * Set the pseudo-count for the log-transformation.
     * This avoids problems with undefined values at zero counts.
     *
     * @param p Pseudo-count, should be a positive number.
     *
     * @return A reference to this `LogNormCounts` object.
     */
    LogNormCounts& set_pseudo_count (double p = 1) {
        pseudo_count = p;
        return *this;
    }

    /** 
     * Specify whether to center the size factors in `run()`.
     * If `true`, we center the size factors across cells so that their average is equal to 1;
     * this ensures that the normalized values can still be interpreted on the same scale as the input counts.
     *
     * If `false`, no further centering is performed.
     * This is more efficient when size factors are already centered;
     * it may also be useful for re-using this class to compute other normalized values like log-CPMs.
     *
     * @param c Whether to center the size factors.
     *
     * @return A reference to this `LogNormCounts` object.
     */
    LogNormCounts& set_center(bool c = true) {
        center = c;
        return *this;
    }

public:
    /**
     * Compute log-normalized expression values from an input matrix.
     * To avoid copying the data, this is done in a delayed manner using the `DelayedIsometricOp` class from the **tatami** package.
     *
     * If `block` is specified, centering of size factors is performed within each block.
     * This allows users to easily mimic normalization of different blocks of cells (e.g., from different samples) in the same matrix.
     * In contrast, without blocking, the centering would depend on the size factors in different blocks.
     *
     * @tparam A `tatami::typed_matrix`, most typically a `tatami::numeric_matrix`.
     * @tparam V A vector class supporting `size()`, random access via `[`, `begin()`, `end()` and `data()`.
     * @tparam BPTR Pointer to an integer type, to hold the block IDs.
     *
     * @param mat Pointer to an input count matrix, with features in the rows and cells in the columns.
     * @param size_factors A vector of positive size factors, of length equal to the number of columns in `mat`.
     * @param[in] block Pointer to an array of block identifiers.
     * If provided, the array should be of length equal to the number of columns in `mat`.
     * Values should be integer IDs in \f$[0, N)\f$ where \f$N\f$ is the number of blocks.
     * This can also be a `nullptr`, in which case all cells are assumed to belong to the same block.
     *
     * @return A pointer to a matrix of log-transformed and normalized values.
     */
    template<class MAT, class V, typename BPTR>
    std::shared_ptr<MAT> run(std::shared_ptr<MAT> mat, V size_factors, BPTR block) {
        if (size_factors.size() != mat->ncol()) {
            throw std::runtime_error("number of size factors and columns are not equal");
        }

        if (center) {
            if constexpr(!std::is_same<BPTR, std::nullptr_t>::value) {
                auto by_group = block_indices(mat->ncol(), block);
                for (const auto& g : by_group) {
                    if (g.size()) {
                        double mean = 0;
                        for (auto i : g) {
                            mean += size_factors[i];
                        }
                        mean /= g.size();

                        if (mean > 0) {
                            for (auto i : g) {
                                size_factors[i] /= mean;
                            }
                        }
                    }
                }
            } else {
                if (size_factors.size()) { // avoid division by zero
                    double mean = std::accumulate(size_factors.begin(), size_factors.end(), static_cast<double>(0)) / size_factors.size();
                    if (mean) { // avoid division by zero
                        for (auto& x : size_factors) {
                            x /= mean;
                        }
                    }
                }
            }
        }

        for (auto x : size_factors) {
            if (x <= 0) {
                throw std::runtime_error("non-positive size factors detected");
            }
        }

        auto div = tatami::make_DelayedIsometricOp(mat, tatami::make_DelayedDivideVectorHelper<true, 1>(std::move(size_factors)));
        if (pseudo_count == 1) {
            return tatami::make_DelayedIsometricOp(div, tatami::DelayedLog1pHelper(2.0));
        } else {
            auto add = tatami::make_DelayedIsometricOp(div, tatami::DelayedAddScalarHelper<double>(pseudo_count));
            return tatami::make_DelayedIsometricOp(add, tatami::DelayedLogHelper(2.0));
        }
    }

private:
    double pseudo_count = 1;
    bool center = true;
};

};

#endif
