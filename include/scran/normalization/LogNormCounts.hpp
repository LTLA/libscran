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
 * This is done in a delayed manner using the `DelayedIsometricOp` class from the **tatami** library.
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
     * Define blocks of cells, where centering of size factors is performed within each block.
     * This allows users to easily mimic normalization of different blocks of cells (e.g., from different samples) in the same matrix.
     * In contrast, without blocking, the centering would depend on the size factors in different blocks.
     *
     * @tparam V Class of the blocking vector.
     * @param p Blocking vector of length equal to the number of cells.
     * Values should be integer block IDs in \f$[0, N)\f$ where \f$N\f$ is the number of blocks.
     *
     * @return A reference to this `LogNormCounts` object. 
     */
    template<class V>
    LogNormCounts& set_blocks(const V& p) {
        return set_blocks(p.size(), p.begin());
    }

    /**
     * Define blocks of cells, see `set_blocks(const V& p)`.
     *
     * @tparam SIT Iterator class for the blocking vector.
     *
     * @param n Length of the blocking vector, should be equal to the number of cells.
     * @param p Pointer or iterator to the start of the blocking vector.
     *
     * @return A reference to this `LogNormCounts` object. 
     */
    template<typename SIT>
    LogNormCounts& set_blocks(size_t n, SIT b) {
        block_indices(n, b, by_group);
        group_ncells = n;
        return *this;
    }

    /**
     * Unset any previous blocking structure.
     *
     * @return A reference to this `PerCellQCFilters` object. 
     */
    LogNormCounts& set_blocks() {
        by_group.clear();
        group_ncells = 0;
        return *this;
    }

public:
    /**
     * Compute log-normalized expression values from an input matrix.
     * To avoid copying the data, this is done in a delayed manner using the `DelayedIsometricOp` class from the **tatami** package.
     *
     * @tparam A `tatami::typed_matrix`, most typically a `tatami::numeric_matrix`.
     * @tparam V A vector class supporting `size()`, random access via `[`, `begin()`, `end()` and `data()`.
     *
     * @param mat Pointer to an input count matrix, with features in the rows and cells in the columns.
     * @param size_factors A vector of positive size factors, of length equal to the number of columns in `mat`.
     *
     * @return A pointer to a matrix of log-transformed and normalized values.
     */
    template<class MAT, class V>
    std::shared_ptr<MAT> run(std::shared_ptr<MAT> mat, V size_factors) {
        if (size_factors.size() != mat->ncol()) {
            throw std::runtime_error("number of size factors and columns are not equal");
        }

        if (center) {
            if (by_group.size()) {
                if (group_ncells != mat->ncol()) {
                    throw std::runtime_error("length of grouping vector and number of columns are not equal");
                }
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
            } else if (size_factors.size()) {
                double mean = std::accumulate(size_factors.begin(), size_factors.end(), static_cast<double>(0)) / size_factors.size();
                if (mean) {
                    for (auto& x : size_factors) {
                        x /= mean;
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

    std::vector<std::vector<size_t> > by_group;
    size_t group_ncells = 0;
};

};

#endif
