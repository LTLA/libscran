#ifndef SCRAN_LOG_NORM_COUNTS_H
#define SCRAN_LOG_NORM_COUNTS_H

#include <algorithm>
#include <vector>
#include <numeric>

#include "tatami/base/DelayedIsometricOp.hpp"
#include "tatami/stats/sums.hpp"

#include "CenterSizeFactors.hpp"
#include "../utils/block_indices.hpp"

/**
 * @file LogNormCounts.hpp
 *
 * @brief Compute log-normalized expression values.
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

    /** 
     * @param b Blocking mode, see `CenterSizeFactors::set_block_mode()` for details.
     *
     * @return A reference to this `LogNormCounts` object.
     */
    LogNormCounts& set_block_mode(CenterSizeFactors::BlockMode b = CenterSizeFactors::Defaults::block_mode) {
        centerer.set_block_mode(b);
        return *this;
    }

public:
    /**
     * Compute log-normalized expression values from an input matrix.
     * To avoid copying the data, this is done in a delayed manner using the `DelayedIsometricOp` class from the **tatami** package.
     *
     * @tparam MAT A **tatami** matrix class, most typically a `tatami::NumericMatrix`.
     * @tparam V A vector class supporting `size()`, random access via `[`, `begin()`, `end()` and `data()`.
     *
     * @param mat Pointer to an input count matrix, with features in the rows and cells in the columns.
     * @param size_factors A vector of positive size factors, of length equal to the number of columns in `mat`.
     *
     * @return A pointer to a matrix of log-transformed and normalized values.
     */
    template<class MAT, class V>
    std::shared_ptr<MAT> run(std::shared_ptr<MAT> mat, V size_factors) {
        return run_blocked(std::move(mat), std::move(size_factors), static_cast<int*>(NULL));
    }

    /**
     * Compute log-normalized expression values from an input matrix with blocking.
     * Specifically, centering of size factors is performed within each block.
     * This allows users to easily mimic normalization of different blocks of cells (e.g., from different samples) in the same matrix.
     *
     * @tparam MAT A **tatami** matrix class, most typically a `tatami::NumericMatrix`.
     * @tparam V A vector class supporting `size()`, random access via `[`, `begin()`, `end()` and `data()`.
     * @tparam B An integer type, to hold the block IDs.
     *
     * @param mat Pointer to an input count matrix, with features in the rows and cells in the columns.
     * @param size_factors A vector of positive size factors, of length equal to the number of columns in `mat`.
     * @param[in] block Pointer to an array of block identifiers.
     * If provided, the array should be of length equal to the number of columns in `mat`.
     * Values should be integer IDs in \f$[0, N)\f$ where \f$N\f$ is the number of blocks.
     * This can also be a `NULL`, in which case all cells are assumed to belong to the same block.
     *
     * @return A pointer to a matrix of log-transformed and normalized values.
     */
    template<class MAT, class V, typename B>
    std::shared_ptr<MAT> run_blocked(std::shared_ptr<MAT> mat, V size_factors, const B* block) {
        // One might ask why we don't require a pointer for size_factors here.
        // It's because size_factors need to be moved into the Delayed operation
        // anyway, so we might as well ask the user to construct a vector for us.
        if (size_factors.size() != mat->ncol()) {
            throw std::runtime_error("number of size factors and columns are not equal");
        }

        if (center) {
#ifdef SCRAN_LOGGER
            SCRAN_LOGGER("scran::LogNormCounts", "Centering size factors to unity");
#endif
            // Falls back to centerer.run() if block=NULL.
            centerer.run_blocked(size_factors.size(), size_factors.data(), block);
        } else {
            // CenterSizeFactors do their own validity checks, 
            // so we don't need to call it again in that case.
            CenterSizeFactors::validate(size_factors.size(), size_factors.data());
        }

#ifdef SCRAN_LOGGER
        SCRAN_LOGGER("scran::LogNormCounts", "Dividing each cell by its size factor");
#endif
        auto div = tatami::make_DelayedIsometricOp(mat, tatami::make_DelayedDivideVectorHelper<true, 1>(std::move(size_factors)));

#ifdef SCRAN_LOGGER
        SCRAN_LOGGER("scran::LogNormCounts", "Applying the log-transformation");
#endif
        if (pseudo_count == 1) {
            return tatami::make_DelayedIsometricOp(div, tatami::DelayedLog1pHelper(2.0));
        } else {
            auto add = tatami::make_DelayedIsometricOp(div, tatami::DelayedAddScalarHelper<double>(pseudo_count));
            return tatami::make_DelayedIsometricOp(add, tatami::DelayedLogHelper(2.0));
        }
    }

public:
    /**
     * Compute log-normalized expression values from an input matrix.
     * Size factors are defined as the sum of the total counts for each cell. 
     *
     * @tparam MAT A **tatami** matrix class, most typically a `tatami::NumericMatrix`.
     *
     * @param mat Pointer to an input count matrix, with features in the rows and cells in the columns.
     *
     * @return A pointer to a matrix of log-transformed and normalized values.
     */
    template<class MAT>
    std::shared_ptr<MAT> run(std::shared_ptr<MAT> mat) {
        auto size_factors = tatami::column_sums(mat.get());
        return run_blocked(std::move(mat), std::move(size_factors), static_cast<int*>(NULL));
    }

    /**
     * Compute log-normalized expression values from an input matrix with blocking, see `run_blocked()` for details.
     * Size factors are defined as the sum of the total counts for each cell. 
     *
     * @tparam MAT A **tatami** matrix class, most typically a `tatami::NumericMatrix`.
     * @tparam B An integer type, to hold the block IDs.
     *
     * @param mat Pointer to an input count matrix, with features in the rows and cells in the columns.
     * @param[in] block Pointer to an array of block identifiers, see `run_blocked()` for details.
     *
     * @return A pointer to a matrix of log-transformed and normalized values.
     */
    template<class MAT, typename B>
    std::shared_ptr<MAT> run_blocked(std::shared_ptr<MAT> mat, const B* block) {
        auto size_factors = tatami::column_sums(mat.get());
        return run_blocked(mat, std::move(size_factors), block);
    }

private:
    double pseudo_count = 1;
    bool center = true;
    CenterSizeFactors centerer;
};

};

#endif
