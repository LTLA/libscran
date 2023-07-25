#ifndef SCRAN_LOG_NORM_COUNTS_H
#define SCRAN_LOG_NORM_COUNTS_H

#include "../utils/macros.hpp"

#include <algorithm>
#include <vector>
#include <numeric>

#include "tatami/tatami.hpp"

#include "utils.hpp"
#include "CenterSizeFactors.hpp"
#include "ChoosePseudoCount.hpp"

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
     * @brief Default parameter settings.
     */
    struct Defaults {
        /**
         * See `set_pseudo_count()` for more details.
         */
        static constexpr double pseudo_count = 1;

        /**
         * See `set_sparse_addition()` for more details.
         */
        static constexpr bool sparse_addition = true;

        /**
         * See `set_choose_pseudo_count()` for more details.
         */
        static constexpr bool choose_pseudo_count = false;

        /**
         * See `set_center()` for more details.
         */
        static constexpr bool center = true;

        /**
         * Set `set_handle_zeros()` for more details.
         */
        static constexpr bool handle_zeros = false;

        /**
         * Set `set_handle_non_finite()` for more details.
         */
        static constexpr bool handle_non_finite = false;

        /**
         * See `set_num_threads()` for more details.
         */
        static constexpr int num_threads = 1;
    };

private:
    double pseudo_count = Defaults::pseudo_count;
    bool sparse_addition = Defaults::sparse_addition;
    bool handle_zeros = Defaults::handle_zeros;
    bool handle_non_finite = Defaults::handle_non_finite;
    int nthreads = Defaults::num_threads;

    bool center = Defaults::center;
    CenterSizeFactors centerer;

    bool choose_pseudo_count = Defaults::choose_pseudo_count;
    ChoosePseudoCount pseudo_chooser;

public:
    /** 
     * Set the pseudo-count to add to the normalized expression values prior to the log-transformation.
     * Larger pseudo-counts will shrink the log-expression values towards zero such that the dataset variance is driven more by high-abundance genes;
     * this is occasionally useful to mitigate biases introduced by log-expression at low counts.
     * See also `set_choose_pseudo_count()`.
     *
     * @param p Pseudo-count, should be a positive number.
     *
     * @return A reference to this `LogNormCounts` object.
     */
    LogNormCounts& set_pseudo_count (double p = Defaults::pseudo_count) {
        pseudo_count = p;
        return *this;
    }

    /** 
     * Naive addition of a non-unity pseudo-count will break sparsity.
     * This can be avoided by instead dividing the normalized expression values by the pseudo-count and then applying the usual `log1p` transformation.
     * However, the resulting values can not be interpreted on the scale of log-counts.
     *
     * @param a Whether to use an effective pseudo-count that avoids breaking sparsity.
     *
     * @return A reference to this `LogNormCounts` object.
     */
    LogNormCounts& set_sparse_addition(bool a = Defaults::sparse_addition) {
        sparse_addition = a;
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
    LogNormCounts& set_center(bool c = Defaults::center) {
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

    /**
     * Specify whether to handle zero size factors. 
     * If false, size factors of zero will raise an error;
     * otherwise, they will be automatically set to the smallest non-zero size factor after centering (or 1, if all size factors are zero).
     * Setting this to `true` ensures that any all-zero cells are represented by all-zero columns in the normalized matrix,
     * which is a reasonable outcome if those cells cannot be filtered out during upstream quality control.
     * Note that the centering process ignores zeros, see `CenterSizeFactors::set_ignore_zeros()` for more details.
     *
     * @param z Whether to replace zero size factors with the smallest non-zero size factor.
     *
     * @return A reference to this `LogNormCounts` object.
     */
    LogNormCounts& set_handle_zeros(bool z = Defaults::handle_zeros) {
        handle_zeros = z;
        return *this;
    }

    /**
     * Specify whether to handle non-finite size factors. 
     * If false, non-finite size factors will raise an error.
     * Otherwise, size factors of infinity will be automatically set to the largest finite size factor after centering (or 1, if all size factors are non-finite).
     * Missing (i.e., NaN) size factors will be automatically set to 1 so that scaling is a no-op.
     * Note that the centering process ignores non-finite factors, see `CenterSizeFactors` for more details.
     *
     * @param z Whether to replace non-finite size factors with the largest finite size factor.
     *
     * @return A reference to this `LogNormCounts` object.
     */
    LogNormCounts& set_handle_non_finite(bool n = Defaults::handle_non_finite) {
        handle_non_finite = n;
        return *this;
    }

    /**
     * @param n Number of threads to use. 
     * @return A reference to this `LogNormCounts` object.
     *
     * Parallelization is only performed to compute size factors,
     * so this method only has an effect if `size_factors` are not passed to `run()`.
     */
    LogNormCounts& set_num_threads(int n = Defaults::num_threads) {
        nthreads = n;
        return *this;
    }

public:
    /** 
     * @param c Whether to automatically choose an appropriate pseudo-count based on the (centered) size factors.
     * See `ChoosePseudoCount` for details.
     *
     * @return A reference to this `LogNormCounts` object.
     */
    LogNormCounts& set_choose_pseudo_count(bool c = Defaults::choose_pseudo_count) {
        choose_pseudo_count = c;
        return *this;
    }

    /** 
     * @param m See `ChoosePseudoCount::set_max_bias()` for details.
     *
     * @return A reference to this `LogNormCounts` object.
     */
    LogNormCounts& set_max_bias(double m = ChoosePseudoCount::Defaults::max_bias) {
        pseudo_chooser.set_max_bias(m);
        return *this;
    }

    /** 
     * @param q See `ChoosePseudoCount::set_quantile()` for details.
     *
     * @return A reference to this `LogNormCounts` object.
     */
    LogNormCounts& set_quantile(double q = ChoosePseudoCount::Defaults::quantile) {
        pseudo_chooser.set_quantile(q);
        return *this;
    }

    /** 
     * @param m See `ChoosePseudoCount::set_min_value()` for details.
     *
     * @return A reference to this `LogNormCounts` object.
     */
    LogNormCounts& set_min_value(double m = ChoosePseudoCount::Defaults::min_value) {
        pseudo_chooser.set_min_value(m);
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
    std::shared_ptr<MAT> run(std::shared_ptr<MAT> mat, V size_factors) const {
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
    std::shared_ptr<MAT> run_blocked(std::shared_ptr<MAT> mat, V size_factors, const B* block) const {
        // One might ask why we don't require a pointer for size_factors here.
        // It's because size_factors need to be moved into the Delayed operation
        // anyway, so we might as well ask the user to construct a vector for us.
        if (size_factors.size() != mat->ncol()) {
            throw std::runtime_error("number of size factors and columns are not equal");
        }

        typename CenterSizeFactors::Results cresults;
        if (center) {
            cresults = centerer.run_blocked(size_factors.size(), size_factors.data(), block);
        } else {
            cresults = CenterSizeFactors::validate(size_factors.size(), size_factors.data());
        }

        if (cresults.has_zero) {
            if (!handle_zeros) {
                throw std::runtime_error("all size factors should be positive");
            } else {
                sanitize_zeros(size_factors);
            }
        }

        if (cresults.has_non_finite) {
            if (!handle_non_finite) {
                throw std::runtime_error("all size factors should be finite");
            } else {
                sanitize_non_finite(size_factors);
            }
        }

        double current_pseudo = pseudo_count;
        if (choose_pseudo_count) {
            current_pseudo = pseudo_chooser.run(size_factors.size(), size_factors.data());
        }

        if (sparse_addition && current_pseudo != 1) {
            for (auto& d : size_factors) {
                d *= current_pseudo;
            }
            current_pseudo = 1; // effectively 1 now.
        }

        typedef typename MAT::value_type Value_;
        auto div = tatami::make_DelayedUnaryIsometricOp(std::move(mat), tatami::make_DelayedDivideVectorHelper<true, 1, Value_>(std::move(size_factors)));

        if (current_pseudo == 1) {
            return tatami::make_DelayedUnaryIsometricOp(std::move(div), tatami::DelayedLog1pHelper<Value_>(2.0));
        } else {
            auto add = tatami::make_DelayedUnaryIsometricOp(std::move(div), tatami::make_DelayedAddScalarHelper<Value_>(current_pseudo));
            return tatami::make_DelayedUnaryIsometricOp(std::move(add), tatami::DelayedLogHelper<Value_>(2.0));
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
    std::shared_ptr<MAT> run(std::shared_ptr<MAT> mat) const {
        auto size_factors = tatami::column_sums(mat.get(), nthreads);
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
    std::shared_ptr<MAT> run_blocked(std::shared_ptr<MAT> mat, const B* block) const {
        auto size_factors = tatami::column_sums(mat.get(), nthreads);
        return run_blocked(mat, std::move(size_factors), block);
    }
};

};

#endif
