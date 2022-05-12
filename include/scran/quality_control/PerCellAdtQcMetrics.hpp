#ifndef SCRAN_PER_CELL_ADT_QC_METRICS_HPP
#define SCRAN_PER_CELL_ADT_QC_METRICS_HPP

#include "PerCellQCMetrics.hpp"

/**
 * @file PerCellAdtQcMetrics.hpp
 *
 * @brief Compute per-cell quality control metrics from an ADT count matrix.
 */

namespace scran {

/**
 * @brief Compute per-cell quality control metrics from an ADT count matrix.
 *
 * Given a feature-by-cell ADT count matrix, this class computes several QC metrics:
 * 
 * - The total sum for each cell, which represents the efficiency of library preparation and sequencing.
 *   This is less useful as a QC metric for ADT data given that the total count may be strongly skewed by the presence or absence of a single feature.
 *   Nonetheless, we compute it, because why not.
 * - The number of detected features per cell
 *   Even though ADTs are commonly applied in situations where few features are expressed, we still expect detectable coverage of most features due to ambient contamination.
 *   The absence of detectable coverage indicates that library preparation or sequencing depth was suboptimal.
 * - The total sum of counts in pre-defined feature subsets.
 *   The exact interpretation depends on the nature of the subset - the most common use case involves isotype control (IgG) features.
 *   IgG antibodies should not bind to anything, so high coverage suggests that non-specific binding is a problem, e.g., due to antibody conjugates.
 */
class PerCellAdtQcMetrics {
public:
    /**
     * @brief Result store for QC metric calculations.
     * 
     * Meaningful instances of this object should generally be constructed by calling the `PerCellQCMetrics::run()` methods.
     * Empty instances can be default-constructed as placeholders.
     */
    struct Results {
        /**
         * @cond
         */
        Results() {}

        Results(size_t ncells, size_t nsubsets) : sums(ncells), detected(ncells), subset_totals(nsubsets, std::vector<double>(ncells)) {}
        /**
         * @endcond
         */

        /**
         * Sum of counts for each cell.
         */
        std::vector<double> sums;

        /**
         * Number of detected features in each cell.
         */
        std::vector<int> detected;

        /**
         * Total count in each feature subset in each cell.
         * Each inner vector corresponds to a feature subset and is of length equal to the number of cells.
         */
        std::vector<std::vector<double> > subset_totals;
    };

    /**
     * Compute the QC metrics from an ADT count matrix and return the results.
     *
     * @tparam Matrix Type of matrix, usually a `tatami::NumericMatrix`.
     * @tparam SubPtr Pointer to a type interpretable as boolean.
     *
     * @param mat Pointer to a feature-by-cells matrix containing ADT counts.
     * @param[in] subsets Vector of pointers to arrays of length equal to `mat->nrow()`.
     * Each array represents a feature subset and indicating whether each feature in `mat` belongs to that subset.
     * Users can pass `{}` if no subsets are to be used. 
     *
     * @return A `PerCellAdtQcMetrics::Results` object containing the QC metrics.
     * Subset proportions are returned depending on the `subsets`.
     */
    template<class Matrix, typename SubPtr = const uint8_t*>
    Results run(const Matrix* mat, std::vector<SubPtr> subsets) const {
        Results output(mat->ncol(), subsets.size());
        run(mat, std::move(subsets), output.sums.data(), output.detected.data(), vector_to_pointers(output.subset_totals));
        return output;
    }

public:
    /**
     * Compute the QC metrics from an ADT count matrix.
     *
     * @tparam Matrix Type of matrix, usually a `tatami::NumericMatrix`.
     * @tparam SubPtr Pointer to a type interpretable as boolean.
     * @tparam Sum Floating-point value, to store the sums.
     * @tparam Detected Integer value, to store the number of detected features.
     * @tparam SubTotal Floating point value, to store the subset proportions.
     *
     * @param mat Pointer to a feature-by-cells matrix containing ADT counts.
     * @param[in] subsets Vector of pointers to arrays of length equal to `mat->nrow()`.
     * Each array represents a feature subset and indicating whether each feature in `mat` belongs to that subset.
     * Users can pass `{}` if no subsets are to be used. 
     * @param[out] sums Pointer to an array of length equal to the number of columns in `mat`.
     * This is used to store the computed sums for all cells.
     * @param[out] detected Pointer to an array of length equal to the number of columns in `mat`.
     * This is used to store the number of detected features for all cells.
     * @param[out] subset_totals Vector of pointers to arrays of length equal to the number of columns in `mat`.
     * Each array corresponds to a feature subset and is used to store the total count in that subset across all cells.
     * The vector should be of length equal to that of `subsets`.
     * Users can pass `{}` if no subsets are used.
     *
     * @return `sums`, `detected`, and each array in `subset_proportions` is filled with the relevant statistics.
     */
    template<class Matrix, typename SubPtr = const uint8_t*, typename Sum, typename Detected, typename SubTotal>
    void run(const Matrix* mat, const std::vector<SubPtr>& subsets, Sum* sums, Detected* detected, std::vector<SubTotal*> subset_totals) const {
        PerCellQCMetrics runner;
        runner.set_subset_totals(true);
        runner.run(mat, subsets, sums, detected, std::move(subset_totals));
        return;
    }
};

}

#endif
