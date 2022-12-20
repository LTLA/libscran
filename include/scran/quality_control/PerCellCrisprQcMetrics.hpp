#ifndef SCRAN_PER_CELL_CRISPR_QC_METRICS_HPP
#define SCRAN_PER_CELL_CRISPR_QC_METRICS_HPP

#include "../utils/macros.hpp"

#include <vector>
#include <limits>

#include "tatami/base/Matrix.hpp"
#include "PerCellQcMetrics.hpp"
#include "utils.hpp"

/**
 * @file PerCellCrisprQcMetrics.hpp
 *
 * @brief Compute per-cell quality control metrics from a CRISPR guide count matrix.
 */

namespace scran {

/**
 * @brief Compute per-cell quality control metrics from a CRISPR guide count matrix.
 *
 * Given a feature-by-cell guide count matrix, this class computes several QC metrics:
 * 
 * - The total sum of counts for each cell.
 *   Low counts indicate that the cell was not successfully transfected with a construct,
 *   or that library preparation and sequencing failed.
 *   In either case, the cell is considered to be of low quality.
 * - The number of detected guides per cell.
 *   In theory, this should be 1, as each cell should express no more than one guide construct.
 *   However, ambient contamination may introduce non-zero counts for multiple guides, without necessarily interfering with downstream analyses.
 *   As such, this metric is less useful for guide data, though we compute it anyway.
 * - The proportion of counts in the most abundant guide construct.
 *   Low values indicate that the cell was transfected with multiple guides.
 *   The identity of the most abundant guide is also reported.
 *
 * Under the hood, this class is just a pre-configured wrapper around `PerCellQcMetrics`.
 * with some careful division of the maximum count.
 */
class PerCellCrisprQcMetrics {
public:
    /**
     * Number of threads to use. 
     */
    int num_threads = 1;

public:
    /**
     * @brief Buffers for direct storage of the calculated statistics.
     * @tparam Float Floating point type to store the totals.
     * @tparam Integer Integer type to store the counts and indices.
     */
    template<typename Float = double, typename Integer = int>
    struct Buffers {
        /**
         * Pointer to an array of length equal to the number of cells, see `Results::sums`.
         * Set to `NULL` to skip this calculation.
         */
        Float* sums = NULL;

        /**
         * Pointer to an array of length equal to the number of cells, see `Results::detected`.
         * Set to `NULL` to skip this calculation.
         */
        Integer* detected = NULL;

        /**
         * Pointer to an array of length equal to the number of cells, see `Results::max_proportion`.
         * Set to `NULL` to skip this calculation.
         */
        Float* max_proportion = NULL;

        /**
         * Pointer to an array of length equal to the number of cells, see `Results::max_index`.
         * Set to `NULL` to skip this calculation.
         */
        Integer* max_index = NULL;
    };

public:
    /**
     * Compute the QC metrics from an CRISPR count matrix.
     *
     * @tparam Matrix Type of matrix, usually a `tatami::NumericMatrix`.
     * @tparam Float Floating point type to store the totals.
     * @tparam Integer Integer type to store the counts and indices.
     *
     * @param mat Pointer to a feature-by-cells matrix containing CRISPR guide counts.
     * @param[out] output `Buffers` object in which to store the output.
     */
    template<class Matrix, typename Float, typename Integer>
    void run(const Matrix* mat, Buffers<Float, Integer>& output) const {
        size_t NC = mat->ncol();

        // Calling the general-purpose PerCellQcMetrics function.
        PerCellQcMetrics general;
        general.num_threads = num_threads;

        PerCellQcMetrics::Buffers<Float, Integer> tmp;
        tmp.total = output.sums;
        tmp.detected = output.detected;
        tmp.max_count = output.max_proportion;
        tmp.max_index = output.max_index;

        // Make sure that sums are computed one way or another.
        std::vector<Float> placeholder;
        if (!tmp.total && tmp.subset_total.size()) {
            placeholder.resize(NC);
            tmp.total = placeholder.data();            
        }

        general.run(mat, {}, tmp); 

        // Computing the proportion safely.
        quality_control::safe_divide(NC, output.max_proportion, tmp.total);
        return;
    }

public:
    /**
     * @brief Result store for QC metric calculations.
     * 
     * Meaningful instances of this object should generally be constructed by calling the `PerCellCrisprQcMetrics::run()` methods.
     * Empty instances can be default-constructed as placeholders.
     */
    struct Results {
        /**
         * @cond
         */
        Results() {}

        Results(size_t ncells) : sums(ncells), detected(ncells), max_proportion(ncells), max_index(ncells) {}
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
         * Proportion of counts in the most abundant guide.
         */
        std::vector<double> max_proportion;

        /**
         * Index of the most abundant guide.
         */
        std::vector<int> max_index;
    };

public:
    /**
     * Compute the QC metrics from an CRISPR count matrix and return the results.
     *
     * @tparam Matrix Type of matrix, usually a `tatami::NumericMatrix`.
     *
     * @param mat Pointer to a feature-by-cells matrix containing CRISPR counts.
     *
     * @return A `PerCellCrisprQcMetrics::Results` object containing the QC metrics.
     */
    template<class Matrix>
    Results run(const Matrix* mat) const {
        Results output(mat->ncol());

        Buffers<> buffers;
        buffers.sums = output.sums.data();
        buffers.detected = output.detected.data();
        buffers.max_proportion = output.max_proportion.data();
        buffers.max_index = output.max_index.data();

        run(mat, buffers);
        return output;
    }
};

}

#endif
