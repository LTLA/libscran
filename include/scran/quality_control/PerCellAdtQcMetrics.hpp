#ifndef SCRAN_PER_CELL_ADT_QC_METRICS_HPP
#define SCRAN_PER_CELL_ADT_QC_METRICS_HPP

#include "../utils/macros.hpp"

#include <vector>
#include <limits>

#include "tatami/base/Matrix.hpp"
#include "PerCellQcMetrics.hpp"

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
 * - The total sum of counts for each cell, which (in theory) represents the efficiency of library preparation and sequencing.
 *   This is less useful as a QC metric for ADT data given that the total count is strongly influenced by the actual abundance of the targeted features,
 *   i.e., the presence of a surface protein will typically result in an order-of-magnitude increase to the total count that is independent of the cell's technical quality.
 *   Nonetheless, we compute it for diagnostic purposes.
 * - The number of detected features per cell.
 *   Even though ADTs are commonly applied in situations where few features are highly abundant, we still expect detectable coverage of most features due to ambient contamination, non-specific binding or some background expression.
 *   The absence of detectable coverage indicates that library preparation or sequencing depth was suboptimal.
 * - The total sum of counts in pre-defined feature subsets.
 *   The exact interpretation depends on the nature of the subset - the most common use case involves isotype control (IgG) features.
 *   IgG antibodies should not bind to anything, so high coverage suggests that non-specific binding is a problem, e.g., due to antibody conjugates.
 *
 * Under the hood, this class is just a pre-configured wrapper around `PerCellQcMetrics`.
 */
class PerCellAdtQcMetrics {
public:
    /**
     * @brief Default parameters.
     */
    struct Defaults {
        /**
         * See `set_num_threads()` for details.
         */
        static constexpr int num_threads = 1;
    };

    /**
     * @param n Number of threads to use. 
     * @return A reference to this `PerCellAdtQcMetrics` object.
     */
    PerCellAdtQcMetrics& set_num_threads(int n = Defaults::num_threads) {
        num_threads = n;
        return *this;
    }

private:
    int num_threads = Defaults::num_threads;

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
         */
        Float* sums = NULL;

        /**
         * Pointer to an array of length equal to the number of cells, see `Results::detected`.
         */
        Integer* detected = NULL;

        /**
         * Vector of pointers of length equal to the number of feature subsets.
         * Each pointer should be to an array of length equal to the number of cells, see `Results::subset_totals`.
         */
        std::vector<Float*> subset_totals;
    };

public:
    /**
     * Compute the QC metrics from an ADT count matrix.
     *
     * @tparam Matrix Type of matrix, usually a `tatami::NumericMatrix`.
     * @tparam Subset Pointer to a type interpretable as boolean.
     * @tparam Float Floating point type to store the totals.
     * @tparam Integer Integer type to store the counts and indices.
     *
     * @param mat Pointer to a feature-by-cells matrix containing ADT counts.
     * @param[in] subsets Vector of pointers to arrays of length equal to `mat->nrow()`.
     * Each array represents a feature subset and indicating whether each feature in `mat` belongs to that subset.
     * Users can pass `{}` if no subsets are to be used. 
     * @param[out] output `Buffers` object in which to store the output.
     * All pointers should be set to non-`NULL` values.
     */
    template<class Matrix, typename Subset = const uint8_t*, typename Float, typename Integer>
    void run(const Matrix* mat, const std::vector<Subset>& subsets, Buffers<Float, Integer>& output) const {
        // Calling the general-purpose PerCellQcMetrics function.
        PerCellQcMetrics general;
        general.set_num_threads(num_threads);

        PerCellQcMetrics::Buffers<Float, Integer> tmp;
        tmp.total = output.sums;
        tmp.detected = output.detected;
        tmp.subset_total = output.subset_totals;

        general.run(mat, subsets, tmp); 
        return;
    }

public:
    /**
     * @brief Result store for QC metric calculations.
     * 
     * Meaningful instances of this object should generally be constructed by calling the `PerCellAdtQcMetrics::run()` methods.
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

        /**
         * We assume that all members have already been allocated enough memory for use with `PerCellAdtQcMetrics::run()`. 
         *
         * @return A `Buffers` object with appropriate pointers to the members of this `Results` instance.
         */
        Buffers<> buffers() {
            Buffers<> output;
            populate_buffers(output, *this);
            return output;
        }

        /**
         * @overload
         * @return A `Buffers` object with const pointers to the members of this `Results` instance.
         */
        Buffers<const double, const int> buffers() const {
            Buffers<const double, const int> output;
            populate_buffers(output, *this);
            return output;
        }
    private:
        template<class SomeBuffer, class Results>
        static void populate_buffers(SomeBuffer& x, Results& y) {
            x.sums = y.sums.data();
            x.detected = y.detected.data();

            size_t nsubsets = y.subset_totals.size();
            x.subset_totals.resize(nsubsets);
            for (size_t s = 0; s < nsubsets; ++s) {
                x.subset_totals[s] = y.subset_totals[s].data();
            }
        }
    };

public:
    /**
     * Compute the QC metrics from an ADT count matrix and return the results.
     *
     * @tparam Matrix Type of matrix, usually a `tatami::NumericMatrix`.
     * @tparam Subset Pointer to a type interpretable as boolean.
     *
     * @param mat Pointer to a feature-by-cells matrix containing ADT counts.
     * @param[in] subsets Vector of pointers to arrays of length equal to `mat->nrow()`.
     * Each array represents a feature subset and indicating whether each feature in `mat` belongs to that subset.
     * Users can pass `{}` if no subsets are to be used. 
     *
     * @return A `PerCellAdtQcMetrics::Results` object containing the QC metrics.
     * Subset totals are returned depending on the `subsets`.
     */
    template<class Matrix, typename Subset = const uint8_t*>
    Results run(const Matrix* mat, const std::vector<Subset>& subsets) const {
        size_t nsubsets = subsets.size();
        Results output(mat->ncol(), nsubsets);
        auto buffers = output.buffers();
        run(mat, subsets, buffers);
        return output;
    }
};

}

#endif
