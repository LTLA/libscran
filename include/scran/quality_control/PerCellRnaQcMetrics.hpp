#ifndef SCRAN_PER_CELL_RNA_QC_METRICS_HPP
#define SCRAN_PER_CELL_RNA_QC_METRICS_HPP

#include "../utils/macros.hpp"

#include <vector>
#include <limits>

#include "tatami/base/Matrix.hpp"
#include "PerCellQcMetrics.hpp"
#include "utils.hpp"

/**
 * @file PerCellRnaQcMetrics.hpp
 *
 * @brief Compute typical per-cell quality control metrics from an RNA count matrix.
 */

namespace scran {

/**
 * @brief Compute typical per-cell quality control metrics from an RNA count matrix.
 *
 * Given a feature-by-cell RNA count matrix, this class computes several QC metrics:
 * 
 * - The total sum of counts for each cell, which represents the efficiency of library preparation and sequencing.
 *   Low totals indicate that the library was not successfully captured.
 * - The number of detected features.
 *   This also quantifies the library preparation efficiency, but with a greater focus on capturing the transcriptional complexity.
 * - The proportion of counts in pre-defined feature subsets.
 *   The exact interpretation depends on the nature of the subset -
 *   most commonly, one subset will contain all genes on the mitochondrial chromosome,
 *   where higher proportions indicate cell damage due to loss of cytoplasmic transcripts.
 *   Spike-in proportions can be interpreted in a similar manner.
 *
 * This class is just a pre-configured wrapper around `PerCellQcMetrics`,
 * with some careful division of the subset totals to obtain the subset proportions. 
 */
class PerCellRnaQcMetrics {
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
     * @return A reference to this `PerCellRnaQcMetrics` object.
     */
    PerCellRnaQcMetrics& set_num_threads(int n = Defaults::num_threads) {
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
         * Each entry should point to an array of length equal to the number of cells, see `Results::subset_proportions`.
         */
        std::vector<Float*> subset_proportions;
    };

public:
    /**
     * Compute the QC metrics from an input matrix.
     *
     * @tparam Matrix Type of matrix, usually a `tatami::NumericMatrix`.
     * @tparam Subset Pointer to an array of values interpretable as booleans.
     * @tparam Float Floating point type to store the totals.
     * @tparam Integer Integer type to store the counts and indices.
     *
     * @param mat Pointer to a feature-by-cells matrix containing counts.
     * @param[in] subsets Vector of pointers to arrays of length equal to `mat->nrow()`.
     * Each array represents a feature subset and indicating whether each feature in `mat` belongs to that subset.
     * Users can pass `{}` if no subsets are to be used. 
     * @param[out] output `Buffers` object in which to store the output.
     * All pointers should be set to non-`NULL` values.
     */
    template<class Matrix, typename Subset = const uint8_t*, typename Float, typename Integer>
    void run(const Matrix* mat, const std::vector<Subset>& subsets, Buffers<Float, Integer>& output) const {
        size_t NC = mat->ncol();

        // Calling the general-purpose PerCellQcMetrics function.
        PerCellQcMetrics general;
        general.set_num_threads(num_threads);

        PerCellQcMetrics::Buffers<Float, Integer> tmp;
        tmp.total = output.sums;
        tmp.detected = output.detected;
        tmp.subset_total = output.subset_proportions;

        // Make sure that sums are computed one way or another.
        std::vector<Float> placeholder;
        if (!tmp.total && tmp.subset_total.size()) {
            placeholder.resize(NC);
            tmp.total = placeholder.data();            
        }

        general.run(mat, subsets, tmp); 

        // Computing the proportions safely.
        for (auto s : output.subset_proportions) {
            if (s) {
                quality_control::safe_divide(NC, s, tmp.total);
            }
        }

        return;
    }

public:
    /**
     * @brief Result store for QC metric calculations.
     * 
     * Meaningful instances of this object should generally be constructed by calling the `PerCellRnaQcMetrics::run()` methods.
     * Empty instances can be default-constructed as placeholders.
     */
    struct Results {
        /**
         * @cond
         */
        Results() {}

        Results(size_t ncells, size_t nsubsets) : sums(ncells), detected(ncells), subset_proportions(nsubsets, std::vector<double>(ncells)) {}
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
         * Proportion of counts in each feature subset in each cell.
         * Each inner vector corresponds to a feature subset and is of length equal to the number of cells.
         */
        std::vector<std::vector<double> > subset_proportions;

    public:
        /**
         * We assume that all members have already been allocated enough memory for use with `PerCellRnaQcMetrics::run()`. 
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

            size_t nsubsets = y.subset_proportions.size();
            x.subset_proportions.resize(nsubsets);
            for (size_t s = 0; s < nsubsets; ++s) {
                x.subset_proportions[s] = y.subset_proportions[s].data();
            }
        }
   };

public:
    /**
     * Compute the QC metrics from an input matrix and return the results.
     *
     * @tparam Matrix Type of matrix, usually a `tatami::NumericMatrix`.
     * @tparam Subset Pointer to an array of values interpretable as booleans.
     *
     * @param mat Pointer to a feature-by-cells **tatami** matrix containing counts.
     * @param[in] subsets Vector of pointers to arrays of length equal to `mat->nrow()`.
     * Each array represents a feature subset and indicating whether each feature in `mat` belongs to that subset.
     * Users can pass `{}` if no subsets are to be used. 
     *
     * @return A `PerCellRnaQcMetrics::Results` object containing the QC metrics.
     * Subset proportions are returned depending on the `subsets`.
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
