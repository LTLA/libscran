#ifndef SCRAN_SUGGEST_RNA_QC_FILTERS_H
#define SCRAN_SUGGEST_RNA_QC_FILTERS_H

#include "../utils/macros.hpp"

#include <vector>
#include <cstdint>

#include "PerCellRnaQcMetrics.hpp"
#include "ComputeMedianMad.hpp"
#include "FilterOutliers.hpp"

/**
 * @file SuggestRnaQcFilters.hpp
 *
 * @brief Create filters to identify low-quality cells from RNA-derived QC metrics.
 */

namespace scran {

/**
 * @brief Create filters to identify low-quality cells from RNA-derived QC metrics.
 *
 * Use an outlier-based approach on common QC metrics on the RNA data (see the `PerCellRnaQcMetrics` class) to identify low-quality cells.
 * Specifically, low-quality cells are defined as those with:
 *
 * - Low total counts, indicating that library preparation or sequencing depth was suboptimal.
 * - Low numbers of detected features, a slightly different flavor of the above reasoning.
 * - High proportions of counts in the mitochondrial (or spike-in) subsets, representing cell damage.
 *
 * Outliers are defined on each metric by counting the number of MADs from the median value across all cells.
 * This assumes that most cells in the experiment are of high (or at least acceptable) quality;
 * any anomalies are indicative of low-quality cells that should be filtered out.
 * See the `FilterOutliers` class for implementation details.
 *
 * For the total counts and number of detected features, the outliers are defined after log-transformation of the metrics.
 * This improves resolution at low values and ensures that the defined threshold is not negative.
 * Note that all thresholds are still reported on the original scale, so no further exponentiation is required.
 */
class SuggestRnaQcFilters {
public:
    /**
     * Number of MADs below the median, to define the threshold for outliers in the number of detected features.
     * This should be non-negative.
     */
    double detected_num_mads = 3;

    /**
     * Number of MADs below the median, to define the threshold for outliers in the total count per cell.
     * This should be non-negative.
     */
    double sums_num_mads = 3;

    /**
     * Number of MADs above the median, to define the threshold for outliers in the subset proportions.
     * This should be non-negative.
     */
    double subset_num_mads = 3;

public:
    /**
     * @brief Thresholds to define outliers on each metric.
     *
     * `SuggestRnaQcFilters::run()` and related methods will suggest some (hopefully) sensible thresholds based on outlier calls.
     * Users can directly modify the thresholds in instances of this class if the default suggestions are not suitable. 
     * These thresholds can then be used to generate calls for low-quality cells via the `filter()` methods.
     */
    struct Thresholds {
        /**
         * Lower thresholds to define small outliers on the total counts.
         * Each entry contains the threshold used for the corresponding block.
         * For unblocked analyses, this will be of length 1 as all cells are assumed to belong to the same block.
         */
        std::vector<double> sums;

        /**
         * Lower thresholds to define small outliers on the number of detected features.
         * Each entry contains the threshold used for the corresponding block.
         * For unblocked analyses, this will be of length 1 as all cells are assumed to belong to the same block.
         */
        std::vector<double> detected;

        /**
         * Upper thresholds to define large outliers on the subset proportions.
         * Each vector corresponds to a feature subset while each entry of the inner vector corresponds to a block of cells.
         * For unblocked analyses, all cells are assumed to belong to a single block, i.e., all inner vectors have length 1.
         */
        std::vector<std::vector<double> > subset_proportions;

    public:
        /**
         * @tparam Float Floating point type for the metrics.
         * @tparam Integer Integer for the metrics.
         * @tparam Output Boolean type for the low-quality calls.
         *
         * @param n Number of cells.
         * @param[in] buffers Pointers to arrays of length `n`, containing the per-cell RNA-derived metrics.
         * These should be comparable to the values used to create this `Thresholds` object.
         * @param[out] output Pointer to an array of length `n`, to store the low-quality calls.
         * Values are set to 1 for low-quality cells, 0 otherwise.
         *
         * Use `filter_blocked()` instead for multi-block datasets. 
         */
        template<typename Float, typename Integer, typename Output>
        void filter(size_t n, const PerCellRnaQcMetrics::Buffers<Float, Integer>& buffers, Output* output) const {
            if (detected.size() != 1) {
                throw std::runtime_error("should use filter_blocked() for multiple batches");
            }
            filter_(n, buffers, output, [](size_t i) -> size_t { return 0; });
        }

        /**
         * @overload
         *
         * @tparam Output Boolean type for the low-quality calls.
         *
         * @param metrics Collection of arrays of per-cell RNA metrics.
         * These should be comparable to the values used to create this `Thresholds` object.
         *
         * @return Vector of low-quality calls, of length equal to the number of cells in `metrics`.
         */
        template<typename Output = uint8_t>
        std::vector<Output> filter(const PerCellRnaQcMetrics::Results& metrics) const {
            std::vector<Output> output(metrics.detected.size());
            filter(output.size(), metrics.buffers(), output.data());
            return output;
        }

    public:
        /**
         * @tparam Block Integer type for the block assignments.
         * @tparam Float Floating point type for the metrics.
         * @tparam Integer Integer for the metrics.
         * @tparam Output Boolean type for the low-quality calls.
         *
         * @param n Number of cells.
         * @param[in] block Pointer to an array of length `n`, containing the block assignment for each cell.
         * This may be `NULL`, in which case all cells are assumed to belong to the same block.
         * @param[in] buffers Pointers to arrays of length `n`, containing the per-cell RNA-derived metrics.
         * These should be comparable to the values used to create this `Thresholds` object.
         * @param[out] output Pointer to an array of length `n`, to store the low-quality calls.
         * Values are set to 1 for low-quality cells, 0 otherwise.
         */
        template<typename Block, typename Float, typename Integer, typename Output>
        void filter_blocked(size_t n, const Block* block, const PerCellRnaQcMetrics::Buffers<Float, Integer>& buffers, Output* output) const {
            if (block) {
                filter_(n, buffers, output, [&](size_t i) -> Block { return block[i]; });
            } else {
                filter(n, buffers, output);
            }
        }

        /**
         * @overload
         *
         * @tparam Block Integer type for the block assignments.
         * @tparam Output Boolean type for the low-quality calls.
         *
         * @param metrics Collection of arrays of per-cell RNA metrics.
         * These should be comparable to the values used to create this `Thresholds` object.
         * @param[in] block Pointer to an array of length `n`, containing the block assignment for each cell.
         * This may be `NULL`, in which case all cells are assumed to belong to the same block.
         *
         * @return Vector of low-quality calls, of length equal to the number of cells in `metrics`.
         */
        template<typename Output = uint8_t, typename Block>
        std::vector<Output> filter_blocked(const PerCellRnaQcMetrics::Results& metrics, const Block* block) const {
            std::vector<Output> output(metrics.detected.size());
            filter_blocked(output.size(), block, metrics.buffers(), output.data());
            return output;
        }

    private:
        template<typename Float, typename Integer, typename Output, typename Function>
        void filter_(size_t n, const PerCellRnaQcMetrics::Buffers<Float, Integer>& buffers, Output* output, Function indexer) const {
            size_t nsubsets = subset_proportions.size();

            for (size_t i = 0; i < n; ++i) {
                auto b = indexer(i);
                if (buffers.sums[i] < sums[b]) {
                    output[i] = true;
                    continue;
                }

                if (buffers.detected[i] < detected[b]) {
                    output[i] = true;
                    continue;
                }

                bool fail = false;
                for (size_t s = 0; s < nsubsets; ++s) {
                    if (buffers.subset_proportions[s][i] > subset_proportions[s][b]) {
                        fail = true;
                        break;
                    }
                }
                if (fail) {
                    output[i] = true;
                    continue;
                }

                output[i] = false;
            }

            return;
        }
    };

public:
    /**
     * @tparam Float Floating point type for the metrics.
     * @tparam Integer Integer for the metrics.
     *
     * @param n Number of cells.
     * @param[in] buffers Pointers to arrays of length `n`, containing the per-cell RNA-derived metrics.
     *
     * @return Filtering thresholds for each metric.
     */
    template<typename Float, typename Integer>
    Thresholds run(size_t n, const PerCellRnaQcMetrics::Buffers<Float, Integer>& buffers) const {
        return run_blocked(n, static_cast<int*>(NULL), buffers);
    }

    /**
     * @overload
     * @param metrics Collection of arrays of length equal to the number of cells, containing the per-cell RNA-derived metrics.
     *
     * @return Filtering thresholds for each metric.
     */
    Thresholds run(const PerCellRnaQcMetrics::Results& metrics) const {
        return run(metrics.detected.size(), metrics.buffers());
    }

public:
    /**
     * @tparam Block Integer type for the block assignments.
     * @tparam Float Floating point type for the metrics.
     * @tparam Integer Integer for the metrics.
     *
     * @param n Number of cells.
     * @param[in] block Pointer to an array of length `n`, containing the block assignments for each cell.
     * This may be `NULL`, in which case all cells are assumed to belong to the same block.
     * @param[in] buffers Pointers to arrays of length `n`, containing the per-cell RNA-derived metrics.
     *
     * @return Filtering thresholds for each metric in each block.
     */
    template<typename Block, typename Float, typename Integer>
    Thresholds run_blocked(size_t n, const Block* block, const PerCellRnaQcMetrics::Buffers<Float, Integer>& buffers) const {
        Thresholds output;
        std::vector<double> workspace(n);
        std::vector<int> starts;
        if (block) {
            starts = ComputeMedianMad::compute_block_starts(n, block);
        }

        // Filtering on the total counts.
        {
            ComputeMedianMad meddler;
            meddler.log = true;

            FilterOutliers filter;
            filter.num_mads = sums_num_mads;
            filter.upper = false;

            auto sums_res = meddler.run_blocked(n, block, starts, buffers.sums, workspace.data());
            auto sums_filt = filter.run(std::move(sums_res));
            output.sums = std::move(sums_filt.lower);
        }

        // Filtering on the detected features.
        {
            ComputeMedianMad meddler;
            meddler.log = true;

            FilterOutliers filter;
            filter.num_mads = sums_num_mads;
            filter.upper = false;

            auto detected_res = meddler.run_blocked(n, block, starts, buffers.detected, workspace.data());
            auto detected_filt = filter.run(std::move(detected_res));
            output.detected = std::move(detected_filt.lower);
        }

        // Filtering on the subset proportions (no log). 
        {
            ComputeMedianMad meddler;

            FilterOutliers filter;
            filter.num_mads = subset_num_mads;
            filter.lower = false;

            size_t nsubsets = buffers.subset_proportions.size();
            for (size_t s = 0; s < nsubsets; ++s) {
                auto subset_res = meddler.run_blocked(n, block, starts, buffers.subset_proportions[s], workspace.data());
                auto subset_filt = filter.run(std::move(subset_res));
                output.subset_proportions.push_back(std::move(subset_filt.upper));
            }
        }

        return output;
    }

    /**
     * @overload
     * @tparam Block Integer type for the block assignments.
     *
     * @param metrics Collection of arrays of length equal to the number of cells, containing the per-cell RNA-derived metrics.
     * @param[in] block Pointer to an array of length equal to the number of cells, containing the block assignments for each cell.
     * This may be `NULL`, in which case all cells are assumed to belong to the same block.
     *
     * @return Filtering thresholds for each metric in each block.
     */
    template<typename Block>
    Thresholds run_blocked(const PerCellRnaQcMetrics::Results& metrics, const Block* block) const {
        return run_blocked(metrics.detected.size(), block, metrics.buffers());
    }
};

}

#endif
