#ifndef SCRAN_SUGGEST_CRISPR_QC_FILTERS_H
#define SCRAN_SUGGEST_CRISPR_QC_FILTERS_H

#include "../utils/macros.hpp"

#include <vector>
#include <cmath>

#include "utils.hpp"
#include "PerCellCrisprQcMetrics.hpp"
#include "ComputeMedianMad.hpp"
#include "ChooseOutlierFilters.hpp"

/**
 * @file SuggestCrisprQcFilters.hpp
 *
 * @brief Create filters to identify low-quality cells from CRISPR-derived QC metrics.
 */

namespace scran {

/**
 * @brief Create filters to identify low-quality cells from CRISPR-derived QC metrics.
 *
 * In CRISPR guide count matrices, the QC filtering decisions are somewhat different than those for the other modalities.
 * Here, low-quality cells are defined as those with:
 *
 * - A low maximum count.
 *   This indicates that library preparation or sequencing was suboptimal,
 *   or that the cell was not transfected with any guide construct,
 *   or that the cell failed to express that guide construct.
 *
 * Directly defining a threshold on the maximum count is somewhat tricky as unsuccessful transfection is not uncommon.
 * This often results in a large subpopulation with low maximum counts, inflating the MAD and compromising the threshold calculation.
 * Instead, we use the following approach:
 *
 * 1. Compute the median of the proportion of counts in the most abundant guide (i.e., the maximum proportion),
 * 2. Subset the cells to only those with maximum proportions above the median,
 * 3. Define a threshold for low outliers on the log-transformed maximum count within the subset.
 *
 * This assumes that over 50% of cells were successfully transfected with a single guide construct and have high maximum proportions.
 * In contrast, unsuccessful transfections will be dominated by ambient contamination and have low proportions.
 * By taking the subset above the median proportion, we remove all of the unsuccessful transfections and enrich for mostly-high-quality cells.
 * From there, we can apply the usual outlier detection methods on the maximum count, with log-transformation to avoid a negative threshold.
 *
 * Keep in mind that the maximum proportion is only used to define the subset for threshold calculation.
 * Once the maximum count threshold is computed, they are applied to all cells, regardless of their maximum proportions.
 * This allows us to recover good cells that would have been filtered out by our aggressive median subset.
 * It also ensures that we do not remove cells transfected with multiple guides - such cells are not necessarily uninteresting, e.g., for examining interaction effects,
 * so we will err on the side of caution and leave them in.
 *
 * For datasets with multiple blocks, `SuggestCrisprQcFilters::run_blocked()` will compute block-specific thresholds for the maximum count.
 * See comments in `SuggestRnaQcFilters` for more details.
 */
class SuggestCrisprQcFilters {
public:
    /**
     * @brief Default parameters.
     */
    struct Defaults {
        /**
         * See `set_num_mads()` for details.
         */
        static constexpr double num_mads = 3;
    };

private:
    double num_mads = Defaults::num_mads;

public:
    /**
     * @param n Number of MADs below the median, to define the threshold for outliers in the maximum count.
     * This should be non-negative.
     *
     * @return Reference to this `SuggestCrisprQcFilters` object.
     */
    SuggestCrisprQcFilters& set_num_mads(double n = Defaults::num_mads) {
        num_mads = n;
        return *this;
    }

public:
    /**
     * @brief Thresholds to define outliers on each metric.
     *
     * `SuggestCrisprQcFilters::run()` and related methods will suggest some (hopefully) sensible thresholds based on outlier calls.
     * Users can directly modify the thresholds in instances of this class if the default suggestions are not suitable. 
     * These thresholds can then be used to generate calls for low-quality cells via the `filter()` methods.
     */
    struct Thresholds {
        /**
         * Lower thresholds to define small outliers on the maximum count.
         * Each entry contains the threshold used for the corresponding block.
         * For unblocked analyses, this will be of length 1 as all cells are assumed to belong to the same block.
         */
        std::vector<double> max_count;

    public:
        /**
         * @tparam overwrite Whether to overwrite existing truthy entries in `output`.
         * @tparam Float Floating point type for the metrics.
         * @tparam Integer Integer for the metrics.
         * @tparam Output Boolean type for the low-quality calls.
         *
         * @param n Number of cells.
         * @param[in] buffers Pointers to arrays of length `n`, containing the per-cell CRISPR-derived metrics.
         * These should be comparable to the values used to create this `Thresholds` object.
         * Only `max_proportion` and `sums` are used; `detected` is ignored and does not need to be set.
         * @param[out] output Pointer to an array of length `n`, to store the low-quality calls.
         * Values are set to `true` for low-quality cells.
         * If `overwrite = true`, values are set to `false` for high-quality cells, otherwise the existing entry is preserved.
         *
         * Use `filter_blocked()` instead for multi-block datasets. 
         */
        template<bool overwrite = true, typename Float, typename Integer, typename Output>
        void filter(size_t n, const PerCellCrisprQcMetrics::Buffers<Float, Integer>& buffers, Output* output) const {
            if (max_count.size() != 1) {
                throw std::runtime_error("should use filter_blocked() for multiple batches");
            }
            filter_<overwrite>(n, buffers, output, [](size_t i) -> size_t { return 0; });
        }

        /**
         * @overload
         *
         * @tparam Output Boolean type for the low-quality calls.
         *
         * @param metrics Collection of arrays of per-cell CRISPR metrics.
         * These should be comparable to the values used to create this `Thresholds` object.
         * Only `max_proportion` and `sums` are used; `detected` is ignored and does not need to be set.
         *
         * @return Vector of low-quality calls, of length equal to the number of cells in `metrics`.
         */
        template<typename Output = uint8_t>
        std::vector<Output> filter(const PerCellCrisprQcMetrics::Results& metrics) const {
            std::vector<Output> output(metrics.sums.size());
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
         * @param[in] buffers Pointers to arrays of length `n`, containing the per-cell CRISPR-derived metrics.
         * These should be comparable to the values used to create this `Thresholds` object.
         * Only `max_proportion` and `sums` are used; `detected` is ignored and does not need to be set.
         * @param[out] output Pointer to an array of length `n`, to store the low-quality calls.
         * Values are set to `true` for low-quality cells.
         * If `overwrite = true`, values are set to `false` for high-quality cells, otherwise the existing entry is preserved.
         */
        template<bool overwrite = true, typename Block, typename Float, typename Integer, typename Output>
        void filter_blocked(size_t n, const Block* block, const PerCellCrisprQcMetrics::Buffers<Float, Integer>& buffers, Output* output) const {
            if (block) {
                filter_<overwrite>(n, buffers, output, [&](size_t i) -> Block { return block[i]; });
            } else {
                filter<overwrite>(n, buffers, output);
            }
        }

        /**
         * @overload
         *
         * @tparam Block Integer type for the block assignments.
         * @tparam Output Boolean type for the low-quality calls.
         *
         * @param metrics Collection of arrays of per-cell CRISPR metrics.
         * These should be comparable to the values used to create this `Thresholds` object.
         * Only `max_proportion` and `sums` are used; `detected` is ignored and does not need to be set.
         * @param[in] block Pointer to an array of length `n`, containing the block assignment for each cell.
         * This may be `NULL`, in which case all cells are assumed to belong to the same block.
         *
         * @return Vector of low-quality calls, of length equal to the number of cells in `metrics`.
         */
        template<typename Output = uint8_t, typename Block>
        std::vector<Output> filter_blocked(const PerCellCrisprQcMetrics::Results& metrics, const Block* block) const {
            std::vector<Output> output(metrics.sums.size());
            filter_blocked(output.size(), block, metrics.buffers(), output.data());
            return output;
        }

    private:
        template<bool overwrite, typename Float, typename Integer, typename Output, typename Function>
        void filter_(size_t n, const PerCellCrisprQcMetrics::Buffers<Float, Integer>& buffers, Output* output, Function indexer) const {
            for (size_t i = 0; i < n; ++i) {
                auto b = indexer(i);
                auto candidate = buffers.max_proportion[i] * buffers.sums[i];

                if (quality_control::is_less_than<double>(candidate, max_count[b])) {
                    output[i] = true;
                    continue;
                }

                if constexpr(overwrite) {
                    output[i] = false;
                }
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
     * @param[in] buffers Pointers to arrays of length `n`, containing the per-cell CRISPR-derived metrics.
     *
     * @return Filtering thresholds for each metric.
     */
    template<typename Float, typename Integer>
    Thresholds run(size_t n, const PerCellCrisprQcMetrics::Buffers<Float, Integer>& buffers) const {
        return run_blocked(n, static_cast<int*>(NULL), buffers);
    }

    /**
     * @overload
     * @param metrics Collection of arrays of length equal to the number of cells, containing the per-cell CRISPR-derived metrics.
     *
     * @return Filtering thresholds for each metric.
     */
    Thresholds run(const PerCellCrisprQcMetrics::Results& metrics) const {
        return run(metrics.max_proportion.size(), metrics.buffers());
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
     * @param[in] buffers Pointers to arrays of length `n`, containing the per-cell CRISPR-derived metrics.
     * Only `max_proportion` and `sums` are used; `detected` is ignored and does not need to be set.
     *
     * @return Filtering thresholds for each metric in each block.
     */
    template<typename Block, typename Float, typename Integer>
    Thresholds run_blocked(size_t n, const Block* block, const PerCellCrisprQcMetrics::Buffers<Float, Integer>& buffers) const {
        Thresholds output;
        std::vector<int> starts;
        std::vector<Block> subblock;
        if (block) {
            starts = ComputeMedianMad::compute_block_starts(n, block);
            subblock.reserve(n);
        }

        // Subsetting to the observations in the top 50% of proportions.
        std::vector<double> workspace(n);
        std::vector<double> subvalues;
        subvalues.reserve(n);

        {
            ComputeMedianMad meddler;
            meddler.set_median_only(true);
            auto prop_res = meddler.run_blocked(n, block, starts, buffers.max_proportion, workspace.data());

            if (block) {
                for (size_t i = 0; i < n; ++i) {
                    auto p = buffers.max_proportion[i];
                    if (quality_control::is_greater_than_or_equal_to(p, prop_res.medians[block[i]])) {
                        subvalues.push_back(p * buffers.sums[i]);
                        subblock.push_back(block[i]);
                    }
                }
            } else {
                for (size_t i = 0; i < n; ++i) {
                    auto p = buffers.max_proportion[i];
                    if (quality_control::is_greater_than_or_equal_to(p, prop_res.medians[0])) {
                        subvalues.push_back(p * buffers.sums[i]);
                    }
                }
            }
        }

        // Filtering on the max counts.
        {
            ComputeMedianMad meddler;
            meddler.set_log(true);

            const Block* bptr = NULL;
            starts.clear();
            if (block) {
                bptr = subblock.data();
                starts = ComputeMedianMad::compute_block_starts(subblock.size(), bptr);
            }

            auto sums_res = meddler.run_blocked(subvalues.size(), bptr, starts, subvalues.data(), workspace.data());

            ChooseOutlierFilters filter;
            filter.set_num_mads(num_mads);
            filter.set_upper(false);
            auto sums_filt = filter.run(std::move(sums_res));

            output.max_count = std::move(sums_filt.lower);
        }

        return output;
    }

    /**
     * @overload
     * @tparam Block Integer type for the block assignments.
     *
     * @param metrics Collection of arrays of length equal to the number of cells, containing the per-cell CRISPR-derived metrics.
     * Only `max_proportion` and `sums` are used; `detected` is ignored and does not need to be set.
     * @param[in] block Pointer to an array of length equal to the number of cells, containing the block assignments for each cell.
     * This may be `NULL`, in which case all cells are assumed to belong to the same block.
     *
     * @return Filtering thresholds for each metric in each block.
     */
    template<typename Block>
    Thresholds run_blocked(const PerCellCrisprQcMetrics::Results& metrics, const Block* block) const {
        return run_blocked(metrics.max_proportion.size(), block, metrics.buffers());
    }
};

}

#endif
