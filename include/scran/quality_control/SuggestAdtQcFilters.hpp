#ifndef SCRAN_SUGGEST_ADT_QC_FILTERS_H
#define SCRAN_SUGGEST_ADT_QC_FILTERS_H

#include "../utils/macros.hpp"

#include <vector>
#include <cmath>

#include "utils.hpp"
#include "PerCellAdtQcMetrics.hpp"
#include "ComputeMedianMad.hpp"
#include "ChooseOutlierFilters.hpp"

/**
 * @file SuggestAdtQcFilters.hpp
 *
 * @brief Create filters to identify low-quality cells from ADT-derived QC metrics.
 */

namespace scran {

/**
 * @brief Create filters to identify low-quality cells from ADT-derived QC metrics.
 *
 * In antibody-derived tag (ADT) count matrices, the QC filtering decisions are slightly different than those for RNA count matrices (see `SuggestAdtQcFilters` for the latter).
 * Here, low-quality cells are defined as those with:
 *
 * - Low numbers of detected features, which indicates that library preparation or sequencing depth was suboptimal.
 *   Even in ADT data, we should expect to detect many features in a cell due to ambient contamination.
 * - High total counts in the isotype control (IgG) subsets.
 *   The IgG antibodies should not bind to anything, so high coverage is indicative of non-specific binding or antibody conjugates.
 *   
 * We define a threshold on each metric based on a certain number of MADs from the median.
 * This assumes that most cells in the experiment are of high (or at least acceptable) quality;
 * any outliers are indicative of low-quality cells that should be filtered out.
 * See the `ChooseOutlierFilters` class for implementation details.
 *
 * For the number of detected features and the total IgG counts, the outliers are defined after log-transformation of the metrics.
 * This improves resolution at low values and ensures that the defined threshold is not negative.
 * Note that all thresholds are still reported on the original scale, so no further exponentiation is required.
 *
 * For the number of detected features, we supplement the MAD-based threshold with a minimum drop.
 * Cells are only considered to be low quality if the difference in the number of detected features from the median is greater than a certain percentage.
 * By default, the number must drop by at least 10% from the median.
 * This avoids overly aggressive filtering when the MAD is zero due to the discrete nature of this statistic in datasets with few tags.
 *
 * For datasets with multiple blocks, `SuggestAdtQcFilters::run_blocked()` will compute block-specific thresholds for each metric.
 * See comments in `SuggestRnaQcFilters` for more details.
 */
class SuggestAdtQcFilters {
public:
    /**
     * @brief Default parameters.
     */
    struct Defaults {
        /**
         * See `set_num_mads()` for details.
         */
        static constexpr double num_mads = 3;

        /**
         * See `set_min_detected_drop()` for details.
         */
        static constexpr double min_detected_drop = 0.1;
    };

private:
    double detected_num_mads = Defaults::num_mads;
    double subset_num_mads = Defaults::num_mads;
    double min_detected_drop = Defaults::min_detected_drop;

public:
    /**
     * @param n Number of MADs below the median, to define the threshold for outliers in the number of detected features.
     * This should be non-negative.
     *
     * @return Reference to this `SuggestAdtQcFilters` object.
     */
    SuggestAdtQcFilters& set_detected_num_mads(double n = Defaults::num_mads) {
        detected_num_mads = n;
        return *this;
    }

    /**
     * @param n Number of MADs above the median, to define the threshold for outliers in the total count for each subset.
     * This should be non-negative.
     *
     * @return Reference to this `SuggestAdtQcFilters` object.
     */
    SuggestAdtQcFilters& set_subset_num_mads(double n = Defaults::num_mads) {
        subset_num_mads = n;
        return *this;
    }

    /**
     * @param n Number of MADs from the median, overriding previous calls to `set_detected_num_mads()` and `set_subset_num_mads()`.
     * This should be non-negative.
     *
     * @return Reference to this `SuggestAdtQcFilters` object.
     */
    SuggestAdtQcFilters& set_num_mads(double n = Defaults::num_mads) {
        detected_num_mads= n;
        subset_num_mads = n;
        return *this;
    }

    /**
     * @param m Minimum drop in the number of detected features from the median, in order to consider a cell to be of low quality.
     * This should lie in $[0, 1)$.
     *
     * @return Reference to this `SuggestAdtQcFilters` object.
     */
    SuggestAdtQcFilters& set_min_detected_drop(double m = Defaults::min_detected_drop) {
        min_detected_drop= m;
        return *this;
    }

public:
    /**
     * @brief Thresholds to define outliers on each metric.
     *
     * `SuggestAdtQcFilters::run()` and related methods will suggest some (hopefully) sensible thresholds based on outlier calls.
     * Users can directly modify the thresholds in instances of this class if the default suggestions are not suitable. 
     * These thresholds can then be used to generate calls for low-quality cells via the `filter()` methods.
     */
    struct Thresholds {
        /**
         * Lower thresholds to define small outliers on the number of detected features.
         * Each entry contains the threshold used for the corresponding block.
         * For unblocked analyses, this will be of length 1 as all cells are assumed to belong to the same block.
         */
        std::vector<double> detected;

        /**
         * Upper thresholds to define large outliers on the subset totals.
         * Each vector corresponds to a feature subset while each entry of the inner vector corresponds to a block of cells.
         * For unblocked analyses, all cells are assumed to belong to a single block, i.e., all inner vectors have length 1.
         */
        std::vector<std::vector<double> > subset_totals;

    public:
        /**
         * @tparam overwrite Whether to overwrite existing truthy entries in `output`.
         * @tparam Float Floating point type for the metrics.
         * @tparam Integer Integer for the metrics.
         * @tparam Output Boolean type for the low-quality calls.
         *
         * @param n Number of cells.
         * @param[in] buffers Pointers to arrays of length `n`, containing the per-cell ADT-derived metrics.
         * These should be comparable to the values used to create this `Thresholds` object.
         * Only `detected` and `subset_totals` are used; `sums` does not need to be set.
         * @param[out] output Pointer to an array of length `n`, to store the low-quality calls.
         * Values are set to `true` for low-quality cells.
         * If `overwrite = true`, values are set to `false` for high-quality cells, otherwise the existing entry is preserved.
         *
         * Use `filter_blocked()` instead for multi-block datasets. 
         */
        template<bool overwrite = true, typename Float, typename Integer, typename Output>
        void filter(size_t n, const PerCellAdtQcMetrics::Buffers<Float, Integer>& buffers, Output* output) const {
            if (detected.size() != 1) {
                throw std::runtime_error("should use filter_blocked() for multiple batches");
            }
            filter_<overwrite>(n, buffers, output, [](size_t i) -> size_t { return 0; });
        }

        /**
         * @overload
         *
         * @tparam Output Boolean type for the low-quality calls.
         *
         * @param metrics Collection of arrays of per-cell ADT metrics.
         * These should be comparable to the values used to create this `Thresholds` object.
         * Only `detected` and `subset_totals` are used; `sums` does not need to be set.
         *
         * @return Vector of low-quality calls, of length equal to the number of cells in `metrics`.
         */
        template<typename Output = uint8_t>
        std::vector<Output> filter(const PerCellAdtQcMetrics::Results& metrics) const {
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
         * @param[in] buffers Pointers to arrays of length `n`, containing the per-cell ADT-derived metrics.
         * These should be comparable to the values used to create this `Thresholds` object.
         * Only `detected` and `subset_totals` are used; `sums` does not need to be set.
         * @param[out] output Pointer to an array of length `n`, to store the low-quality calls.
         * Values are set to `true` for low-quality cells.
         * If `overwrite = true`, values are set to `false` for high-quality cells, otherwise the existing entry is preserved.
         */
        template<bool overwrite = true, typename Block, typename Float, typename Integer, typename Output>
        void filter_blocked(size_t n, const Block* block, const PerCellAdtQcMetrics::Buffers<Float, Integer>& buffers, Output* output) const {
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
         * @param metrics Collection of arrays of per-cell ADT metrics.
         * These should be comparable to the values used to create this `Thresholds` object.
         * Only `detected` and `subset_totals` are used; `sums` does not need to be set.
         * @param[in] block Pointer to an array of length `n`, containing the block assignment for each cell.
         * This may be `NULL`, in which case all cells are assumed to belong to the same block.
         *
         * @return Vector of low-quality calls, of length equal to the number of cells in `metrics`.
         */
        template<typename Output = uint8_t, typename Block>
        std::vector<Output> filter_blocked(const PerCellAdtQcMetrics::Results& metrics, const Block* block) const {
            std::vector<Output> output(metrics.detected.size());
            filter_blocked(output.size(), block, metrics.buffers(), output.data());
            return output;
        }

    private:
        template<bool overwrite, typename Float, typename Integer, typename Output, typename Function>
        void filter_(size_t n, const PerCellAdtQcMetrics::Buffers<Float, Integer>& buffers, Output* output, Function indexer) const {
            size_t nsubsets = subset_totals.size();

            for (size_t i = 0; i < n; ++i) {
                auto b = indexer(i);

                if (quality_control::is_less_than<double>(buffers.detected[i], detected[b])) {
                    output[i] = true;
                    continue;
                }

                bool fail = false;
                for (size_t s = 0; s < nsubsets; ++s) {
                    if (quality_control::is_greater_than(buffers.subset_totals[s][i], subset_totals[s][b])) {
                        fail = true;
                        break;
                    }
                }
                if (fail) {
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
     * @param[in] buffers Pointers to arrays of length `n`, containing the per-cell ADT-derived metrics.
     *
     * @return Filtering thresholds for each metric.
     */
    template<typename Float, typename Integer>
    Thresholds run(size_t n, const PerCellAdtQcMetrics::Buffers<Float, Integer>& buffers) const {
        return run_blocked(n, static_cast<int*>(NULL), buffers);
    }

    /**
     * @overload
     * @param metrics Collection of arrays of length equal to the number of cells, containing the per-cell ADT-derived metrics.
     *
     * @return Filtering thresholds for each metric.
     */
    Thresholds run(const PerCellAdtQcMetrics::Results& metrics) const {
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
     * @param[in] buffers Pointers to arrays of length `n`, containing the per-cell ADT-derived metrics.
     * Only `detected` and `subset_totals` are used; `sums` does not need to be set.
     *
     * @return Filtering thresholds for each metric in each block.
     */
    template<typename Block, typename Float, typename Integer>
    Thresholds run_blocked(size_t n, const Block* block, const PerCellAdtQcMetrics::Buffers<Float, Integer>& buffers) const {
        Thresholds output;
        std::vector<double> workspace(n);
        std::vector<int> starts;
        if (block) {
            starts = ComputeMedianMad::compute_block_starts(n, block);
        }

        // Filtering on the number of detected features.
        {
            ComputeMedianMad meddler;
            meddler.set_log(true);
            auto detected_res = meddler.run_blocked(n, block, starts, buffers.detected, workspace.data());

            ChooseOutlierFilters filter;
            filter.set_num_mads(detected_num_mads);
            filter.set_min_diff(-std::log(1 - min_detected_drop));
            filter.set_upper(false);
            auto detected_filt = filter.run(std::move(detected_res));

            output.detected = std::move(detected_filt.lower);
        }

        // Filtering on the IgG subset totals.
        {
            ComputeMedianMad meddler;
            meddler.set_log(true);

            ChooseOutlierFilters filter;
            filter.set_num_mads(subset_num_mads);
            filter.set_lower(false);

            size_t nsubsets = buffers.subset_totals.size();
            for (size_t s = 0; s < nsubsets; ++s) {
                auto subset_res = meddler.run_blocked(n, block, starts, buffers.subset_totals[s], workspace.data());
                auto subset_filt = filter.run(std::move(subset_res));
                output.subset_totals.push_back(std::move(subset_filt.upper));
            }
        }

        return output;
    }

    /**
     * @overload
     * @tparam Block Integer type for the block assignments.
     *
     * @param metrics Collection of arrays of length equal to the number of cells, containing the per-cell ADT-derived metrics.
     * Only `detected` and `subset_totals` are used; `sums` does not need to be set.
     * @param[in] block Pointer to an array of length equal to the number of cells, containing the block assignments for each cell.
     * This may be `NULL`, in which case all cells are assumed to belong to the same block.
     *
     * @return Filtering thresholds for each metric in each block.
     */
    template<typename Block>
    Thresholds run_blocked(const PerCellAdtQcMetrics::Results& metrics, const Block* block) const {
        return run_blocked(metrics.detected.size(), block, metrics.buffers());
    }
};

}

#endif
