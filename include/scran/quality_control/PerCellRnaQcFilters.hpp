#ifndef SCRAN_PER_CELL_QC_FILTERS_H
#define SCRAN_PER_CELL_QC_FILTERS_H

#include <vector>
#include <cstdint>

#include "../utils/vector_to_pointers.hpp"

#include "PerCellRnaQcMetrics.hpp"
#include "IsOutlier.hpp"

/**
 * @file PerCellRnaQcFilters.hpp
 *
 * @brief Create filters to identify low-quality cells from RNA metrics.
 */

namespace scran {

/**
 * @brief Create filters to identify low-quality cells from RNA-derived metrics.
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
 * See the `IsOutlier` class for implementation details.
 *
 * For the total counts and number of detected features, the outliers are defined after log-transformation of the metrics.
 * This improves resolution at low values and ensures that the defined threshold is not negative.
 */
class PerCellRnaQcFilters {
public:
    /**
     * @param n Number of MADs to use to define outliers, see `IsOutlier` for details.
     *
     * @return A reference to this `PerCellRnaQcFilters` object. 
     */
    PerCellRnaQcFilters& set_nmads(double n = IsOutlier::Defaults::nmads) {
        nmads = n;
        return *this;
    }

private:
    double nmads = IsOutlier::Defaults::nmads;

public:
    /**
     * @brief Thresholds to define outliers on each metric.
     */
    struct Thresholds {
        /**
         * Lower thresholds to define small outliers on the total counts.
         * Each entry contains the threshold used for the corresponding block.
         * For unblocked analyses, this will be of length 1 as all cells are assumed to belong to the same block.
         *
         * Note that, despite the fact that the outliers are defined on the log-scale,
         * these thresholds are reported on the original scale - see the `IsOutlier` class for details.
         */
        std::vector<double> sums;

        /**
         * Lower thresholds to define small outliers on the number of detected features.
         * Each entry contains the threshold used for the corresponding block.
         * For unblocked analyses, this will be of length 1 as all cells are assumed to belong to the same block.
         *
         * Note that, despite the fact that the outliers are defined on the log-scale,
         * these thresholds are reported on the original scale - see the `IsOutlier` class for details.
         */
        std::vector<double> detected;

        /**
         * Upper thresholds to define large outliers on the subset proportions.
         * Each vector corresponds to a feature subset while each entry of the inner vector corresponds to a block of cells.
         * For unblocked analyses, all cells are assumed to belong to a single block, i.e., all inner vectors have length 1.
         */
        std::vector<std::vector<double> > subset_proportions;
    };

public:
    /**
     * Identify low-quality cells as those that have outlier values for QC metrics.
     * This uses QC metrics that are typically computed by the `PerCellRnaQcMetrics` class.
     * 
     * @tparam S Floating point type, used for the sum.
     * @tparam D Integer type, used for the number of detected features.
     * @tparam PPTR Pointer to a floating point type, for the subset proportions.
     * @tparam X Boolean type to indicate whether a cell should be discarded.
     *
     * @param ncells Number of cells.
     * @param[in] sums Pointer to an array of length equal to `ncells`, containing the per-cell sums.
     * @param[in] detected Pointer to an array of length equal to `ncells`, containing the number of detected features for each cell.
     * @param[in] subset_proportions Vector of pointers of length equal to the number of feature subsets.
     * Each pointer corresponds to a feature subset and should point to an array of length equal to `ncells`, containing the proportion of counts in that subset for each cell.
     * @param[out] filter_by_sums Pointer to an output array of length equal to `ncells`,
     * indicating whether a cell should be filtered out due to low count sums.
     * @param[out] filter_by_detected Pointer to an output array of length equal to `ncells`,
     * indicating whether a cell should be filtered out due to a low number of detected features.
     * @param[out] filter_by_subset_proportions Vector of pointers of length equal to the number of feature subsets.
     * Each pointer corresponds to a feature subset and should point to an array of length equal to `ncells`, 
     * indicating whether a cell should be filtered out due to a high proportion of counts for that subset.
     * @param[out] overall_filter Pointer to an output array of length equal to `ncells`,
     * indicating whether a cell should be filtered out for any reason.
     *
     * @return A `Thresholds` object defining the thresholds for each QC metric.
     */
    template<typename S, typename D, typename PPTR, typename X>
    Thresholds run(size_t ncells, const S* sums, const D* detected, std::vector<PPTR> subset_proportions,
                   X* filter_by_sums, X* filter_by_detected, std::vector<X*> filter_by_subset_proportions, X* overall_filter)
    {
        auto fun = [](IsOutlier& outliers, size_t n, auto metric, auto output, auto& buffer) -> auto {
            return outliers.run(n, metric, output, buffer);
        };
        return run_internal(fun,
                            ncells, 
                            sums, 
                            detected, 
                            std::move(subset_proportions),
                            filter_by_sums, 
                            filter_by_detected, 
                            std::move(filter_by_subset_proportions), 
                            overall_filter);
    }

    /**
     * Identify low-quality cells from QC metrics, with blocking during outlier identification.
     * Specifically, outliers are only computed within each block, which is useful when cells in different blocks have different distributions for the QC metrics, e.g., because they were sequenced at different depth.
     * By blocking, we avoid confounding the outlier detection with systematic differences between blocks.
     *
     * @tparam B Pointer to an integer type, to hold the block IDs.
     * @tparam S Floating point type, used for the sum.
     * @tparam D Integer type, used for the number of detected features.
     * @tparam PPTR Pointer to a floating point type, for the subset proportions.
     * @tparam X Boolean type to indicate whether a cell should be discarded.
     *
     * @param ncells Number of cells.
     * @param[in] block Pointer to an array of block identifiers.
     * If provided, the array should be of length equal to `ncells`.
     * Values should be integer IDs in \f$[0, N)\f$ where \f$N\f$ is the number of blocks.
     * This can also be `NULL`, in which case all cells are assumed to belong to the same block.
     * @param[in] sums Pointer to an array of length equal to `ncells`, containing the per-cell sums.
     * @param[in] detected Pointer to an array of length equal to `ncells`, containing the number of detected features for each cell.
     * @param[in] subset_proportions Vector of pointers of length equal to the number of feature subsets.
     * Each pointer corresponds to a feature subset and should point to an array of length equal to `ncells`, containing the proportion of counts in that subset for each cell.
     * @param[out] filter_by_sums Pointer to an output array of length equal to `ncells`,
     * indicating whether a cell should be filtered out due to low count sums.
     * @param[out] filter_by_detected Pointer to an output array of length equal to `ncells`,
     * indicating whether a cell should be filtered out due to a low number of detected features.
     * @param[out] filter_by_subset_proportions Vector of pointers of length equal to the number of feature subsets.
     * Each pointer corresponds to a feature subset and should point to an array of length equal to `ncells`, 
     * indicating whether a cell should be filtered out due to a high proportion of counts for that subset.
     * @param[out] overall_filter Pointer to an output array of length equal to `ncells`,
     * indicating whether a cell should be filtered out for any reason.
     *
     * @return A `Thresholds` object defining the thresholds for each QC metric in each block.
     */
    template<typename B, typename S, typename D, typename PPTR, typename X>
    Thresholds run_blocked(size_t ncells, const B* block, const S* sums, const D* detected, std::vector<PPTR> subset_proportions,
                           X* filter_by_sums, X* filter_by_detected, std::vector<X*> filter_by_subset_proportions, X* overall_filter)
    {
        if (!block) {
            return run(ncells, 
                       sums, 
                       detected, 
                       std::move(subset_proportions),
                       filter_by_sums, 
                       filter_by_detected, 
                       std::move(filter_by_subset_proportions), 
                       overall_filter);
        } else {
            auto by_block = block_indices(ncells, block);
            auto fun = [&](IsOutlier& outliers, size_t n, auto metric, auto output, auto& buffer) -> auto {
                return outliers.run_blocked(n, by_block, metric, output, buffer);
            };

            return run_internal(fun,
                                ncells, 
                                sums, 
                                detected, 
                                std::move(subset_proportions),
                                filter_by_sums, 
                                filter_by_detected, 
                                std::move(filter_by_subset_proportions), 
                                overall_filter);
        }
    }

private:
    template<class Function, typename S, typename D, typename PPTR, typename X>
    Thresholds run_internal(const Function& fun, size_t ncells, const S* sums, const D* detected, std::vector<PPTR> subset_proportions,
                            X* filter_by_sums, X* filter_by_detected, std::vector<X*> filter_by_subset_proportions, X* overall_filter)
    {
        Thresholds output;
        std::vector<double> buffer(ncells);

        // Filtering to remove outliers on the log-sum.
        {
            IsOutlier outliers;
            outliers.set_nmads(nmads).set_upper(false).set_log(true);
            auto res = fun(outliers, ncells, sums, filter_by_sums, buffer);
            output.sums = res.lower;
            std::copy(filter_by_sums, filter_by_sums + ncells, overall_filter);
        }

        // Filtering to remove outliers on the log-detected number.
        {
            IsOutlier outliers;
            outliers.set_nmads(nmads).set_upper(false).set_log(true);
            auto res = fun(outliers, ncells, detected, filter_by_detected, buffer);
            output.detected = res.lower;
            for (size_t i = 0; i < ncells; ++i) {
                overall_filter[i] |= filter_by_detected[i];
            }
        }

        // Filtering to remove outliers on the subset proportions.
        {
            size_t nsubsets = subset_proportions.size();
            if (filter_by_subset_proportions.size() != nsubsets) {
                throw std::runtime_error("mismatching number of input/outputs for subset proportion filters");
            }

            IsOutlier outliers;
            outliers.set_nmads(nmads).set_lower(false);
            output.subset_proportions.resize(nsubsets);

            for (size_t s = 0; s < subset_proportions.size(); ++s) {
                auto dump = filter_by_subset_proportions[s];
                auto res = fun(outliers, ncells, subset_proportions[s], dump, buffer);
                output.subset_proportions[s] = res.upper;

                for (size_t i = 0; i < ncells; ++i) {
                    overall_filter[i] |= dump[i];
                }
            }
        }

        return output;
    }

public:
    /**
     * @brief Results of the QC filtering.
     *
     * Meaningful instances of this object should generally be constructed by calling the `PerCellRnaQcFilters::run()` methods.
     * Empty instances can be default-constructed as placeholders.
     * 
     * @tparam X Boolean type to indicate whether a cell should be discarded.
     */
    template<typename X = uint8_t>
    struct Results {
        /**
         * @cond
         */
        Results() {}

        Results(size_t ncells, int nsubsets) : filter_by_sums(ncells), filter_by_detected(ncells), 
                                               filter_by_subset_proportions(nsubsets, std::vector<X>(ncells)),
                                               overall_filter(ncells) {}
        /**
         * @endcond
         */

        /**
         * Vector of length equal to the number of cells.
         * Entries are set to 1 if the total count in the corresponding cell is a small outlier (indicating that the cell should be considered as low-quality).
         */
        std::vector<X> filter_by_sums;

        /**
         * Vector of length equal to the number of cells.
         * Entries are set to 1 if the number of detected features in the corresponding cell is a small outlier (indicating that the cell should be considered as low-quality).
         */
        std::vector<X> filter_by_detected;

        /**
         * Vector of length equal to the number of feature subsets.
         * Each inner vector corresponds to a feature subset and is of length equal to the number of cells.
         * Entries are set to 1 if the subset proportion in the corresponding cell is a small outlier (indicating that the cell should be considered as low-quality).
         */
        std::vector<std::vector<X> > filter_by_subset_proportions;

        /**
         * Vector of length equal to the number of cells.
         * Entries are set to 1 if the cell is to be considered as low-quality from any of the reasons listed in the previous vectors.
         */
        std::vector<X> overall_filter;

        /**
         * The thresholds used to define outliers for each of the filtering criteria.
         */
        Thresholds thresholds;
    };

public:
    /**
     * Identify low-quality cells from QC metrics, see `run()`.
     *
     * @tparam X Boolean type to indicate whether a cell should be discarded.
     * @tparam S Type of the sum.
     * @tparam D Type of the number of detected features.
     * @tparam PPTR Type of the pointer to the subset proportions.
     *
     * @param ncells Number of cells.
     * @param[in] sums Pointer to an array of length equal to `ncells`, containing the per-cell sums.
     * @param[in] detected Pointer to an array of length equal to `ncells`, containing the number of detected features for each cell.
     * @param[in] subset_proportions Vector of pointers of length equal to the number of feature subsets.
     * Each pointer corresponds to a feature subset and should point to an array of length equal to `ncells`, containing the proportion of counts in that subset for each cell.
     *
     * @return A `Results` object indicating whether a cell should be filtered out for each reason.
     */
    template<typename X = uint8_t, typename S, typename D, typename PPTR>
    Results<X> run(size_t ncells, const S* sums, const D* detected, std::vector<PPTR> subset_proportions) {
        Results<X> output(ncells, subset_proportions.size());
        output.thresholds = run(ncells, 
                                sums, 
                                detected, 
                                std::move(subset_proportions),
                                output.filter_by_sums.data(), 
                                output.filter_by_detected.data(), 
                                vector_to_pointers(output.filter_by_subset_proportions),
                                output.overall_filter.data());
        return output;
    }

    /**
     * Identify low-quality cells from QC metrics with blocking, see `run_blocked()`.
     *
     * @tparam X Boolean type to indicate whether a cell should be discarded.
     * @tparam B Pointer to an integer type, to hold the block IDs.
     * @tparam S Type of the sum.
     * @tparam D Type of the number of detected features.
     * @tparam PPTR Type of the pointer to the subset proportions.
     *
     * @param ncells Number of cells.
     * @param[in] block Optional pointer to an array of block identifiers, see `run_blocked()` for details.
     * @param[in] sums Pointer to an array of length equal to `ncells`, containing the per-cell sums.
     * @param[in] detected Pointer to an array of length equal to `ncells`, containing the number of detected features for each cell.
     * @param[in] subset_proportions Vector of pointers of length equal to the number of feature subsets.
     * Each pointer corresponds to a feature subset and should point to an array of length equal to `ncells`, containing the proportion of counts in that subset for each cell.
     *
     * @return A `Results` object indicating whether a cell should be filtered out for each reason.
     */
    template<typename X = uint8_t, typename B, typename S, typename D, typename PPTR>
    Results<X> run_blocked(size_t ncells, const B* block, const S* sums, const D* detected, std::vector<PPTR> subset_proportions) {
        Results<X> output(ncells, subset_proportions.size());
        output.thresholds = run_blocked(ncells, 
                                        block, 
                                        sums, 
                                        detected, 
                                        std::move(subset_proportions),
                                        output.filter_by_sums.data(), 
                                        output.filter_by_detected.data(), 
                                        vector_to_pointers(output.filter_by_subset_proportions),
                                        output.overall_filter.data());
        return output;
    }

public:
    /**
     * Identify low-quality cells from QC metrics, see `run()` for details.
     *
     * @tparam X Boolean type to indicate whether a cell should be discarded.
     * @tparam R Class that holds the QC metrics, typically a `PerCellRnaQcMetrics::Results`.
     *
     * @param metrics Precomputed QC metrics, typically generated by `PerCellRnaQcMetrics::run`.
     *
     * @return A `Results` object indicating whether a cell should be filtered out for each reason.
     */
    template<typename X=uint8_t, class R>
    Results<X> run(const R& metrics) {
        return run(metrics.sums.size(), metrics.sums.data(), metrics.detected.data(), vector_to_pointers(metrics.subset_proportions));
    }

    /**
     * Identify low-quality cells from QC metrics with blocking, see `run_blocked()` for details.
     *
     * @tparam X Boolean type to indicate whether a cell should be discarded.
     * @tparam R Class that holds the QC metrics, typically a `PerCellRnaQcMetrics::Results`.
     * @tparam B Integer type, to hold the block IDs.
     *
     * @param metrics Precomputed QC metrics, typically generated by `PerCellRnaQcMetrics::run`.
     * @param[in] block Optional pointer to an array of block identifiers, see `run_blocked()` for details.
     *
     * @return A `Results` object indicating whether a cell should be filtered out for each reason.
     */
    template<typename X=uint8_t, class R, typename B>
    Results<X> run_blocked(const R& metrics, const B* block) {
        return run_blocked(metrics.sums.size(), block, metrics.sums.data(), metrics.detected.data(), vector_to_pointers(metrics.subset_proportions));
    }
};

/**
 * An alias for `PerCellRnaQcFilters`, provided for back-compatibility.
 */
typedef PerCellRnaQcFilters PerCellQCFilters;

}

#endif
