#ifndef SCRAN_PER_CELL_ADT_QC_FILTERS_H
#define SCRAN_PER_CELL_ADT_QC_FILTERS_H

#include <vector>
#include <cstdint>

#include "../utils/vector_to_pointers.hpp"

#include "PerCellAdtQcMetrics.hpp"
#include "IsOutlier.hpp"

/**
 * @file PerCellAdtQcFilters.hpp
 *
 * @brief Create filters to identify low-quality cells from ADT data.
 */

namespace scran {

/**
 * @brief Create filters to identify low-quality cells from ADT data.
 *
 * In antibody-derived tag (ADT) count matrices, the QC filtering decisions are slightly different than those for RNA count matrices (see `PerCellQCFilters` for the latter).
 * Here, low-quality cells are defined as those with:
 *
 * - Low numbers of detected features, which indicates that library preparation or sequencing depth was suboptimal.
 *   Even in ADT data, we should expect to detect many features in a cell due to ambient contamination.
 * - Large total counts in the isotype control (IgG) subsets.
 *   The IgG antibodies should not bind to anything, so high coverage is indicative of non-specific binding or antibody conjugates.
 *   
 * Outliers are defined on each metric by counting the number of MADs from the median value across all cells.
 * This assumes that most cells in the experiment are of high (or at least acceptable) quality;
 * any anomalies are indicative of low-quality cells that should be filtered out.
 * See the `IsOutlier` class for implementation details.
 *
 * For the total counts and number of detected features, the outliers are defined after log-transformation of the metrics.
 * This improves resolution at low values and ensures that the defined threshold is not negative.
 */
class PerCellAdtQcFilters {
public:
    /**
     * @brief Default parameter settings.
     */
    struct Defaults {
        /** 
         * See `set_nmads()` for details.
         */
        static constexpr double nmads = 3;

        /**
         * See `set_min_detected_prop()` for details.
         */
        static constexpr double min_detected_prop = 0.5;
    };

private:
    double detected_nmads = Defaults::nmads;
    double subset_nmads = Defaults::nmads;
    double min_detected_prop = Defaults::min_detected_prop;

public:
    /**
     * @param n Number of MADs from the median, to define the threshold for outliers.
     *
     * @return A reference to this `PerCellAdtQcFilters` object. 
     */
    PerCellAdtQcFilters& set_nmads(double n = Defaults::nmads) {
        detected_nmads = n;
        subset_nmads = n;
        return *this;
    }

    /**
     * @param n Number of MADs from the median, to define the threshold for outliers in the number of detected features.
     *
     * @return A reference to this `PerCellAdtQcFilters` object. 
     */
    PerCellAdtQcFilters& set_detected_nmads(double n = Defaults::nmads) {
        detected_nmads = n;
        return *this;
    }

    /**
     * @param n Number of MADs from the median, to define the threshold for outliers in the total count for each subset.
     *
     * @return A reference to this `PerCellAdtQcFilters` object. 
     */
    PerCellAdtQcFilters& set_subset_nmads(double n = Defaults::nmads) {
        subset_nmads = n;
        return *this;
    }

    /**
     * @param n Number of MADs from the median, to define the threshold for outliers.
     *
     * @return A reference to this `PerCellAdtQcFilters` object. 
     */
    PerCellAdtQcFilters& set_min_detected_prop(double m = Defaults::min_detected_prop) {
        min_detected_prop = m;        
        return *this;
    }

public:
    /**
     * @brief Thresholds to define outliers on each metric.
     */
    struct Thresholds {
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
         * Upper thresholds to define large outliers on the subset totals.
         * Each vector corresponds to a feature subset while each entry of the inner vector corresponds to a block of cells.
         * For unblocked analyses, all cells are assumed to belong to a single block, i.e., all inner vectors have length 1.
         */
        std::vector<std::vector<double> > subset_totals;
    };

public:
    /**
     * Identify low-quality cells based on QC metrics computed from the ADT count data, typically using the `PerCellAdtQcMetrics` class. 
     * 
     * @tparam D Integer type, used for the number of detected features.
     * @tparam PPTR Pointer to a floating point type, for the subset proportions.
     * @tparam X Boolean type to indicate whether a cell should be discarded.
     *
     * @param ncells Number of cells.
     * @param[in] detected Pointer to an array of length equal to `ncells`, containing the number of detected features for each cell.
     * @param[in] subset_totals Vector of pointers of length equal to the number of feature subsets.
     * Each pointer corresponds to a feature subset and should point to an array of length equal to `ncells`, containing the proportion of counts in that subset for each cell.
     * @param[out] filter_by_detected Pointer to an output array of length equal to `ncells`,
     * indicating whether a cell should be filtered out due to a low number of detected features.
     * @param[out] filter_by_subset_totals Vector of pointers of length equal to the number of feature subsets.
     * Each pointer corresponds to a feature subset and should point to an array of length equal to `ncells`, 
     * indicating whether a cell should be filtered out due to a high proportion of counts for that subset.
     * @param[out] overall_filter Pointer to an output array of length equal to `ncells`,
     * indicating whether a cell should be filtered out for any reason.
     *
     * @return A `Thresholds` object defining the thresholds for each QC metric.
     */
    template<typename D, typename PPTR, typename X>
    Thresholds run(size_t ncells, const D* detected, std::vector<PPTR> subset_totals,
                   X* filter_by_detected, std::vector<X*> filter_by_subset_totals, X* overall_filter)
    const {
        auto fun = [](IsOutlier& outliers, size_t n, auto metric, auto output) -> auto {
            return outliers.run(n, metric, output);
        };
        return run_internal(fun,
                            ncells, 
                            detected, 
                            std::move(subset_totals),
                            filter_by_detected, 
                            std::move(filter_by_subset_totals), 
                            overall_filter);
    }

    /**
     * Identify low-quality cells from ADT-derived QC metrics, with blocking during outlier identification.
     * Specifically, outliers are only computed within each block, which is useful when cells in different blocks have different distributions for the QC metrics, 
     * e.g., because they were sequenced at different depth.
     * By blocking, we avoid confounding the outlier detection with systematic differences between blocks.
     *
     * @tparam B Pointer to an integer type, to hold the block IDs.
     * @tparam D Integer type, used for the number of detected features.
     * @tparam PPTR Pointer to a floating point type, for the subset proportions.
     * @tparam X Boolean type to indicate whether a cell should be discarded.
     *
     * @param ncells Number of cells.
     * @param[in] block Pointer to an array of block identifiers.
     * If provided, the array should be of length equal to `ncells`.
     * Values should be integer IDs in \f$[0, N)\f$ where \f$N\f$ is the number of blocks.
     * This can also be `NULL`, in which case all cells are assumed to belong to the same block.
     * @param[in] detected Pointer to an array of length equal to `ncells`, containing the number of detected features for each cell.
     * @param[in] subset_totals Vector of pointers of length equal to the number of feature subsets.
     * Each pointer corresponds to a feature subset and should point to an array of length equal to `ncells`, containing the proportion of counts in that subset for each cell.
     * @param[out] filter_by_detected Pointer to an output array of length equal to `ncells`,
     * indicating whether a cell should be filtered out due to a low number of detected features.
     * @param[out] filter_by_subset_totals Vector of pointers of length equal to the number of feature subsets.
     * Each pointer corresponds to a feature subset and should point to an array of length equal to `ncells`, 
     * indicating whether a cell should be filtered out due to a high proportion of counts for that subset.
     * @param[out] overall_filter Pointer to an output array of length equal to `ncells`,
     * indicating whether a cell should be filtered out for any reason.
     *
     * @return A `Thresholds` object defining the thresholds for each QC metric in each block.
     */
    template<typename B, typename D, typename PPTR, typename X>
    Thresholds run_blocked(size_t ncells, const B* block, const D* detected, std::vector<PPTR> subset_totals,
                           X* filter_by_detected, std::vector<X*> filter_by_subset_totals, X* overall_filter)
    const {
        if (!block) {
            return run(ncells, 
                       detected, 
                       std::move(subset_totals),
                       filter_by_detected, 
                       std::move(filter_by_subset_totals), 
                       overall_filter);
        } else {
            auto by_block = block_indices(ncells, block);
            auto fun = [&](IsOutlier& outliers, size_t n, auto metric, auto output) -> auto {
                return outliers.run_blocked(n, by_block, metric, output);
            };

            return run_internal(fun,
                                ncells, 
                                detected, 
                                std::move(subset_totals),
                                filter_by_detected, 
                                std::move(filter_by_subset_totals), 
                                overall_filter);
        }
    }

private:
    template<class Function, typename D, typename PPTR, typename X>
    Thresholds run_internal(const Function& fun, size_t ncells, const D* detected, std::vector<PPTR> subset_totals,
                            X* filter_by_detected, std::vector<X*> filter_by_subset_totals, X* overall_filter)
    const {
        Thresholds output;

        // Filtering to remove outliers on the log-detected number.
        {
            IsOutlier outliers;
            outliers.set_nmads(detected_nmads).set_lower(true).set_upper(false).set_log(true);

            auto res = fun(outliers, ncells, detected, filter_by_detected);
            output.detected = res.lower;
            for (size_t i = 0; i < ncells; ++i) {
                overall_filter[i] |= filter_by_detected[i];
            }
        }

        // Filtering to remove outliers on the subset totals.
        {
            size_t nsubsets = subset_totals.size();
            if (filter_by_subset_totals.size() != nsubsets) {
                throw std::runtime_error("mismatching number of input/outputs for subset filters");
            }

            IsOutlier outliers;
            outliers.set_nmads(subset_nmads).set_upper(true).set_lower(false).set_log(true);
            output.subset_totals.resize(nsubsets);

            for (size_t s = 0; s < subset_totals.size(); ++s) {
                auto dump = filter_by_subset_totals[s];
                auto res = fun(outliers, ncells, subset_totals[s], dump);
                output.subset_totals[s] = res.upper;

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
     * Meaningful instances of this object should generally be constructed by calling the `PerCellAdtQcFilters::run()` methods.
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

        Results(size_t ncells, int nsubsets) : filter_by_detected(ncells), 
                                               filter_by_subset_totals(nsubsets, std::vector<X>(ncells)),
                                               overall_filter(ncells) {}
        /**
         * @endcond
         */

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
        std::vector<std::vector<X> > filter_by_subset_totals;

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
     * Identify low-quality cells from ADT-derived QC metrics, see `run()`.
     *
     * @tparam X Boolean type to indicate whether a cell should be discarded.
     * @tparam D Type of the number of detected features.
     * @tparam PPTR Type of the pointer to the subset proportions.
     *
     * @param ncells Number of cells.
     * @param[in] detected Pointer to an array of length equal to `ncells`, containing the number of detected features for each cell.
     * @param[in] subset_totals Vector of pointers of length equal to the number of feature subsets.
     * Each pointer corresponds to a feature subset and should point to an array of length equal to `ncells`, containing the proportion of counts in that subset for each cell.
     *
     * @return A `Results` object indicating whether a cell should be filtered out for each reason.
     */
    template<typename X = uint8_t, typename D, typename PPTR>
    Results<X> run(size_t ncells, const D* detected, std::vector<PPTR> subset_totals) const {
        Results<X> output(ncells, subset_totals.size());
        output.thresholds = run(ncells, 
                                detected, 
                                std::move(subset_totals),
                                output.filter_by_detected.data(), 
                                vector_to_pointers(output.filter_by_subset_totals),
                                output.overall_filter.data());
        return output;
    }

    /**
     * Identify low-quality cells from ADT-derived QC metrics with blocking, see `run_blocked()`.
     *
     * @tparam X Boolean type to indicate whether a cell should be discarded.
     * @tparam B Pointer to an integer type, to hold the block IDs.
     * @tparam D Type of the number of detected features.
     * @tparam PPTR Type of the pointer to the subset proportions.
     *
     * @param ncells Number of cells.
     * @param[in] block Optional pointer to an array of block identifiers, see `run_blocked()` for details.
     * @param[in] detected Pointer to an array of length equal to `ncells`, containing the number of detected features for each cell.
     * @param[in] subset_totals Vector of pointers of length equal to the number of feature subsets.
     * Each pointer corresponds to a feature subset and should point to an array of length equal to `ncells`, containing the proportion of counts in that subset for each cell.
     *
     * @return A `Results` object indicating whether a cell should be filtered out for each reason.
     */
    template<typename X = uint8_t, typename B, typename D, typename PPTR>
    Results<X> run_blocked(size_t ncells, const B* block, const D* detected, std::vector<PPTR> subset_totals) const {
        Results<X> output(ncells, subset_totals.size());
        output.thresholds = run_blocked(ncells, 
                                        block, 
                                        detected, 
                                        std::move(subset_totals),
                                        output.filter_by_detected.data(), 
                                        vector_to_pointers(output.filter_by_subset_totals),
                                        output.overall_filter.data());
        return output;
    }

public:
    /**
     * Identify low-quality cells from ADT-derived QC metrics, see `run()` for details.
     *
     * @tparam X Boolean type to indicate whether a cell should be discarded.
     * @tparam R Class that holds the QC metrics, typically a `PerCellAdtQcMetrics::Results`.
     *
     * @param metrics Precomputed QC metrics, typically generated by `PerCellAdtQcMetrics::run`.
     *
     * @return A `Results` object indicating whether a cell should be filtered out for each reason.
     */
    template<typename X=uint8_t, class R>
    Results<X> run(const R& metrics) const {
        return run(metrics.detected.size(), metrics.detected.data(), vector_to_pointers(metrics.subset_totals));
    }

    /**
     * Identify low-quality cells from QC metrics with blocking, see `run_blocked()` for details.
     *
     * @tparam X Boolean type to indicate whether a cell should be discarded.
     * @tparam R Class that holds the QC metrics, typically a `PerCellAdtQcMetrics::Results`.
     * @tparam B Integer type, to hold the block IDs.
     *
     * @param metrics Precomputed QC metrics, typically generated by `PerCellAdtQcMetrics::run`.
     * @param[in] block Optional pointer to an array of block identifiers, see `run_blocked()` for details.
     *
     * @return A `Results` object indicating whether a cell should be filtered out for each reason.
     */
    template<typename X=uint8_t, class R, typename B>
    Results<X> run_blocked(const R& metrics, const B* block) const {
        return run_blocked(metrics.detected.size(), block, metrics.detected.data(), vector_to_pointers(metrics.subset_totals));
    }
};

}

#endif
