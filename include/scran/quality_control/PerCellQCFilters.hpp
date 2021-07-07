#ifndef SCRAN_PER_CELL_QC_FILTERS_H
#define SCRAN_PER_CELL_QC_FILTERS_H

#include <vector>
#include <cstdint>

#include "../utils/vector_to_pointers.hpp"

#include "PerCellQCMetrics.hpp"
#include "IsOutlier.hpp"

/**
 * @file PerCellQCFilters.hpp
 *
 * Create filters to identify low-quality cells.
 */

namespace scran {

/**
 * @brief Create filters to identify low-quality cells.
 *
 * Use an outlier-based approach on common QC metrics (see the `PerCellQCMetrics` class) to identify low-quality cells.
 * Specifically, low-quality cells are defined as those with:
 *
 * - Low total counts, indicating that library preparation or sequencing depth was suboptimal.
 * - Low numbers of detected features, a slightly different flavor of the above reasoning.
 * - High proportions of counts in the mitochondrial (or spike-in) subsets, representing cell damage.
 *
 * Outliers are defined on each metric by counting the number of MADs from the median value across all cells.
 * This assumes that most cells in the experiment are of high (or at least acceptable) quality;
 * any anomalies are indicative of low-quality cells that should be filtered out.
 *
 * For the total counts and number of detected features, the outliers are defined after log-transformation of the metrics.
 * This improves resolution at low values and ensures that the defined threshold is not negative.
 */
class PerCellQCFilters {
public:
    /**
     * Set the number of MADs from the median to define the threshold for outliers.
     *
     * @param n Number of MADs. 
     *
     * @return A reference to this `PerCellQCFilters` object. 
     */
    PerCellQCFilters& set_nmads(double n = 3) {
        outliers.set_nmads(n);
        return *this;
    }

    /**
     * Define blocks of cells, where outliers are only computed within each block.
     * This is useful in cases where cells in different blocks have different distributions for the QC metrics, e.g., because they were sequenced at different depth.
     * In such cases, we avoid confounding the outlier detection with systematic differences between blocks.
     *
     * @tparam V Class of the blocking vector.
     * @param p Blocking vector of length equal to the number of cells.
     * Values should be integer block IDs in \f$[0, N)\f$ where \f$N\f$ is the number of blocks.
     *
     * @return A reference to this `PerCellQCFilters` object. 
     */
    template<class V>
    PerCellQCFilters& set_blocks(const V& p) {
        return set_blocks(p.size(), p.begin());
    }

    /**
     * Define blocks of cells, see `set_blocks(const V& p)`.
     *
     * @tparam SIT Iterator class for the blocking vector.
     *
     * @param n Length of the blocking vector, should be equal to the number of cells.
     * @param p Pointer or iterator to the start of the blocking vector.
     *
     * @return A reference to this `PerCellQCFilters` object. 
     */
    template<typename SIT>
    PerCellQCFilters& set_blocks(size_t n, SIT p) {
        outliers.set_blocks(n, p);
        return *this;
    }

    /**
     * Unset any previous blocking structure.
     *
     * @return A reference to this `PerCellQCFilters` object. 
     */
    PerCellQCFilters& set_blocks() {
        outliers.set_blocks();
        return *this;
    }

public:
    /**
     * @brief Thresholds to define outliers on each metric.
     */
    struct Thresholds {
        /**
         * Lower thresholds to define small outliers on the total counts.
         * Each entry contains the threshold used for the corresponding block.
         *
         * Note that, despite the fact that the outliers are defined on the log-scale,
         * these thresholds are reported on the original scale - see the `IsOutlier` class for details.
         */
        std::vector<double> sums;

        /**
         * Lower thresholds to define small outliers on the number of detected features.
         * Each entry contains the threshold used for the corresponding block.
         *
         * Note that, despite the fact that the outliers are defined on the log-scale,
         * these thresholds are reported on the original scale - see the `IsOutlier` class for details.
         */
        std::vector<double> detected;

        /**
         * Upper thresholds to define large outliers on the subset proportions.
         * Each vector corresponds to a block of cells,
         * while each entry of the inner vector corresponds to a feature subset.
         */
        std::vector<std::vector<double> > subset_proportions;
    };

    /**
     * @brief Results of the QC filtering.
     *
     * @tparam X Boolean type to indicate whether a cell should be discarded.
     */
    template<typename X = uint8_t>
    struct Results {
        /**
         * @param ncells Number of cells.
         * @param nsubsets Number of feature subsets.
         */
        Results(size_t ncells, int nsubsets) : filter_by_sums(ncells), filter_by_detected(ncells), 
                                               filter_by_subset_proportions(nsubsets, std::vector<X>(ncells)),
                                               overall_filter(ncells) {}

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
     * Identify low-quality cells as those that have outlier values for QC metrics.
     * This uses QC metrics that are typically computed by the `PerCellQCMetrics` class.
     *
     * @tparam X Boolean type to indicate whether a cell should be discarded.
     * @tparam S Type of the sum.
     * @tparam D Type of the number of detected features.
     * @tparam PTR Type of the pointer to the subset proportions.
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
     * @param[out] overall_filter  Pointer to an output array of length equal to `ncells`,
     * indicating whether a cell should be filtered out for any reason.
     *
     * @return A `Thresholds` object defining the thresholds for each QC metric.
     */
    template<typename X=uint8_t, typename S=double, typename D=int, typename PTR=const double*>
    Thresholds run(size_t ncells, const S* sums, const D* detected, std::vector<PTR> subset_proportions,
                   X* filter_by_sums, X* filter_by_detected, std::vector<X*> filter_by_subset_proportions, X* overall_filter) 
    {
        Thresholds output;

        // Filtering to remove outliers on the log-sum.
        outliers.set_lower(true).set_upper(false).set_log(true);
        {
            auto res = outliers.run(ncells, sums, filter_by_sums);
            output.sums = res.lower;
            std::copy(filter_by_sums, filter_by_sums + ncells, overall_filter);
        }

        // Filtering to remove outliers on the log-detected number.
        {
            auto res = outliers.run(ncells, detected, filter_by_detected);
            output.detected = res.lower;
            for (size_t i = 0; i < ncells; ++i) {
                overall_filter[i] |= filter_by_detected[i];
            }
        }

        // Filtering to remove outliers on the subset proportions.
        size_t nsubsets = subset_proportions.size();
        if (filter_by_subset_proportions.size() != nsubsets) {
            throw std::runtime_error("mismatching number of input/outputs for subset proportion filters");
        }

        outliers.set_upper(true).set_lower(false).set_log(false);
        output.subset_proportions.resize(nsubsets);

        for (size_t s = 0; s < subset_proportions.size(); ++s) {
            auto dump = filter_by_subset_proportions[s];
            auto res = outliers.run(ncells, subset_proportions[s], dump);
            output.subset_proportions[s] = res.upper;
            for (size_t i = 0; i < ncells; ++i) {
                overall_filter[i] |= dump[i];
            }
        }

        return output;
    }

    /**
     * Identify low-quality cells as those that have outlier values for QC metrics.
     *
     * @tparam X Boolean type to indicate whether a cell should be discarded.
     * @tparam S Type of the sum.
     * @tparam D Type of the number of detected features.
     * @tparam PTR Type of the pointer to the subset proportions.
     *
     * @param ncells Number of cells.
     * @param[in] sums Pointer to an array of length equal to `ncells`, containing the per-cell sums.
     * @param[in] detected Pointer to an array of length equal to `ncells`, containing the number of detected features for each cell.
     * @param[in] subset_proportions Vector of pointers of length equal to the number of feature subsets.
     * Each pointer corresponds to a feature subset and should point to an array of length equal to `ncells`, containing the proportion of counts in that subset for each cell.
     *
     * @return A `Results` object indicating whether a cell should be filtered out for each reason.
     */
    template<typename X=uint8_t, typename S=double, typename D=int, typename PTR=const double*>
    Results<X> run(size_t ncells, const S* sums, const D* detected, std::vector<PTR> subset_proportions) {
        Results<X> output(ncells, subset_proportions.size());
        output.thresholds = run(ncells, sums, detected, std::move(subset_proportions),
                                output.filter_by_sums.data(), output.filter_by_detected.data(), 
                                vector_to_pointers(output.filter_by_subset_proportions),
                                output.overall_filter.data());
        return output;
    }

    /**
     * Identify low-quality cells as those that have outlier values for QC metrics.
     *
     * @tparam X Boolean type to indicate whether a cell should be discarded.
     * @tparam R Class that holds the QC metrics, typically a `PerCellQCMetrics::Results`.
     *
     * @param metrics Precomputed QC metrics, typically generated by `PerCellQCMetrics::run`.
     *
     * @return A `Results` object indicating whether a cell should be filtered out for each reason.
     */
    template<typename X=uint8_t, class R>
    Results<X> run(const R& metrics) {
        return run(metrics.sums.size(), metrics.sums.data(), metrics.detected.data(), vector_to_pointers(metrics.subset_proportions));
    }

private:
    IsOutlier outliers;
};

}

#endif
