#ifndef SCRAN_PER_CELL_QC_METRICS_HPP
#define SCRAN_PER_CELL_QC_METRICS_HPP

#include <vector>
#include <algorithm>
#include <limits>
#include <cstdint>

#include "tatami/base/Matrix.hpp"
#include "tatami/stats/apply.hpp"
#include "../utils/vector_to_pointers.hpp"

/**
 * @file PerCellQCMetrics.hpp
 *
 * @brief Compute typical per-cell quality control metrics.
 */

namespace scran {

/**
 * @brief Compute per-cell quality control metrics from an ADT matrix.
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
 * 
 * This class is implemented as a wrapper around `PerCellQCMetrics`, with the only change being that it defaults to reporting the subset total rather than the proportions.
 * In particular, the `subset_proportions` in `PerCellQCMetrics::run` and `PerCellQCMetrics::Results` will actually hold the subset totals when executed from this class. 
 * Forgive us for this infelicity.
 */
class PerCellAdtQcMetrics : public PerCellQCMetrics {
public:
    /**
     * @cond
     */
    PerCellAdtQcMetrics() {
        runner.set_subset_totals(true);
    }
    /**
     * @endcond
     */
};

}

#endif
