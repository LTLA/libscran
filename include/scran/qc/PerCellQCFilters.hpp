#ifndef SCRAN_PER_CELL_QC_FILTERS_H
#define SCRAN_PER_CELL_QC_FILTERS_H

#include <vector>
#include <cstdint>

#include "PerCellQCMetrics.hpp"
#include "IsOutlier.hpp"

namespace scran {

template<typename X = uint8_t>
class PerCellQCFilters {
public:
    PerCellQCFilters& set_nmads(double n = 3) {
        outliers.set_nmads(n);
        return *this;
    }

    template<typename SIT>
    PerCellQCFilters& set_blocks(size_t n, SIT p) {
        outliers.set_blocks(n, p);
        return *this;
    }

    PerCellQCFilters& set_blocks() {
        outliers.set_blocks();
        return *this;
    }

public:
    PerCellQCFilters& set_filter_by_sums(X* p = NULL) {
        stored_filter_by_sums = p;
        return *this;
    }

    PerCellQCFilters& set_filter_by_detected(X* p = NULL) {
        stored_filter_by_detected = p;
        return *this;
    }

    PerCellQCFilters& set_filter_by_subset_proportions(std::vector<X*> ptrs) {
        stored_filter_by_subset_proportions = ptrs;
        return *this;
    }

    PerCellQCFilters& set_filter_by_subset_proportions() {
        stored_filter_by_subset_proportions.clear();
        return *this;
    }        

    PerCellQCFilters& set_overall_filter(X* p = NULL) {
        stored_overall_filter = p;
        return *this;
    }

public:
    struct Results {
        Results(size_t n, PerCellQCFilters<X>& parent) : ncells(n) {
            // Setting up the overall filter.
            overall_filter = parent.stored_overall_filter;
            if (overall_filter == NULL) {
                internal_overall_filter.resize(ncells);
                overall_filter = internal_overall_filter.data();
            }

            // Setting up the sums.
            filter_by_sums = parent.stored_filter_by_sums;
            if (filter_by_sums == NULL) {
                internal_filter_by_sums.resize(ncells);
                filter_by_sums = internal_filter_by_sums.data();
            }

            // Setting up the detected buffer.
            filter_by_detected = parent.stored_filter_by_detected;
            if (filter_by_detected == NULL) {
                internal_filter_by_detected.resize(ncells);
                filter_by_detected = internal_filter_by_detected.data();
            }

            return;
        }

        size_t ncells;
        X* filter_by_sums = NULL;
        X* filter_by_detected = NULL;
        std::vector<X*> filter_by_subset_proportions;
        X* overall_filter = NULL;
       
        std::vector<double> sums_thresholds;
        std::vector<double> detected_thresholds;
        std::vector<std::vector<double> > subset_proportions_thresholds;
    private:
        friend PerCellQCFilters<X>;
        std::vector<X> internal_filter_by_sums, internal_filter_by_detected;
        std::vector<std::vector<X > > internal_filter_by_subset_proportions;
        std::vector<X> internal_overall_filter;
    };

    template<typename S=double, typename D=int, typename PVEC=std::vector<const double*> >
    Results run(size_t ncells, const S* sums, const D* detected, const PVEC& subset_proportions) {
        Results output(ncells, *this);
        X* overall = output.overall_filter;

        // Filtering to remove outliers on the log-sum.
        outliers.set_lower(true).set_upper(false).set_log(true);
        {
            auto dump = output.filter_by_sums;
            auto res = outliers.set_outliers(dump).run(ncells, sums);
            output.sums_thresholds = res.lower_thresholds;
            std::copy(dump, dump + ncells, overall);
        }

        // Filtering to remove outliers on the log-detected number.
        {
            auto dump = output.filter_by_detected;
            auto res = outliers.set_outliers(dump).run(ncells, detected);
            output.detected_thresholds = res.lower_thresholds;
            for (size_t i = 0; i < ncells; ++i) {
                overall[i] |= dump[i];
            }
        }

        // Filtering to remove outliers on the subset proportions.
        size_t nsubsets = subset_proportions.size();
        if (stored_filter_by_subset_proportions.size()) {
            if (stored_filter_by_subset_proportions.size() != nsubsets) {
                throw std::runtime_error("mismatching number of input/outputs for subset proportion filters");
            }
            output.filter_by_subset_proportions = stored_filter_by_subset_proportions;
        } else {
            output.internal_filter_by_subset_proportions.resize(nsubsets);
            output.filter_by_subset_proportions.resize(nsubsets);
            for (size_t s = 0; s < nsubsets; ++s) {
                auto& internal = output.internal_filter_by_subset_proportions[s];
                internal.resize(ncells);
                output.filter_by_subset_proportions[s] = internal.data();
            }
        }

        outliers.set_upper(true).set_lower(false).set_log(false);
        output.subset_proportions_thresholds.resize(subset_proportions.size());

        for (size_t s = 0; s < subset_proportions.size(); ++s) {
            auto dump = output.filter_by_subset_proportions[s];
            auto res = outliers.set_outliers(dump).run(ncells, subset_proportions[s]);
            output.subset_proportions_thresholds[s] = res.upper_thresholds;
            for (size_t i = 0; i < ncells; ++i) {
                overall[i] |= dump[i];
            }
        }

        return output;
    }

    template<class R>
    Results run(const R& metrics) {
        return run(metrics.ncells, metrics.sums, metrics.detected, metrics.subset_proportions);
    }

private:
    X* stored_filter_by_sums = NULL;
    X* stored_filter_by_detected = NULL;
    std::vector<X*> stored_filter_by_subset_proportions;
    X* stored_overall_filter = NULL;

    IsOutlier<X> outliers;
};

}

#endif
