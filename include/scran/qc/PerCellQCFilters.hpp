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
        filter_by_sums = p;
        return *this;
    }

    PerCellQCFilters& set_filter_by_detected(X* p = NULL) {
        filter_by_detected = p;
        return *this;
    }

    PerCellQCFilters& set_filter_by_subset_proportions(std::vector<X*> ptrs) {
        filter_by_subset_proportions = ptrs;
        return *this;
    }

    PerCellQCFilters& set_filter_by_subset_proportions() {
        filter_by_subset_proportions.clear();
        return *this;
    }        

    PerCellQCFilters& set_overall_filter(X* p = NULL) {
        overall_filter = p;
        return *this;
    }

public:
    const X* get_filter_by_sums() const {
        if (filter_by_sums != NULL) {
            return filter_by_sums;
        } else {
            return internal_filter_by_sums.data(); 
        }
    }

    const X* get_filter_by_detected() const {
        if (filter_by_detected != NULL) {
            return filter_by_detected;
        } else {
            return internal_filter_by_detected.data(); 
        }
    }

    std::vector<const X*> get_filter_by_subset_proportions() const {
        if (filter_by_subset_proportions.size()) {
            return std::vector<const X*>(filter_by_subset_proportions.begin(), filter_by_subset_proportions.end());
        } else {
            std::vector<const X*> output;
            for (const auto& x : internal_filter_by_subset_proportions) {
                output.push_back(x.data());
            }
            return output;
        }
    }

    const X* get_overall_filter() const {
        if (overall_filter != NULL) {
            return overall_filter;
        } else {
            return internal_overall_filter.data(); 
        }
    }

public:
    const std::vector<double>& get_sums_thresholds() const {
        return sums_thresholds;
    }

    const std::vector<double>& get_detected_thresholds() const {
        return detected_thresholds;
    }

    const std::vector<std::vector<double> >& get_subset_proportions_thresholds() const {
        return subset_proportions_thresholds;
    }

public:
    template<typename S=double, typename N=int, typename P=double>
    PerCellQCFilters& run(size_t ncells, const S* sums, const N* detected, const std::vector<const P*>& subset_proportions) {
        // Setting up the overall filter.
        auto overall = overall_filter;
        if (overall_filter == NULL) {
            internal_overall_filter.resize(ncells);
            overall = internal_overall_filter.data();
        }

        // Setting up the sums.
        auto filter_by_sums_out = filter_by_sums;
        if (filter_by_sums == NULL) {
            internal_filter_by_sums.resize(ncells);
            filter_by_sums_out = internal_filter_by_sums.data();
        }

        // Setting up the detected buffer.
        auto filter_by_detected_out = filter_by_detected;
        if (filter_by_detected == NULL) {
            internal_filter_by_detected.resize(ncells);
            filter_by_detected_out = internal_filter_by_detected.data();
        }

        // Setting up buffers for the subset proportions.
        size_t nsubsets = subset_proportions.size();
        if (filter_by_subset_proportions.size() != 0 && filter_by_subset_proportions.size() != nsubsets) {
            throw std::runtime_error("mismatching number of input/outputs for subset proportion filters");
        }

        std::vector<X*> filter_by_subset_proportions_out(nsubsets);
        internal_filter_by_subset_proportions.resize(nsubsets);

        for (size_t s = 0; s < nsubsets; ++s) {
            if (filter_by_subset_proportions.size()) {
                filter_by_subset_proportions_out[s] = filter_by_subset_proportions[s];
            } else {
                auto& internal = internal_filter_by_subset_proportions[s];
                internal.resize(ncells);
                filter_by_subset_proportions_out[s] = internal.data();
            }
        }

        internal_run(ncells, sums, detected, subset_proportions, 
            filter_by_sums_out, filter_by_detected_out, filter_by_subset_proportions_out, overall);
        return *this;
    }

    template<typename S=double, typename N=int, typename P=double>
    PerCellQCFilters& run(size_t ncells, const PerCellQCMetrics<S, N, P>& metrics) {
        run(ncells, metrics.get_sums(), metrics.get_detected(), metrics.get_subset_proportions());
    }

private:
    template<typename S=double, typename N=int, typename P=double>
    void internal_run(size_t ncells, const S* sums, const N* detected, const std::vector<const P*>& subset_proportions,
        X * filter_by_sums_out, X * filter_by_detected_out, std::vector<X*>& filter_by_subset_proportions_out, X* overall) 
    {
        // Filtering to remove outliers on the log-sum.
        outliers.set_upper(false).set_log(true).set_outliers(filter_by_sums_out).run(ncells, sums);
        sums_thresholds = outliers.get_lower_thresholds();
        {
            auto ptr = outliers.get_outliers();
            std::copy(ptr, ptr + ncells, overall);
        }

        // Filtering to remove outliers on the log-detected number.
        outliers.set_outliers(filter_by_detected_out).run(ncells, detected);
        detected_thresholds = outliers.get_lower_thresholds();
        {
            auto ptr = outliers.get_outliers();
            for (size_t i = 0; i < ncells; ++i) {
                overall[i] |= ptr[i];
            }
        }

        // Filtering to remove outliers on the subset proportions.
        outliers.set_upper(true).set_lower(false).set_log(false);
        subset_proportions_thresholds.resize(subset_proportions.size());

        for (size_t s = 0; s < subset_proportions.size(); ++s) {
            outliers.set_outliers(filter_by_subset_proportions_out[s]).run(ncells, subset_proportions[s]);
            subset_proportions_thresholds[s] = outliers.get_upper_thresholds();
            {
                auto ptr = outliers.get_outliers();
                for (size_t i = 0; i < ncells; ++i) {
                    overall[i] |= ptr[i];
                }
            }
        }

        return;
    }

private:
    X* filter_by_sums = NULL;
    X* filter_by_detected = NULL;
    std::vector<X*> filter_by_subset_proportions;
    X* overall_filter;

    std::vector<X> internal_filter_by_sums, internal_filter_by_detected;
    std::vector<std::vector<X > > internal_filter_by_subset_proportions;
    std::vector<X> internal_overall_filter;

    std::vector<double> sums_thresholds;
    std::vector<double> detected_thresholds;
    std::vector<std::vector<double> > subset_proportions_thresholds;

    IsOutlier<X> outliers;
};

}

#endif
