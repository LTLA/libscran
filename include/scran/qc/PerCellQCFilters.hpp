#ifndef SCRAN_PER_CELL_QC_FILTERS_H
#define SCRAN_PER_CELL_QC_FILTERS_H

#include <vector>
#include <cstdint>

#include "../utils/vector_to_pointers.hpp"

#include "PerCellQCMetrics.hpp"
#include "IsOutlier.hpp"

namespace scran {

class PerCellQCFilters {
public:
    PerCellQCFilters& set_nmads(double n = 3) {
        outliers.set_nmads(n);
        return *this;
    }

    template<class V>
    PerCellQCFilters& set_blocks(const V& p) {
        return set_blocks(p.size(), p.begin());
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
    struct Thresholds {
        std::vector<double> sums;
        std::vector<double> detected;
        std::vector<std::vector<double> > subset_proportions;
    };

    template<typename X = uint8_t>
    struct Results {
        Results(size_t ncells, int nsubsets) : filter_by_sums(ncells), filter_by_detected(ncells), 
                                               filter_by_subset_proportions(nsubsets, std::vector<X>(ncells)),
                                               overall_filter(ncells) {}

        std::vector<X> filter_by_sums, filter_by_detected;
        std::vector<std::vector<X> > filter_by_subset_proportions;
        std::vector<X> overall_filter;

        Thresholds thresholds;
    };

public:
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

    template<typename X=uint8_t, typename S=double, typename D=int, typename PTR=const double*>
    Results<X> run(size_t ncells, const S* sums, const D* detected, std::vector<PTR> subset_proportions) {
        Results<X> output(ncells, subset_proportions.size());
        output.thresholds = run(ncells, sums, detected, std::move(subset_proportions),
                                output.filter_by_sums.data(), output.filter_by_detected.data(), 
                                vector_to_pointers(output.filter_by_subset_proportions),
                                output.overall_filter.data());
        return output;
    }

    template<typename X=uint8_t, class R>
    Results<X> run(const R& metrics) {
        return run(metrics.sums.size(), metrics.sums.data(), metrics.detected.data(), vector_to_pointers(metrics.subset_proportions));
    }

private:
    IsOutlier outliers;
};

}

#endif
