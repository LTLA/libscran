#ifndef SCRAN_IS_OUTLIER_H
#define SCRAN_IS_OUTLIER_H

#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>

namespace scran {

class IsOutlier {
public:
    IsOutlier(size_t n) : buffer(n), lower_thresholds(1), upper_thresholds(1) {}

    IsOutlier& set_log(bool l) {
        log = l;
        return *this;
    }

    IsOutlier& set_lower(bool l) {
        lower = l;
        return *this;
    }

    IsOutlier& set_upper(bool u) {
        upper = u;
        return *this;
    }

    IsOutlier& set_nmads(double n) {
        nmads = n;
        return *this;
    }

    template<typename S = int>
    IsOutlier& set_blocks(const S* b) {
        if (b != NULL) {
            int ngroups = (buffer.size() ? *std::max_element(b, b + buffer.size()) + 1 : 1);

            by_group.resize(ngroups);
            for (auto& g : by_group) { 
                g.clear();
            }

            for (size_t i = 0; i < buffer.size(); ++i, ++b) {
                by_group[*b].push_back(i);                                
            }
            lower_thresholds.resize(ngroups);
            upper_thresholds.resize(ngroups);
        } else {
            by_group.clear();
            lower_thresholds.resize(1);
            upper_thresholds.resize(1);
        }
        return *this;
    }

public:
    template<typename T, typename X>
    void find_outliers(const T* metrics, X* outliers) {
        size_t n = buffer.size();

        if (by_group.size() == 0) {
            std::copy(metrics, metrics + n, buffer.data());

            double& lthresh = lower_thresholds[0];
            double& uthresh = upper_thresholds[0];
            compute_thresholds(buffer.data(), n, lthresh, uthresh);

            auto copy = metrics;
            for (size_t i = 0; i < n; ++i, ++copy, ++outliers) {
                const double val = *copy;
                *outliers = (val < lthresh || val > uthresh);
            }
        } else {
            for (size_t g = 0; g < by_group.size(); ++g) {
                auto copy = buffer.data();
                const auto& curgroup = by_group[g];
                for (auto s : curgroup) {
                    *copy = metrics[s];
                    ++copy;
                }

                double& lthresh = lower_thresholds[g];
                double& uthresh = upper_thresholds[g];
                compute_thresholds(buffer.data(), curgroup.size(), lthresh, uthresh);

                for (auto s : curgroup) {
                    const double val = metrics[s];
                    outliers[s] = (val < lthresh || val > uthresh);
                }
            }
        }
        return;
    }

    template<typename T>
    void find_outliers(const T* metrics) {
        outlier_store.resize(buffer.size());
        find_outliers(metrics, outlier_store.data());
        return;
    }

public:
    const std::vector<int>& get_outliers() const {
        return outlier_store;
    }

    const std::vector<double>& get_upper_thresholds() const {
        return upper_thresholds; 
    }

    const std::vector<double>& get_lower_thresholds() const {
        return lower_thresholds; 
    }

private:
    void compute_thresholds(double* ptr, size_t n, double& lower_threshold, double& upper_threshold) {
        if (!n) {
            lower_threshold = std::numeric_limits<double>::quiet_NaN();
            upper_threshold = std::numeric_limits<double>::quiet_NaN();
            return;
        }

        if (log) {
            auto copy = ptr;
            for (size_t i = 0; i < n; ++i, ++copy) {
                if (*copy > 0) {
                    *copy = std::log(*copy);
                } else if (*copy == 0) {
                    *copy = -std::numeric_limits<double>::infinity();
                } else {
                    throw std::runtime_error("cannot log-transform negative values");
                }
            }
        }

        // First, getting the median.
        size_t halfway = n / 2;
        double medtmp = compute_median(ptr, halfway, n);

        if (log && std::isinf(medtmp)) {
            if (lower) {
                lower_threshold = 0;
            } else {
                lower_threshold = -std::numeric_limits<double>::infinity();
            }

            if (upper) {
                upper_threshold = 0;
            } else {
                upper_threshold = std::numeric_limits<double>::infinity();
            }

            return;
        }

        // Now getting the MAD. No need to protect against -inf here;
        // if the median is finite, the MAD must also be finite.
        auto copy = ptr;
        for (size_t i = 0; i < n; ++i, ++copy) {
            *copy = std::abs(*copy - medtmp);
        }
        double madtmp = compute_median(ptr, halfway, n);

        madtmp *= 1.4826; // for equivalence with the standard deviation under normality.

        // Computing the outputs.
        if (lower) {
            lower_threshold = medtmp - nmads * madtmp;
            if (log) {
                lower_threshold = std::exp(lower_threshold);
            }
        } else {
            lower_threshold = -std::numeric_limits<double>::infinity();
        }

        if (upper) {
            upper_threshold = medtmp + nmads * madtmp;
            if (log) {
                upper_threshold = std::exp(upper_threshold);
            }
        } else {
            upper_threshold = std::numeric_limits<double>::infinity();
        }

        return;
    }

    static double compute_median (double* ptr, size_t halfway, size_t n) {
        std::nth_element(ptr, ptr + halfway, ptr + n);
        double medtmp = *(ptr + halfway);

        if (n % 2 == 0) {
            std::nth_element(ptr, ptr + halfway - 1, ptr + n);
            medtmp = (medtmp + *(ptr + halfway - 1))/2;
        }

        return medtmp;
    }

private:
    double nmads = 3;
    bool lower = true, upper = true, log = false;

    std::vector<double> buffer;
    std::vector<std::vector<size_t> > by_group;

private:
    std::vector<double> lower_thresholds, upper_thresholds;
    std::vector<int> outlier_store;
};

}

#endif
