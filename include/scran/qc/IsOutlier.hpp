#ifndef SCRAN_IS_OUTLIER_H
#define SCRAN_IS_OUTLIER_H

#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <cstdint>

namespace scran {

template<typename X=uint8_t>
class IsOutlier {
public:
    IsOutlier& set_log(bool l = false) {
        log = l;
        return *this;
    }

    IsOutlier& set_lower(bool l = true) {
        lower = l;
        return *this;
    }

    IsOutlier& set_upper(bool u = true) {
        upper = u;
        return *this;
    }

    IsOutlier& set_nmads(double n = 3) {
        nmads = n;
        return *this;
    }

    template<typename SIT>
    IsOutlier& set_blocks(size_t n, SIT p) {
        int ngroups = (n ? *std::max_element(p, p + n) + 1 : 1);

        by_group.resize(ngroups);
        for (auto& g : by_group) { 
            g.clear();
        }

        for (size_t i = 0; i < n; ++i, ++p) {
            by_group[*p].push_back(i);
        }

        return *this;
    }

    IsOutlier& set_blocks() {
        by_group.clear();
        return *this;
    }

    IsOutlier& set_outliers(X* p = NULL) {
        stored_outliers = p;
        return *this;
    }

public:
    struct Results {
        Results(size_t n, int nsub, IsOutlier<X>& parent) : lower_thresholds(std::max(1, nsub)), upper_thresholds(std::max(1, nsub)) {
            outliers = parent.stored_outliers;
            if (outliers == NULL) {
                internal_outliers.resize(n);
                outliers = internal_outliers.data();
            }
            return;
        }

        X* outliers = NULL;
        std::vector<double> lower_thresholds, upper_thresholds;
    private:
        std::vector<X> internal_outliers;
    };

public:
    template<typename T>
    Results run(size_t n, const T* metrics) {
        Results output(n, by_group.size(), *this);
        auto out = output.outliers;
        buffer.resize(n);

        if (by_group.size() == 0) {
            std::copy(metrics, metrics + n, buffer.data());

            double& lthresh = output.lower_thresholds[0];
            double& uthresh = output.upper_thresholds[0];
            compute_thresholds(n, buffer.data(), lthresh, uthresh);

            auto copy = metrics;
            for (size_t i = 0; i < n; ++i, ++copy, ++out) {
                const double val = *copy;
                *out = (val < lthresh || val > uthresh);
            }
        } else {
            for (size_t g = 0; g < by_group.size(); ++g) {
                auto copy = buffer.data();
                const auto& curgroup = by_group[g];
                for (auto s : curgroup) {
                    *copy = metrics[s];
                    ++copy;
                }

                double& lthresh = output.lower_thresholds[g];
                double& uthresh = output.upper_thresholds[g];
                compute_thresholds(curgroup.size(), buffer.data(), lthresh, uthresh);

                for (auto s : curgroup) {
                    const double val = metrics[s];
                    out[s] = (val < lthresh || val > uthresh);
                }
            }
        }

        return output;
    }

private:
    void compute_thresholds(size_t n, double* ptr, double& lower_threshold, double& upper_threshold) {
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
    X* stored_outliers = NULL;
    std::vector<double> buffer;
    std::vector<std::vector<size_t> > by_group;
};

}

#endif
