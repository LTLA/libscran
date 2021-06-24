#ifndef SCRAN_IS_OUTLIER_H
#define SCRAN_IS_OUTLIER_H

#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <cstdint>

#include "../utils/block_indices.hpp"

namespace scran {

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

public:
    template<class V>
    IsOutlier& set_blocks(const V& p) {
        return set_blocks(p.size(), p.begin());
    }

    template<typename SIT>
    IsOutlier& set_blocks(size_t n, SIT p) {
        group_ncells = n;
        block_indices(n, p, by_group);
        return *this;
    }

    IsOutlier& set_blocks() {
        group_ncells = 0;
        by_group.clear();
        return *this;
    }

public:
    struct Thresholds {
        Thresholds(int nblocks=0) : lower(nblocks), upper(nblocks) {}
        std::vector<double> lower, upper;
    };

    template<typename X = uint8_t>
    struct Results {
        Results(size_t ncells) : outliers(ncells) {}
        std::vector<X> outliers;
        Thresholds thresholds;
    };

public:
    template<typename X = uint8_t, class V>
    Results<X> run(const V& metrics) {
        Results<X> output(metrics.size()); 
        output.thresholds = run(metrics.size(), metrics.data(), output.outliers.data());
        return output;
    }

    template<typename X = uint8_t, typename T>
    Results<X> run(size_t n, const T* metrics) {
        Results<X> output(n); 
        output.thresholds = run(n, metrics, output.outliers.data());
        return output;
    }

    template<typename X = uint8_t, typename T>
    Thresholds run(size_t n, const T* metrics, X* outliers) {
        Thresholds output(std::max(static_cast<size_t>(1), by_group.size()));
        buffer.resize(n);

        if (by_group.size() == 0) {
            std::copy(metrics, metrics + n, buffer.data());

            double& lthresh = output.lower[0];
            double& uthresh = output.upper[0];
            compute_thresholds(n, buffer.data(), lthresh, uthresh);

            auto copy = metrics;
            for (size_t i = 0; i < n; ++i, ++copy, ++outliers) {
                const double val = *copy;
                *outliers = (val < lthresh || val > uthresh);
            }
        } else {
            if (group_ncells != n) {
                throw std::runtime_error("length of grouping vector and number of cells are not equal");
            }

            for (size_t g = 0; g < by_group.size(); ++g) {
                auto copy = buffer.data();
                const auto& curgroup = by_group[g];
                for (auto s : curgroup) {
                    *copy = metrics[s];
                    ++copy;
                }

                double& lthresh = output.lower[g];
                double& uthresh = output.upper[g];
                compute_thresholds(curgroup.size(), buffer.data(), lthresh, uthresh);

                for (auto s : curgroup) {
                    const double val = metrics[s];
                    outliers[s] = (val < lthresh || val > uthresh);
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
    std::vector<double> buffer;

    std::vector<std::vector<size_t> > by_group;
    size_t group_ncells = 0;
};

}

#endif
