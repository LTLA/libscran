#ifndef SCRAN_PAIRWISE_AUC_HPP
#define SCRAN_PAIRWISE_AUC_HPP

#include "../utils/macros.hpp"

#include <vector>
#include <algorithm>

namespace scran {

namespace differential_analysis {

typedef std::vector<std::pair<double, int> > PairedStore;

inline void compute_pairwise_auc(PairedStore& input, const std::vector<int>& num_zeros, const std::vector<int>& totals, double* output, bool normalize) {
    size_t ngroups = num_zeros.size();
    std::sort(input.begin(), input.end());
    std::vector<double> less_than(ngroups), equal(ngroups);

    std::vector<double*> outputs(ngroups, output);
    for (size_t i = 0; i < ngroups; ++i) {
        outputs[i] += i * ngroups;
    }

    auto inner_loop = [&](size_t& pos) -> void {
        const auto& current = input[pos];

        ++pos;
        bool tied = false;
        while (pos != input.size() && input[pos].first == current.first) {
            tied = true;
            ++equal[input[pos].second];
            ++pos;
        }

        if (tied) {
            ++equal[current.second]; // contribution of the current (tied) observation.

            for (size_t l = 1; l < ngroups; ++l) { // starting from 1 as zero has no work for the g < l condition anyway.
                if (equal[l]) {
                    auto outptr = outputs[l];
                    for (size_t g = 0; g < l; ++g) {
                        outptr[g] += equal[l] * (less_than[g] + 0.5 * equal[g]);
                    }
                }
            }

            for (size_t l = 0; l < ngroups; ++l) {
                less_than[l] += equal[l];
                equal[l] = 0;
            }
        } else {
            auto outptr = outputs[current.second];
            for (size_t g = 0; g < current.second; ++g) {
                outptr[g] += less_than[g];
            }
            ++less_than[current.second];
        }
    };

    size_t pos = 0;

    // Values < 0.
    while (pos != input.size() && input[pos].first < 0) {
        inner_loop(pos);
    }

    // Values == 0. 
    for (size_t l = 1; l < ngroups; ++l) { // starting from 1, see above.
        if (num_zeros[l]) {
            auto outptr = outputs[l];
            for (size_t g = 0; g < l; ++g) {
                outptr[g] += num_zeros[l] * (less_than[g] + 0.5 * num_zeros[g]);
            }
        }
    }

    for (size_t l = 0; l < ngroups; ++l) {
        less_than[l] += num_zeros[l];
    }

    // Values > 0. We expect that all values in 'input' are already non-zero;
    // any zeros should have been moved into the 'num_zeros' already, e.g.,
    // as done in the ComplexPerRowFactory::compute() method.
    while (pos != input.size()) {
        inner_loop(pos);
    }

    // Filling in the other side.
    for (size_t l = 1; l < ngroups; ++l) { // startinf rom 1, see above.
        for (size_t g = 0; g < l; ++g) {
            double prod = static_cast<double>(totals[l]) * static_cast<double>(totals[g]);
            outputs[g][l] = prod - outputs[l][g];

            if (normalize) {
                if (prod) {
                    outputs[l][g] /= prod;
                    outputs[g][l] /= prod;
                } else {
                    outputs[l][g] = std::numeric_limits<double>::quiet_NaN();
                    outputs[g][l] = std::numeric_limits<double>::quiet_NaN();
                }
            }
        }
    }
}

inline void compute_pairwise_auc(PairedStore& input, const std::vector<int>& num_zeros, const std::vector<int>& totals, double* output, double threshold, bool normalize) {
    size_t ngroups = num_zeros.size();
    std::sort(input.begin(), input.end());
    std::vector<double> less_than(ngroups), equal(ngroups);

    std::vector<double*> outputs(ngroups, output);
    for (size_t i = 0; i < ngroups; ++i) {
        outputs[i] += i * ngroups;
    }

    auto inner_loop = [&](size_t& pos, size_t& comp) -> void {
        const auto& current = input[pos];
        double limit = current.first - threshold;

        // Hunting all entities less than the limit.
        while (comp != input.size() && input[comp].first < limit) {
            ++less_than[input[comp].second];
            ++comp;
        }

        // Checking for ties with the limit.
        bool tied = false;
        while (comp != input.size() && input[comp].first == limit) {
            tied = true;
            ++equal[input[comp].second];
            ++comp;
        }

        if (tied) {
            do {
                auto outptr = outputs[input[pos].second];
                for (size_t g = 0; g < ngroups; ++g) {
                    outptr[g] += less_than[g] + 0.5 * equal[g];
                }
                ++pos;
            } while (pos != input.size() && input[pos].first == current.first);

            for (size_t l = 0; l < ngroups; ++l) {
                less_than[l] += equal[l];
                equal[l] = 0;
            }
        } else {
            do {
                auto outptr = outputs[input[pos].second];
                for (size_t g = 0; g < ngroups; ++g) {
                    outptr[g] += less_than[g];
                }
                ++pos;
            } while (pos != input.size() && input[pos].first == current.first);
        }
    };

//    auto print_output = [&]() -> void {
//        std::cout << "Less than is: ";
//        for (size_t m = 0; m < ngroups; ++m) {
//            std::cout << " " << less_than[m];
//        }
//        std::cout << std::endl;
//        std::cout << "Equal is: ";
//        for (size_t m = 0; m < ngroups; ++m) {
//            std::cout << " " << equal[m];
//        }
//        std::cout << std::endl;
//        std::cout << "AUCs are:\n";
//        for (size_t n = 0; n < ngroups; ++n) {
//            std::cout << "  Group " << n << ": ";
//            for (size_t m = 0; m < ngroups; ++m) {
//                std::cout << " " << outputs[n][m];
//            }
//            std::cout << std::endl;
//        }
//    };

    size_t pos = 0, comp = 0;

    while (pos != input.size() && input[pos].first < 0) {
        inner_loop(pos, comp);
    }

    // Adding the contribution of zeros (in terms of the things they're greater than).
    // This effectively replicates the inner_loop but accounts for lots of zeros.
    // Again, we expect that all values in 'input' are already non-zero;
    // any zeros should have been moved into the 'num_zeros' already.
    {
        while (comp != input.size() && input[comp].first < -threshold) {
            ++less_than[input[comp].second];
            ++comp;
        }

        for (size_t l = 0; l < ngroups; ++l) {
            if (num_zeros[l]) {
                auto outptr = outputs[l];
                for (size_t g = 0; g < ngroups; ++g) {
                    outptr[g] += less_than[g] * num_zeros[l];
                }
            }
        }

        // Handling ties at the threshold boundary...
        bool tied = false;
        while (comp != input.size() && input[comp].first == -threshold) {
            tied = true;
            ++equal[input[comp].second];
            ++comp;
        }

        if (tied) {
            for (size_t l = 0; l < ngroups; ++l) {
                auto outptr = outputs[l];
                if (num_zeros[l]) {
                    for (size_t g = 0; g < ngroups; ++g) {
                        outptr[g] += 0.5 * equal[g] * num_zeros[l];
                    }
                }
            }
            for (size_t l = 0; l < ngroups; ++l) {
                less_than[l] += equal[l];
                equal[l] = 0;
            }
        }

        // Or to each other, if the threshold is zero.
        if (threshold == 0) {
            for (size_t l = 0; l < ngroups; ++l) {
                if (num_zeros[l]) {
                    auto outptr = outputs[l];
                    for (size_t g = 0; g < ngroups; ++g) {
                        outptr[g] += num_zeros[l] * 0.5 * num_zeros[g];
                    }
                }
            }
        }
    }

    while (pos != input.size() && input[pos].first < threshold) {
        inner_loop(pos, comp);
    }

    // Adding the contribution of zeros (in terms of the limit _being_ at zero + threshold)
    while (pos != input.size() && input[pos].first == threshold) {
        auto outptr = outputs[input[pos].second];
        for (size_t g = 0; g < ngroups; ++g) {
            outptr[g] += less_than[g] + 0.5 * num_zeros[g];
        }
        ++pos;
    }

    for (size_t l = 0; l < ngroups; ++l) {
        less_than[l] += num_zeros[l];
    }

    while (pos != input.size()) {
        inner_loop(pos, comp);
    }

    // Dividing by the product of sizes.
    if (normalize) {
        for (size_t l = 0; l < ngroups; ++l) {
            for (size_t g = 0; g < ngroups; ++g) {
                int prod = totals[l] * totals[g];
                if (prod) {
                    outputs[l][g] /= prod;
                } else {
                    outputs[l][g] = std::numeric_limits<double>::quiet_NaN();
                }
            }
        }
    }
}

}

}

#endif
