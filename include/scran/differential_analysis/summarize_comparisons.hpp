#ifndef SCRAN_SUMMARIZE_COMPARISONS_HPP
#define SCRAN_SUMMARIZE_COMPARISONS_HPP

#include <algorithm>
#include <numeric>
#include <vector>
#include <cmath>

namespace scran {

namespace differential_analysis {

static constexpr int n_summaries = 5; // min, mean, median, max, min-rank.

template<class IT>
double median (IT start, size_t n) {
    int halfway = n / 2;
    std::nth_element(start, start + halfway, start + n);

    if (n % 2 == 1) {
        return start[halfway];
    }

    double med = start[halfway];
    std::nth_element(start, start + halfway - 1, start + halfway);
    return (med + start[halfway - 1])/ 2;
}

template<typename Stat>
void summarize_comparisons(size_t ngenes, int ngroups, Stat* effects, std::vector<std::vector<Stat*> >& output) {
    #pragma omp parallel for 
    for (size_t gene = 0; gene < ngenes; ++gene) {
        auto base = effects + gene * ngroups * ngroups;
        for (int l = 0; l < ngroups; ++l) {
            auto start = base + l * ngroups;

            // Ignoring the self comparison.
            int restart = 1;
            std::swap(start[0], start[l]); 

            // Pruning out NAs.
            for (int r = restart; r < ngroups; ++r) {
                if (std::isnan(start[r])) {
                    if (r != restart) {
                        std::swap(start[restart], start[r]); 
                    }
                    ++restart;
                }
            }

            if (restart == ngroups) {
                for (size_t i = 0; i < 4; ++i) {
                    if (output[i][l]) {
                        output[i][l][gene] = std::numeric_limits<double>::quiet_NaN();
                    }
                }
            } else {
                int ncomps = ngroups - restart;
                if (ncomps > 1) {
                    if (output[0][l]) {
                        output[0][l][gene] = *std::min_element(start + restart, start + ngroups);
                    }
                    if (output[1][l]) {
                        output[1][l][gene] = std::accumulate(start + restart, start + ngroups, 0.0) / ncomps; // Mean
                    }
                    if (output[2][l]) {
                        output[2][l][gene] = median(start + restart, ncomps); // Median 
                    }
                    if (output[3][l]) {
                        output[3][l][gene] = *std::max_element(start + restart, start + ngroups); // Maximum
                    }
                } else {
                    for (size_t i = 0; i < 4; ++i) {
                        if (output[i][l]) {
                            output[i][l][gene] = start[restart]; 
                        }
                    }
                }
            }
        }
    }
    return;
}

template<typename Stat, class V>
void compute_min_rank(size_t ngenes, int ngroups, const Stat* effects, std::vector<Stat*>& output, V& buffer) {
    auto shift = ngroups * ngroups;

    #pragma omp parallel for
    for (int g = 0; g < ngroups; ++g) {
        auto target = output[g];
        std::fill(target, target + ngenes, ngenes + 1); 

        for (int g2 = 0; g2 < ngroups; ++g2) {
            if (g == g2) {
                continue;
            }
            auto copy = effects + g * ngroups + g2;
            for (size_t i = 0; i < ngenes; ++i, copy += shift) {
                buffer[i].first = -*copy; // negative to sort by decreasing value.
                buffer[i].second = i;
            }
            std::sort(buffer.begin(), buffer.end());

            double counter = 1;
            for (const auto& x : buffer) {
                target[x.second] = std::min(counter, target[x.second]);
                ++counter;
            }
        }
    }
}

}

}

#endif
