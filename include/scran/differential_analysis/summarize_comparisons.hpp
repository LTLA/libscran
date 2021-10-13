#ifndef SCRAN_SUMMARIZE_COMPARISONS_HPP
#define SCRAN_SUMMARIZE_COMPARISONS_HPP

#include <algorithm>
#include <numeric>
#include <vector>
#include <cmath>

namespace scran {

namespace differential_analysis {

enum summary {
    MIN,
    MEAN,
    MEDIAN,
    MAX,
    MIN_RANK,
    n_summaries 
};

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
                for (size_t i = 0; i < MIN_RANK; ++i) {
                    if (output[i].size()) {
                        output[i][l][gene] = std::numeric_limits<double>::quiet_NaN();
                    }
                }
            } else {
                int ncomps = ngroups - restart;
                if (ncomps > 1) {
                    if (output[MIN].size()) {
                        output[MIN][l][gene] = *std::min_element(start + restart, start + ngroups);
                    }
                    if (output[MEAN].size()) {
                        output[MEAN][l][gene] = std::accumulate(start + restart, start + ngroups, 0.0) / ncomps; // Mean
                    }
                    if (output[MEDIAN].size()) {
                        output[MEDIAN][l][gene] = median(start + restart, ncomps); // Median 
                    }
                    if (output[MAX].size()) {
                        output[MAX][l][gene] = *std::max_element(start + restart, start + ngroups); // Maximum
                    }
                } else {
                    for (size_t i = 0; i < MIN_RANK; ++i) {
                        if (output[i].size()) {
                            output[i][l][gene] = start[restart]; 
                        }
                    }
                }
            }
        }
    }
    return;
}

template<typename Stat>
void compute_min_rank(size_t ngenes, int ngroups, const Stat* effects, std::vector<Stat*>& output) {
    auto shift = ngroups * ngroups;

    #pragma omp parallel
    {
        std::vector<std::pair<Stat, int> > buffer(ngenes);

        #pragma omp for
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

}

#endif
