#ifndef SCRAN_SUMMARIZE_COMPARISONS_HPP
#define SCRAN_SUMMARIZE_COMPARISONS_HPP

#include <algorithm>
#include <numeric>
#include <vector>
#include <cmath>

namespace scran {

namespace differential_analysis {

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

template<class Source, typename Stat>
void summarize_comparisons(int ngroups, Source* buffer, size_t gene, size_t ngenes, std::vector<Stat*>& output) {
    for (int l = 0; l < ngroups; ++l) {
        auto start = buffer + l * ngroups;

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
            for (int i = 0; i < 4; ++i) {
                output[l][gene + ngenes * i] = std::numeric_limits<double>::quiet_NaN();                                
            }
        } else {
            output[l][gene] = *std::min_element(start + restart, start + ngroups);

            int ncomps = ngroups - restart;
            if (ncomps > 1) {
                output[l][gene + ngenes] = std::accumulate(start + restart, start + ngroups, 0.0) / ncomps; // Mean
                output[l][gene + ngenes * 2] = median(start + restart, ncomps); // Median 
            } else {
                output[l][gene + ngenes] = start[restart];
                output[l][gene + ngenes * 2] = start[restart];
            }

            output[l][gene + ngenes * 3] = *std::max_element(start + restart, start + ngroups);
        }
    }
}

}

}

#endif
