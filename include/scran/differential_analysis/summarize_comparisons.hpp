#ifndef SCRAN_SUMMARIZE_COMPARISONS_HPP
#define SCRAN_SUMMARIZE_COMPARISONS_HPP

#include <cmath>
#include <algorithm>
#include <vector>

namespace scran {

namespace differential_analysis {

template<class IT>
double quantile(IT start, int size, int k, int q) {
    int idx = std::floor(static_cast<double>((size - 1) * k) / q);  
    
    // Check if observed quantile == expected quantile of k/q.
    if (idx * q == k * (size - 1)) { 
        return start[idx];
    }

    // Compute interpolation between the two points.
    return ((k * (size - 1) - idx * q) * start[idx] + ((idx + 1) * q - k * (size - 1)) * start[idx + 1])/q;
}

template<class Source, typename OUT>
void summarize_comparisons(size_t ngenes, int ngroups, Source src, std::vector<std::vector<OUT*> > output) {
    // You'll need to make a copy of 'src' in each thread via openmP's firstprivate.
    std::vector<double> buffer(ngroups * ngroups);
    for (size_t g = 0; g < ngenes; ++g) {
        src(g, buffer);

        for (int l = 0; l < ngroups; ++l) {
            auto start = buffer.data() + l * ngroups;

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

            std::sort(start + restart, start + ngroups);

            output[l][0][g] = start[restart]; // minimum.

            int ncomps = ngroups - restart;
            if (ncomps > 1) {
                output[l][1][g] = quantile(start + restart, ncomps, 1, 4); // First quartile.
                output[l][2][g] = quantile(start + restart, ncomps, 2, 4); // Median
                output[l][3][g] = quantile(start + restart, ncomps, 3, 4); // Third quartile.
            } else {
                output[l][1][g] = start[restart];
                output[l][2][g] = start[restart];
                output[l][3][g] = start[restart];
            }

            output[l][4][g] = start[ngroups-1]; // maximum.
                   
        }
    }
}

}

}

#endif
