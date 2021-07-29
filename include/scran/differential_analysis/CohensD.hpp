#ifndef SCRAN_COHEN_D_HPP
#define SCRAN_COHEN_D_HPP

#include "../feature_selection/block_summaries.hpp"
#include "summarize_comparisons.hpp"

#include <vector>
#include <limits>
#include <algorithm>

namespace scran {

class CohensD {
private:
    static double compute_cohens_d(double left_mean, double left_var, double right_mean, double right_var) {
        if (std::isnan(left_var) && std::isnan(right_var)) {
            return std::numeric_limits<double>::quiet_NaN();
        } else {
            double base = (left_mean - right_mean);
            if (std::isnan(left_var)) {
                base /= std::sqrt(right_var);
            } else if (std::isnan(right_var)) {
                base /= std::sqrt(left_var);
            } else {
                base /= std::sqrt((left_var + right_var)/2);
            }
            return base;
        }
    }

    struct Simple {
        Simple(int nge, int ngr, const double* m, const double* v) : ngenes(nge), ngroups(ngr), means(m), vars(v), local_means(ngr), local_vars(ngr) {}

        int ngenes, ngroups;
        const double* means;
        const double* vars;

        std::vector<double> local_means, local_vars;

        template<class Buf>
        void operator()(int gene, Buf& buffer) {
            // Copy to improve cache-friendliness in pairwise look-ups.
            for (int g = 0; g < ngroups; ++g) {
                local_means[g] = means[g * ngenes + gene];
            }
            for (int g = 0; g < ngroups; ++g) {
                local_vars[g] = vars[g * ngenes + gene];
            }

            // Performing the pairwise comparisons.
            for (int g1 = 0; g1 < ngroups; ++g1) {
                const auto& left_mean = means[g1]; 
                const auto& left_var = vars[g1];

                for (int g2 = 0; g2 < g1; ++g2) {
                    const auto& right_mean = means[g2]; 
                    const auto& right_var = vars[g2];

                    double d = compute_cohens_d(left_mean, left_var, right_mean, right_var);
                    buffer[g1 * ngroups + g2] = d;
                    buffer[g2 * ngroups + g1] = -d;
                }
                buffer[g1 * ngroups + g1] = 0;
            }
        }
    };

public:
    struct Blocked {
        Blocked(int nge, int ngr, int nb, const double* m, const double* v, const double* w) : ngenes(nge), ngroups(ngr), nblocks(nb), means(m), vars(v), weights(w), 
            weightsum(nb * ngr), local_means(nb * ngr), local_vars(nb * ngr) {}

        int ngenes, ngroups, nblocks;
        const double* means;
        const double* vars;
        const double* weights;

        std::vector<double> weightsum;
        std::vector<double> local_means, local_vars;

        template<class Buf>
        void operator()(int gene, Buf& buffer) {
            // Copy to improve cache friendliness in pairwise look-ups.
            for (int b = 0; b < nblocks; ++b) {
                int offset = b * ngenes * ngroups + gene;
                for (int g = 0; g < ngroups; ++g) {
                    local_means[b * ngroups + g] = means[offset + g * ngenes];
                }
            }

            for (int b = 0; b < nblocks; ++b) {
                int offset = b * ngenes * ngroups + gene;
                for (int g = 0; g < ngroups; ++g) {
                    local_vars[b * ngroups + g] = vars[offset + g * ngenes];
                }
            }

            // Performing pairwise comparisons within each batch.
            std::fill(buffer, buffer + ngroups * ngroups, 0);
            std::fill(weightsum.begin(), weightsum.end(), 0);

            for (int b = 0; b < nblocks; ++b) {
                int offset = b * ngroups;

                for (int g1 = 0; g1 < ngroups; ++g1) {
                    const auto& left_mean = means[offset + g1]; 
                    const auto& left_var = vars[offset + g1];

                    for (int g2 = 0; g2 < g1; ++g2) {
                        const auto& right_mean = means[offset + g2]; 
                        const auto& right_var = vars[offset + g2];

                        double weight = weights[(offset + g1) * ngroups + g2];
                        if (weight) {
                            double d = compute_cohens_d(left_mean, left_var, right_mean, right_var);
                            buffer[g1 * ngroups + g2] += d * weight;
                            weightsum[g1 * ngroups + g2] += weight;
                        }
                    }
                }
            }

            for (int g1 = 0; g1 < ngroups; ++g1) {
                for (int g2 = 0; g2 < g1; ++g2) {
                    double total_weight = weightsum[g1 * ngroups + g2];
                    if (total_weight) {
                        buffer[g1 * ngroups + g2] /= total_weight;
                        buffer[g2 * ngroups + g1] = -buffer[g1 * ngroups + g2];
                    } else {
                        buffer[g1 * ngroups + g2] = std::numeric_limits<double>::quiet_NaN();
                        buffer[g2 * ngroups + g1] = std::numeric_limits<double>::quiet_NaN();
                    }
                }
            }
        }
    };

public:
    template<class MAT, typename G, typename OUT>
    void run(const MAT* p, const G* group, std::vector<std::vector<OUT*> > output) {
        auto ngroups = *std::max_element(group, group + p->ncol()) + 1;
        std::vector<double> means(p->nrow() * ngroups);
        std::vector<double> vars(p->nrow() * ngroups);

        std::vector<double*> meanptrs, varptrs;
        for (G i = 0; i < ngroups; ++i) {
            meanptrs.push_back(means.data() + i * p->nrow());
            varptrs.push_back(vars.data() + i * p->nrow());
        }

        block_summaries<true>(p, group, std::move(meanptrs), std::move(varptrs));

        Simple src(p->nrow(), ngroups, means.data(), vars.data());
        differential_analysis::summarize_comparisons(p->nrow(), ngroups, src, std::move(output));
    }

    template<class MAT, typename G, typename B, typename OUT>
    void run_blocked(const MAT* p, const G* group, const B* block, std::vector<std::vector<OUT*> > output) {
        if (block == NULL) {
            run(p, group, block, std::move(output));
            return;
        }

        auto ngroups = *std::max_element(group, group + p->ncol()) + 1;
        auto nblocks = *std::max_element(block, block + p->ncol()) + 1;
        std::vector<double> means(p->nrow() * ngroups * nblocks);
        std::vector<double> vars(p->nrow() * ngroups * nblocks);

        std::vector<double*> meanptrs, varptrs;
        for (G i = 0; i < ngroups * nblocks; ++i) {
            meanptrs.push_back(means.data() + i * p->nrow());
            varptrs.push_back(vars.data() + i * p->nrow());
        }

        std::vector<int> combos(p->ncol());
        for (size_t i = 0; i < combos.size(); ++i) {
            combos[i] = block[i] * ngroups + group[i];
        }

        block_summaries<true>(p, combos.data(), std::move(meanptrs), std::move(varptrs));
        
        Blocked src(p->nrow(), ngroups, nblocks, means.data(), vars.data());
        differential_analysis::summarize_comparisons(p->nrow(), ngroups, src, std::move(output));
    }
};


}

#endif
