#ifndef SCRAN_COHEN_D_HPP
#define SCRAN_COHEN_D_HPP

#include "../feature_selection/block_summaries.h"

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
        Simple(int nge, int ngr, const double* m, const double* v) : ngenes(nge), ngroups(ngr), means(m), vars(v) {}
        int ngenes, ngroups;
        const double* means;
        const double* vars;

        template<class Buf>
        void operator()(int gene, Buf& buffer) const {
            for (int g1 = 0; g1 < ngroups; ++g1) {
                left_mean = means[g1 * ngenes + gene]; 
                left_var = vars[g1 * ngenes + gene];

                for (int g2 = 0; g2 < g1; ++g2) {
                    right_mean = means[g2 * ngenes + gene]; 
                    right_var = vars[g2 * ngenes + gene];

                    double d = cohens_d(left_mean, left_var, right_mean, right_var);
                    buffer[g1 * ngroups + g2] = d;
                    buffer[g2 * ngroups + g1] = -d;
                }
                buffer[g1 * ngroups + g1] = 0;
            }
        }
    };

public:
    struct Blocked {
        Blocked(int nge, int ngr, int nb, const double* m, const double* v, const double* w) : ngenes(nge), ngroups(ngr), nblocks(nb), means(m), vars(v), weights(w), weightsum(nb * ngr) {}

        int ngenes, ngroups, nblocks;
        const double* means;
        const double* vars;
        const double* weights;
        std::vector<double> weightsum;

        template<class Buf>
        void operator()(int gene, Buf& buffer) const {
            std::fill(buffer, buffer + ngroups * ngroups, 0);
            std::fill(weightsum.begin(), weightsum.end(), 0);

            for (int b = 0; b < nblocks; ++b) {
                for (int g1 = 0; g1 < ngroups; ++g1) {
                    left_mean = means[b * ngenes * ngroups + g1 * ngenes + gene]; 
                    left_var = vars[b * ngenes + ngroups + g1 * ngenes + gene];

                    for (int g2 = 0; g2 < g1; ++g2) {
                        right_mean = means[b * nblocks * ngenes + g2 * ngenes + gene]; 
                        right_var = vars[b * nblocks * ngenes + g2 * ngenes + gene];

                        double d = cohens_d(left_mean, left_var, right_mean, right_var);
                        if (!std::isnan(d)) {
                            double weight = weights[g1 * ngroups + g2];
                            buffer[g1 * ngroups + g2] += d * weight;
                            weightsum[g1 * ngroups + g2] += weight;
                        }
                    }
                }
            }

            for (int g1 = 0; g1 < ngroups; ++g1) {
                for (int g2 = 0; g2 < g1; ++g2) {
                    buffer[g1 * ngroups + g2] /= weightsum[g1 * ngroups + g2];
                    buffer[g2 * ngroups + g1] = -buffer[g1 * ngroups + g2];
                }
            }
        }
    };

public:
    template<class MAT, typename G, typename OUT>
    void run(const MAT* p, const G* group, std::vector<std::vector<OUT*> > output) {
        maxg = *std::max_element(group, group + p->ncol());
        std::vector<double> means(p->nrow() * maxg);
        std::vector<double> vars(p->nrow() * maxg);

        std::vector<double*> meanptrs, varptrs;
        for (G i = 0; i < maxg; ++i) {
            meanptrs.push_back(means.data() + i * p->nrow());
            varptrs.push_back(vars.data() + i * p->nrow());
        }

        block_summaries(p, group, std::move(meanptrs), std::move(varptrs));

        Simple src(p->nrow(), maxg, means, vars);
        differential_analyses::summarize_comparisons(p->nrow(), maxg, src, std::move(output));
    }

    template<class MAT, typename G, typename B, typename OUT>
    void run_blocked(const MAT* p, const G* group, const B* block, std::vector<std::vector<OUT*> > output) {
        if (block == NULL) {
            run(p, group, block, std::move(output));
            return;
        }

        maxg = *std::max_element(group, group + p->ncol());
        maxb = *std::max_element(block, block + p->ncol());
        std::vector<double> means(p->nrow() * maxg * maxb);
        std::vector<double> vars(p->nrow() * maxg * maxb);

        std::vector<double*> meanptrs, varptrs;
        for (G i = 0; i < maxg * maxb; ++i) {
            meanptrs.push_back(means.data() + i * p->nrow());
            varptrs.push_back(vars.data() + i * p->nrow());
        }

        std::vector<int> combos(p->ncol());
        for (size_t i = 0; i < combos; ++i) {
            combos[i] = block[i] * maxg + group[i];
        }

        block_summaries(p, combos.data(), std::move(meanptrs), std::move(varptrs));
        
        BlockedSimple src(p->nrow(), maxg, maxb, means, vars);
        differential_analyses::summarize_comparisons(p->nrow(), maxg, src, std::move(output));
    }
};


}

#endif
