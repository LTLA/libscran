#ifndef SCRAN_COHEN_D_HPP
#define SCRAN_COHEN_D_HPP

#include "../feature_selection/block_summaries.h"

#include <vector>
#include <limits>
#include <algorithm>

namespace scran {

class CohensD {
private:
    std::vector<double> means, vars;

    template<class V>
    struct Functor {
        Functor(int nge, int ngr, const V& m, const V& v) : ngenes(nge), ngroups(ngr), means(m), vars(v) {}
        int ngenes, ngroups;
        const V& means;
        const V& vars;

        template<class Buf>
        void operator()(int gene, Buf& buffer) const {
            for (int g1 = 0; g1 < ngroups; ++g1) {
                left_mean = means[g1 * ngroups + gene]; 
                left_var = vars[g1 * ngroups + gene];

                for (int g2 = 0; g2 < g1; ++g2) {
                    right_mean = means[g2 * ngroups + gene]; 
                    right_var = vars[g2 * ngroups + gene];

                    if (std::isnan(left_var) || std::isnan(right_var)) {
                        buffer[g1 * ngroups + g2] = std::numeric_limits<double>::quiet_NaN();
                        buffer[g2 * ngroups + g1] = std::numeric_limits<double>::quiet_NaN();
                    } else {
                        double base = (left_mean - right_mean) / std::sqrt((left_var + right_var)/2);
                        buffer[g1 * ngroups + g2] = base;
                        buffer[g2 * ngroups + g1] = -base;
                    }
                }
                buffer[g1 * ngroups + g1] = 0;
            }
        }
    };

public:
    template<class MAT, typename G, typename OUT>
    void run(const MAT* p, const G* group, std::vector<std::vector<OUT*> > output) {
        maxg = *std::max_element(group, group + p->ncol());
        means.resize(p->nrow() * maxg);
        vars.resize(p->nrow() * maxg);

        std::vector<double*> meanptrs, varptrs;
        for (G i = 0; i < maxg; ++i) {
            meanptrs.push_back(means.data() + i * p->nrow());
            varptrs.push_back(vars.data() + i * p->nrow());
        }

        block_summaries(p, group, std::move(meanptrs), std::move(varptrs));

        // And now doing pairwise comparisons between groups.
        Functor src(p->nrow(), maxg, means, vars);
        summarize_comparisons(p->nrow(), maxg, src, std::move(output));
    }
};


}

#endif
