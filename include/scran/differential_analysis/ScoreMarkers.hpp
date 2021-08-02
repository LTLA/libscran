#ifndef SCRAN_SCORE_MARKERS_HPP
#define SCRAN_SCORE_MARKERS_HPP

#include "BidimensionalFactory.hpp"
#include "tatami/stats/apply.hpp"

namespace scran {

class ScoreMarkers {
public:
    template<class MAT, typename G, typename Stat>
    void run(const MAT* p, const G* group, std::vector<Stat*> output, Stat* means, Stat* detected) {
        auto ngroups = output.size();
        std::vector<int> group_size(ngroups);
        for (size_t i = 0; i < p->ncol(); ++i) {
            ++(group_size[group[i]]);
        }

        std::vector<Stat*> stats { means, detected };
        differential_analysis::BidimensionalFactory fact(p->nrow(), p->ncol(), output, stats, group, group_size, ngroups, 1);
        tatami::apply<0>(p, fact);
        return;
    }

public:
    struct Results {
        Results(size_t ngenes, int ngroups, int neffects) : 
            effects(neffects, std::vector<std::vector<double> >(ngroups, std::vector<double>(ngenes * 4))), 
            means(ngenes * ngroups), detected(ngenes * ngroups) {}

        std::vector<std::vector<std::vector<double> > > effects;
        std::vector<double> means;
        std::vector<double> detected;
    };

    template<class MAT, typename G> 
    Results run(const MAT* p, const G* group) {
        auto ngroups = *std::max_element(group, group + p->ncol()) + 1;
        Results res(p->nrow(), ngroups, 1); 

        std::vector<double*> cohen_ptrs;
        for (size_t o = 0; o < ngroups; ++o) {
            cohen_ptrs.push_back(res.effects[0][o].data());
        }
        
        run(p, group, std::move(cohen_ptrs), res.means.data(), res.detected.data());
        return res;
    }

//    template<class MAT, typename G, typename B, typename OUT>
//    void run_blocked(const MAT* p, const G* group, const B* block, std::vector<std::vector<OUT*> > output) {
//        if (block == NULL) {
//            run(p, group, block, std::move(output));
//            return;
//        }
//
//        auto ngroups = *std::max_element(group, group + p->ncol()) + 1;
//        auto nblocks = *std::max_element(block, block + p->ncol()) + 1;
//        std::vector<double> means(p->nrow() * ngroups * nblocks);
//        std::vector<double> vars(p->nrow() * ngroups * nblocks);
//
//        std::vector<double*> meanptrs, varptrs;
//        for (G i = 0; i < ngroups * nblocks; ++i) {
//            meanptrs.push_back(means.data() + i * p->nrow());
//            varptrs.push_back(vars.data() + i * p->nrow());
//        }
//
//        std::vector<int> combos(p->ncol());
//        for (size_t i = 0; i < combos.size(); ++i) {
//            combos[i] = block[i] * ngroups + group[i];
//        }
//
//        feature_selection::block_summaries<true>(p, combos.data(), std::move(meanptrs), std::move(varptrs));
//        
//        Blocked src(p->nrow(), ngroups, nblocks, means.data(), vars.data());
//        differential_analysis::summarize_comparisons(p->nrow(), ngroups, src, std::move(output));
//    }
};

}

#endif
