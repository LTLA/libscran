#ifndef SCRAN_PAIRWISE_EFFECTS_HPP
#define SCRAN_PAIRWISE_EFFECTS_HPP

#include "../utils/macros.hpp"

#include "Factory.hpp"

#include "tatami/stats/apply.hpp"

namespace scran {

/**
 * @cond
 */
template<typename Stat>
std::vector<std::vector<Stat*> > vector_to_pointers2(std::vector<std::vector<std::vector<Stat> > >& input) {
    std::vector<std::vector<Stat*> > ptrs;
    for (auto& current : input) {
        ptrs.push_back(vector_to_pointers(current));
    }
    return ptrs;
}

template<typename Stat>
std::vector<Stat*> vector_to_pointers3(std::vector<std::vector<std::vector<Stat> > >& input) {
    std::vector<Stat*> ptrs;
    for (auto& current : input) {
        ptrs.push_back(current.front().data()); // first vector from each element.
    }
    return ptrs;
}
/**
 * @endcond
 */

class PairwiseEffects {
public:
    /**
     * @brief Default parameter settings.
     */
    struct Defaults {
        /**
         * See `set_threshold()` for details.
         */
        static constexpr double threshold = 0;

        /**
         * See `set_num_threads()`.
         */
        static constexpr int num_threads = 1;
    };

private:
    double threshold = Defaults::threshold;

    int num_threads = Defaults::num_threads;

public:
    /**
     * @param t Threshold on the log-fold change.
     * This should be non-negative.
     *
     * @return A reference to this `PairwiseEffects` object.
     */
    PairwiseEffects& set_threshold(double t = Defaults::threshold) {
        threshold = t;
        return *this;
    }

    /**
     * @param n Number of threads to use. 
     * @return A reference to this `PairwiseEffects` object.
     */
    PairwiseEffects& set_num_threads(int n = Defaults::num_threads) {
        num_threads = n;
        return *this;
    }

public:
    template<class Matrix, typename G, typename Stat>
    void run(
        const Matrix* p, 
        const G* group, 
        std::vector<Stat*> means, 
        std::vector<Stat*> detected, 
        Stat* cohen,
        Stat* auc,
        Stat* lfc,
        Stat* delta_detected) 
    const {
        size_t ngroups = means.size();
        std::vector<int> group_size(ngroups);
        for (size_t i = 0; i < p->ncol(); ++i) {
            ++(group_size[group[i]]);
        }
        core(p, group, group_size, group, ngroups, static_cast<const int*>(NULL), 1, means, detected, cohen, auc, lfc, delta_detected);
    }

    template<class Matrix, typename G, typename B, typename Stat>
    void run_blocked(
        const Matrix* p, 
        const G* group, 
        const B* block, 
        std::vector<Stat*> means, 
        std::vector<Stat*> detected, 
        Stat* cohen,
        Stat* auc,
        Stat* lfc,
        Stat* delta_detected) 
    const {
        size_t ngroups = means.size();
        if (block == NULL) {
            run(p, group, fetch_first(means), fetch_first(detected), cohen, auc, lfc, delta_detected);
            return;
        }

        size_t nblocks = (ngroups ? means[0].size() : 0); // if means.size() == 0, then there are no groups, so there are no blocks, either.
        size_t ncombos = ngroups * nblocks;
        std::vector<int> combos(p->ncol());
        std::vector<int> combo_size(ncombos);

        for (size_t i = 0; i < combos.size(); ++i) {
            combos[i] = group[i] * nblocks + block[i];
            ++(combo_size[combos[i]]);
        }

        std::vector<Stat*> means2(ncombos), detected2(ncombos);
        auto mIt = means2.begin(), dIt = detected2.begin();
        for (int g = 0; g < ngroups; ++g) {
            for (int b = 0; b < nblocks; ++b, ++mIt, ++dIt) {
                *mIt = means[g][b];
                *dIt = detected[g][b];
            }
        }

        core(p, combos.data(), combo_size, group, ngroups, block, nblocks, means2, detected2, cohen, auc, lfc, delta_detected);
    }

private:
    template<class MAT, typename L, class Ls, typename G, typename B, typename Stat>
    void core(const MAT* p, 
        const L* level, 
        const Ls& level_size, 
        const G* group, 
        size_t ngroups, 
        const B* block, 
        size_t nblocks, 
        std::vector<Stat*> means, 
        std::vector<Stat*> detected, 
        Stat* cohen, 
        Stat* auc,
        Stat* lfc,
        Stat* delta_detected) 
    const {
        size_t buffer_size = p->nrow() * ngroups * ngroups;

        if (auc == NULL) {
            differential_analysis::BidimensionalFactory fact(
                p->nrow(), 
                p->ncol(), 
                std::move(means), 
                std::move(detected), 
                cohen,
                lfc,
                delta_detected,
                level, 
                &level_size, 
                ngroups, 
                nblocks, 
                threshold
            );
            tatami::apply<0>(p, fact, num_threads);

        } else {
            // Need to remake this, as there's no guarantee that 'blocks' exists.
            std::vector<B> tmp_blocks;
            if (!block) {
                tmp_blocks.resize(p->ncol());
                block = tmp_blocks.data();
            }

            differential_analysis::PerRowFactory fact(
                p->nrow(), 
                p->ncol(), 
                std::move(means), 
                std::move(detected), 
                cohen,
                auc,
                lfc,
                delta_detected,
                level, 
                &level_size, 
                group, 
                ngroups, 
                block, 
                nblocks, 
                threshold
            );
            tatami::apply<0>(p, fact, num_threads);
        }
    }

public:
    template<typename Stat>
    struct Results {
        Results(size_t ngenes, size_t ngroups, bool do_cohen, bool do_auc, bool do_lfc, bool do_delta) {
            size_t nelements = ngenes * ngroups * ngroups;
            if (do_cohen) {
                cohen.resize(nelements);
            }
            if (do_auc) {
                auc.resize(nelements);
            }
            if (do_lfc) {
                lfc.resize(nelements);
            }
            if (do_delta) {
                delta_detected.resize(nelements);
            }
        }

        std::vector<Stat> cohen;
        std::vector<Stat> auc;
        std::vector<Stat> lfc;
        std::vector<Stat> delta_detected;
    };

    template<typename Stat = double, class MAT, typename G>
    Results<Stat> run(const Matrix* p, const G* group, std::vector<Stat*> means, std::vector<Stat*> detected) const {
        auto ngroups = means.size();
        Results<Stat> res(p->nrow(), ngroups, do_cohen, do_auc, do_lfc, do_delta_detected); 
        run(
            p, 
            group, 
            std::move(means), 
            std::move(detected), 
            harvest_pointer(res.cohen, do_cohen),
            harvest_pointer(res.auc, do_auc),
            harvest_pointer(res.lfc, do_lfc),
            harvest_pointer(res.delta_detected, do_delta_detected)
        );
        return; 
    }

    template<typename Stat = double, class MAT, typename G, typename B> 
    Results<Stat> run_blocked(const MAT* p, const G* group, const B* block, std::vector<std::vector<Stat*> > means, std::vector<std::vector<Stat*> > detected) const {
        if (block == NULL) {
            return run(p, group);
        }
        auto ngroups = means.size();
        Results<Stat> res(p->nrow(), ngroups, do_cohen, do_auc, do_lfc, do_delta_detected); 
        run_blocked(
            p,
            group,
            block,
            std::move(means),
            std::move(detected),
            harvest_pointer(res.cohen, do_cohen),
            harvest_pointer(res.auc, do_auc),
            harvest_pointer(res.lfc, do_lfc),
            harvest_pointer(res.delta_detected, do_delta_detected)
        );
        return res;
    }

public:
    template<typename Stat>
    struct ResultsWithMeans : public Results {
        ResultsWithMeans(size_t ngenes, size_t ngroups, size_t nblocks, bool do_cohen, bool do_auc, bool do_lfc, bool do_delta) :
            Results(ngenes, ngroups, do_cohen, do_auc, do_lfc, do_delta), means(ngroups), detected(ngroups) 
        {
            for (size_t g = 0; g < ngroups; ++g) {
                means[g].reserve(nblocks);
                detected[g].reserve(nblocks);
                for (size_t b = 0; b < nblocks; ++b) {
                    means[g].emplace_back(ngenes);
                    detected[g].emplace_back(ngenes);
                }
            }
        }

        std::vector<std::vector<std::vector<Stat> > > means;
        std::vector<std::vector<std::vector<Stat> > > detected;
    };

    template<typename Stat = double, class MAT, typename G>
    ResultsWithMeans<Stat> run(const Matrix* p, const G* group) {
        auto ngroups = *std::max_element(group, group + p->ncol()) + 1;
        ResultsWithMeans<Stat> res(p->nrow(), ngroups, 1, do_cohen, do_auc, do_lfc, do_delta_detected); 
        run(
            p, 
            group, 
            vector_to_pointers3(res.means), 
            vector_to_pointers3(res.detected), 
            harvest_pointer(res.cohen, do_cohen),
            harvest_pointer(res.auc, do_auc),
            harvest_pointer(res.lfc, do_lfc),
            harvest_pointer(res.delta_detected, do_delta_detected)
        );
        return; 
    }

    template<typename Stat = double, class MAT, typename G, typename B> 
    PairwiseResults<Stat> run_blocked(const MAT* p, const G* group, const B* block) {
        if (block == NULL) {
            return run(p, group);
        }
        auto ngroups = *std::max_element(group, group + p->ncol()) + 1;
        auto nblocks = *std::max_element(block, block + p->ncol()) + 1;
        ResultsWithMeans<Stat> res(p->nrow(), ngroups, nblocks, do_cohen, do_auc, do_lfc, do_delta_detected); 
        run_blocked(
            p,
            group,
            block,
            vector_to_pointers2(res.means),
            vector_to_pointers2(res.detected)
            harvest_pointer(res.cohen, do_cohen),
            harvest_pointer(res.auc, do_auc),
            harvest_pointer(res.lfc, do_lfc),
            harvest_pointer(res.delta_detected, do_delta_detected)
        );
        return res;
    }

private:
    template<typename Ptr>
    static std::vector<Ptr> fetch_first(const std::vector<std::vector<Ptr> >& input) {
        std::vector<Ptr> output;
        output.reserve(input.size());
        for (const auto& i : input) {
            output.push_back(i.front());
        }
        return output;
    }

    template<typename Stat> 
    static Stat* harvest_pointers(std::vector<Stat>& source, bool use) {
        if (use) {
            return source.data();
        } else {
            return static_cast<Stat*>(NULL);
        }
    }
};

}

#endif
