#ifndef SCRAN_PAIRWISE_EFFECTS_HPP
#define SCRAN_PAIRWISE_EFFECTS_HPP

#include "../utils/macros.hpp"

#include "Factory.hpp"
#include "cohens_d.hpp"
#include "lfc.hpp"
#include "delta_detected.hpp"
#include "../utils/vector_to_pointers.hpp"

#include "tatami/stats/apply.hpp"

/**
 * @file PairwiseEffects.hpp
 *
 * @brief Compute pairwise effect sizes between groups of cells.
 */

namespace scran {

/**
 * @cond
 */
namespace differential_analysis {

class EffectsCalculator {
public:
    EffectsCalculator(int nt, double t) : num_threads(nt), threshold(t) {}

private:
    int num_threads;
    double threshold;

public:
    template<class Matrix, typename G, class Overlord>
    void run(const Matrix* p, const G* group, int ngroups, Overlord& overlord) const {
        std::vector<int> group_size(ngroups);
        for (size_t i = 0, end = p->ncol(); i < end; ++i) {
            ++(group_size[group[i]]);
        }
        core(p, group, group_size, group, ngroups, static_cast<const int*>(NULL), 1, overlord);
    }

    template<class Matrix, typename G, typename B, class Overlord>
    void run_blocked(const Matrix* p, const G* group, int ngroups, const B* block, int nblocks, Overlord& overlord) const {
        if (block == NULL) {
            run(p, group, block, overlord);
            return;
        }

        size_t ncombos = ngroups * nblocks;
        std::vector<int> combos(p->ncol());
        std::vector<int> combo_size(ncombos);
        for (size_t i = 0; i < combos.size(); ++i) {
            combos[i] = group[i] * nblocks + block[i];
            ++(combo_size[combos[i]]);
        }

        core(p, combos.data(), combo_size, group, ngroups, block, nblocks, overlord);
    }

private:
    template<class Matrix, typename L, class Ls, typename G, typename B, class Overlord>
    void core(const Matrix* p, const L* level, const Ls& level_size, const G* group, size_t ngroups, const B* block, size_t nblocks, Overlord& overlord) const {
        size_t ngenes = p->nrow();
        size_t nlevels = level_size.size();
        size_t holding = nlevels * ngenes;
        std::vector<double> tmp_means(holding), tmp_variances(holding), tmp_detected(holding);

        if (!overlord.needs_auc()) {
            differential_analysis::SimpleBidimensionalFactory fact(
                p->nrow(), 
                p->ncol(), 
                level, 
                &level_size, 
                tmp_means.data(),
                tmp_variances.data(),
                tmp_detected.data()
            );
            tatami::apply<0>(p, fact, num_threads);

        } else {
            // Need to remake this, as there's no guarantee that 'blocks' exists.
            std::vector<B> tmp_blocks;
            if (!block) {
                tmp_blocks.resize(p->ncol());
                block = tmp_blocks.data();
            }

            differential_analysis::ComplexPerRowFactory fact(
                p->nrow(), 
                p->ncol(), 
                level, 
                &level_size, 
                tmp_means.data(),
                tmp_variances.data(),
                tmp_detected.data(),
                group,
                ngroups,
                block,
                nblocks,
                threshold,
                &overlord
            );
            tatami::apply<0>(p, fact, num_threads);
        }

        // Dividing through to get the actual detected proportions.
        // Don't bother parallelizing this, given how simple it is.
        auto dptr = tmp_detected.data();
        for (size_t gene = 0; gene < ngenes; ++gene) {
            size_t in_offset = gene * nlevels;
            for (size_t l = 0; l < nlevels; ++l, ++dptr) {
                if (level_size[l]) {
                    *dptr /= level_size[l];
                } else {
                    *dptr = std::numeric_limits<double>::quiet_NaN();
                }
            }
        }

        // Allowing the overlord to process the statistics as it sees fit.
        overlord.process(p->nrow(), level_size, ngroups, nblocks, tmp_means.data() + in_offset, tmp_variances.data(), tmp_detected.data(), num_threads);
    }
};

}

#endif
