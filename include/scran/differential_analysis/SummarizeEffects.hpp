#ifndef SCRAN_SUMMARIZE_EFFECTS_HPP
#define SCRAN_SUMMARIZE_EFFECTS_HPP

#include "../utils/macros.hpp"

#include "../utils/vector_to_pointers.hpp"
#include "summarize_comparisons.hpp"

#include <vector>

namespace scran {

class SummarizeEffects {
public:
    /**
     * @brief Default parameter settings.
     */
    struct Defaults {
        /**
         * See `set_num_threads()`.
         */
        static constexpr int num_threads = 1;

        static constexpr bool compute_min = true;

        static constexpr bool compute_mean = true;

        static constexpr bool compute_median = true;

        static constexpr bool compute_max = true;

        static constexpr bool compute_min_rank = true;
    };

private:
    int num_threads = Defaults::num_threads;
    bool compute_min = Defaults::compute_min;
    bool compute_mean = Defaults::compute_mean;
    bool compute_median = Defaults::compute_median;
    bool compute_max = Defaults::compute_max;
    bool compute_min_rank = Defaults::compute_min_rank;

public:
    /**
     * @param n Number of threads to use. 
     * @return A reference to this `SummarizeEffects` object.
     */
    SummarizeEffects& set_num_threads(int n = Defaults::num_threads) {
        num_threads = n;
        return *this;
    }

    SummarizeEffects& set_compute_min(bool c = Defaults::compute_min) {
        compute_min = c;
        return *this;
    }

    SummarizeEffects& set_compute_mean(bool c = Defaults::compute_mean) {
        compute_mean = c;
        return *this;
    }

    SummarizeEffects& set_compute_median(bool c = Defaults::compute_median) {
        compute_median = c;
        return *this;
    }

    SummarizeEffects& set_compute_max(bool c = Defaults::compute_max) {
        compute_max = c;
        return *this;
    }

    SummarizeEffects& set_compute_min_rank(bool c = Defaults::compute_min_rank) {
        compute_min_rank = c;
        return *this;
    }

public:
    template<typename Stat>
    void run(size_t ngenes, size_t ngroups, const Stat* effects, std::vector<std::vector<Stat*> > summaries) const {
        if (summaries.empty()) {
            return;
        }

        auto& min_rank = summaries[differential_analysis::MIN_RANK];
        if (min_rank.size()) {
            differential_analysis::compute_min_rank(ngenes, ngroups, effects, min_rank, num_threads);
        }

        differential_analysis::summarize_comparisons(ngenes, ngroups, effects, summaries, num_threads); 
    }

    template<typename Stat>
    std::vector<std::vector<std::vector<Stat> > > run(size_t ngenes, size_t ngroups, const Stat* effects) const {
        std::vector<std::vector<std::vector<Stat> > > output(differential_analysis::n_summaries);

        auto inflate = [&](auto& o) -> void {
            o.resize(ngroups);
            for (auto& o2 : o) {
                o2.resize(ngenes);
            }
        };

        if (compute_min) {
            inflate(output[differential_analysis::MIN]);
        }
        if (compute_mean) {
            inflate(output[differential_analysis::MEAN]);
        }
        if (compute_median) {
            inflate(output[differential_analysis::MEDIAN]);
        }
        if (compute_max) {
            inflate(output[differential_analysis::MAX]);
        }
        if (compute_min_rank) {
            inflate(output[differential_analysis::MIN_RANK]);
        }

        run(ngenes, ngroups, effects, vector_to_pointers(output));
        return output;
    }
};

}

#endif
