#ifndef SCRAN_AGGREGATE_ACROSS_CELLS_HPP
#define SCRAN_AGGREGATE_ACROSS_CELLS_HPP

#include <algorithm>
#include <vector>
#include "tatami/tatami.hpp"
#include "Factory.hpp"

namespace scran {

class AggregateAcrossCells {
public:
    template<typename T>
    struct Combinations {
        Combinations(size_t n) : factors(n) {}
        std::vector<std::vector<T> > factors;
        std::vector<size_t> counts;
    };

    template<typename Factor, typename Combined>
    static Combinations<Factor> combine_factors(size_t n, std::vector<const Factor*> factors, Combined* combined) {
        std::vector<size_t> indices(n);
        std::iota(indices.begin(), indices.end(), 0);

        std::sort(indices.begin(), indices.end(), [&](size_t left, size_t right) -> bool {
            for (auto curf : factors) {
                if (curf[left] < curf[right]) {
                    return true;
                } else if (curf[left] > curf[right]) {
                    return false;
                }
            }
            return false;
        });

        Combinations<Factor> output(factors.size()); 
        size_t last = 0;
        Combined counter = 0;
        if (n) {
            last = indices[0];
            combined[last] = counter;
            output.counts.push_back(1);
            for (size_t f = 0; f < factors.size(); ++f) {
                output.factors[f].push_back(factors[f][last]);
            }
        }

        for (size_t i = 1; i < n; ++i) {
            auto current = indices[i];
            bool diff = false;
            for (auto curf : factors) {
                if (curf[last] < curf[current]) {
                    diff = true;
                    break;
                }
            }

            if (diff) {
                for (size_t f = 0; f < factors.size(); ++f) {
                    output.factors[f].push_back(factors[f][current]);
                }
                output.counts.push_back(1);
                ++counter;
            } else {
                ++(output.counts.back());
            }

            combined[current] = counter;
            last = current;
        }

        return output;
    }

    template<typename Combined = int, typename Factor>
    static std::pair<Combinations<Factor>, std::vector<Combined> > combine_factors(size_t n, std::vector<const Factor*> factors) {
        std::vector<Combined> combined(n);
        auto output = combine_factors(n, std::move(factors), combined.data());
        return std::make_pair(std::move(output), std::move(combined));
    }

public:
    template<typename Data, typename Index, typename Factor, typename Sum, typename Detected>
    void run(const tatami::Matrix<Data, Index>* input, const Factor* groups, std::vector<Sum*> sums, std::vector<Detected*> detected) {
        aggregate_across_cells::BidimensionalFactory fac(input->nrow(), input->ncol(), groups, std::move(sums), std::move(detected));
        tatami::apply<0>(input, fac);
        return;
    } 

public:
    struct Defaults {
        static constexpr bool compute_sums = true;

        static constexpr bool compute_detected = true;
    };

    AggregateAcrossCells& set_compute_sums(bool c = Defaults::compute_sums) {
        compute_sums = c;
        return *this;
    }

    AggregateAcrossCells& set_compute_detected(bool c = Defaults::compute_detected) {
        compute_detected = c;
        return *this;
    }

private:
    bool compute_sums = Defaults::compute_sums;
    bool compute_detected = Defaults::compute_detected;

public:
    template <typename Sum, typename Detected>
    struct Results {
        std::vector<std::vector<Sum> > sums;
        std::vector<std::vector<Detected> > detected;
    };

    template<typename Sum = double, typename Detected = int, typename Data, typename Index, typename Factor>
    Results<Sum, Detected> run(const tatami::Matrix<Data, Index>* input, const Factor* groups) {
        size_t NC = input->ncol();
        size_t nlevels = (NC ? *std::max_element(groups, groups + NC) + 1 : 0);
        size_t ngenes = input->nrow();

        Results<Sum, Detected> output;
        std::vector<Sum*> sumptr;
        std::vector<Detected*> detptr;

        if (compute_sums) {
            output.sums.resize(nlevels, std::vector<Sum>(ngenes));
            sumptr.resize(nlevels);
            for (size_t l = 0; l < nlevels; ++l) {
                sumptr[l] = output.sums[l].data();
            }
        }

        if (compute_detected) {
            output.detected.resize(nlevels, std::vector<Detected>(ngenes));
            detptr.resize(nlevels);
            for (size_t l = 0; l < nlevels; ++l) {
                detptr[l] = output.detected[l].data();
            }
        }

        run(input, groups, std::move(sumptr), std::move(detptr));
        return output;
    } 
};

}

#endif
