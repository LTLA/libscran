#ifndef SCRAN_AGGREGATE_ACROSS_CELLS_HPP
#define SCRAN_AGGREGATE_ACROSS_CELLS_HPP

#include "../utils/macros.hpp"

#include <algorithm>
#include <vector>
#include "tatami/tatami.hpp"

#include "../utils/blocking.hpp"

/**
 * @file AggregateAcrossCells.hpp
 *
 * @brief Aggregate expression values across cells.
 */

namespace scran {

/**
 * @brief Aggregate expression values across cells.
 *
 * This class computes the sum of expression values for each grouping of cells,
 * typically for the creation of pseudo-bulk expression profiles for cluster/sample combinations.
 * Expression values are generally expected to be counts, though the same code can be trivially re-used to compute the average log-expression.
 * We can also report the number of cells with detected (i.e., positive) expression values in each grouping.
 */ 
class AggregateAcrossCells {
public:
    /**
     * @brief Unique combinations of factors.
     *
     * @tparam Factor Factor type, typically an integer.
     */
    template<typename Factor>
    struct Combinations {
        /**
         * @cond
         */
        Combinations(size_t n) : factors(n) {}
        /**
         * @endcond
         */

        /**
         * Unique combinations of factor levels.
         * Each inner vector corresponds to a factor.
         * All inner vectors have the same length.
         * Corresponding entries of the inner vectors define a particular combination of levels.
         * Combinations are guaranteed to be sorted.
         */
        std::vector<std::vector<Factor> > factors;

        /**
         * Number of cells in each unique combination of factor levels.
         * This has the same length as each inner vector of `factors`.
         * All entries are guaranteed to be positive.
         */
        std::vector<size_t> counts;
    };

    /**
     * @tparam Factor Factor type.
     * @tparam Combined Type of the combined factor.
     * May need to be different from `Factor` if the latter does not have enough unique levels.
     *
     * @param n Number of observations (i.e., cells).
     * @param[in] factors Pointers to arrays of length `n`, each containing a different factor.
     * @param[out] combined Pointer to an array of length `n`, in which the combined factor is to be stored.
     *
     * @return 
     * A `Combinations` object is returned containing the unique combinations of levels observed in `factors`.
     * A combined factor is saved to `combined`; values are indices of combinations in the output `Combinations` object.
     *
     * This function compresses multiple `factors` into a single `combined` factor, which can then be used as the `factor` in `run()`.
     * In this manner, `AggregateAcrossCells` can easily handle aggregation across any number of factors.
     */
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

    /**
     * @tparam Factor Factor type.
     * @tparam Combined Type of the combined factor.
     * May need to be different from `Factor` if the latter does not have enough unique levels.
     *
     * @param n Number of observations (i.e., cells).
     * @param[in] factors Pointers to arrays of length `n`, each containing a different factor.
     *
     * @return 
     * A pair containing:
     *
     * - A `Combinations` object, containing the unique combinations of levels observed in `factors`.
     * - A vector of length `n` containing the combined factor.
     *
     * See the other `combine_factors()` method for details.
     */
    template<typename Combined = int, typename Factor>
    static std::pair<Combinations<Factor>, std::vector<Combined> > combine_factors(size_t n, std::vector<const Factor*> factors) {
        std::vector<Combined> combined(n);
        auto output = combine_factors(n, std::move(factors), combined.data());
        return std::make_pair(std::move(output), std::move(combined));
    }

private:
    template<bool sparse_, typename Index_, typename Contents_, typename Factor_, typename Sum_, typename Detected_>
    void compute_row(Index_ i, Index_ nc, const Contents_& row, const Factor_* factor, std::vector<Sum_>& tmp_sums, std::vector<Sum_*>& sums, std::vector<Detected_>& tmp_detected, std::vector<Detected_*>& detected) {
        if (sums.size()) {
            std::fill(tmp_sums.begin(), tmp_sums.end(), 0);

            if constexpr(sparse_) {
                for (Index_ j = 0; j < row.number; ++j) {
                    tmp_sums[factor[row.index[j]]] += row.value[j];
                }
            } else {
                for (Index_ j = 0; j < nc; ++j) {
                    tmp_sums[factor[j]] += row[j];
                }
            }

            // Computing before transferring for more cache-friendliness.
            for (Index_ l = 0; l < tmp_sums.size(); ++l) {
                sums[l][i] = tmp_sums[l];
            }
        }

        if (detected.size()) {
            std::fill(tmp_detected.begin(), tmp_detected.end(), 0);

            if constexpr(sparse_) {
                for (Index_ j = 0; j < row.number; ++j) {
                    tmp_detected[factor[row.index[j]]] += (row.value[j] > 0);
                }
            } else {
                for (Index_ j = 0; j < nc; ++j) {
                    tmp_detected[factor[j]] += (row[j] > 0);
                }
            }

            for (Index_ l = 0; l < tmp_detected.size(); ++l) {
                detected[l][i] = tmp_detected[l];
            }
        }
    }

    template<bool row_, bool sparse_, typename Data_, typename Index_, typename Factor_, typename Sum_, typename Detected_>
    void compute(const tatami::Matrix<Data_, Index_>* p, const Factor_* factor, std::vector<Sum_*>& sums, std::vector<Detected_*>& detected) {
        tatami::Options opt;
        opt.sparse_ordered_index = false;

        if constexpr(row_) {
            tatami::parallelize([&](size_t, Index_ s, Index_ l) {
                auto ext = tatami::consecutive_extractor<row_, sparse_>(p, s, l, opt);
                std::vector<Sum_> tmp_sums(sums.size());
                std::vector<Detected_> tmp_detected(detected.size());

                auto NC = p->ncol();
                std::vector<Data_> vbuffer(NC);
                typename std::conditional<sparse_, std::vector<Index_>, Index_>::type ibuffer(NC);

                for (Index_ x = s, end = s + l; x < end; ++x) {
                    if constexpr(sparse_) {
                        compute_row<true>(x, NC, ext->fetch(x, vbuffer.data(), ibuffer.data()), factor, tmp_sums, sums, tmp_detected, detected);
                    } else {
                        compute_row<false>(x, NC, ext->fetch(x, vbuffer.data()), factor, tmp_sums, sums, tmp_detected, detected);
                    }
                }
            }, p->nrow(), num_threads);

        } else {
            tatami::parallelize([&](size_t, Index_ s, Index_ l) {
                auto NC = p->ncol();
                auto ext = tatami::consecutive_extractor<row_, sparse_>(p, 0, NC, s, l, opt);
                std::vector<Data_> vbuffer(l);
                typename std::conditional<sparse_, std::vector<Index_>, Index_>::type ibuffer(l);

                for (Index_ x = 0; x < NC; ++x) {
                    auto current = factor[x];

                    if constexpr(sparse_) {
                        auto col = ext->fetch(x, vbuffer.data(), ibuffer.data());
                        if (sums.size()) {
                            auto& cursum = sums[current];
                            for (Index_ i = 0; i < col.number; ++i) {
                                cursum[col.index[i]] += col.value[i];
                            }
                        }

                        if (detected.size()) {
                            auto& curdetected = detected[current];
                            for (Index_ i = 0; i < col.number; ++i) {
                                curdetected[col.index[i]] += (col.value[i] > 0);
                            }
                        }

                    } else {
                        auto col = ext->fetch(x, vbuffer.data());

                        if (sums.size()) {
                            auto cursum = sums[current] + s;
                            for (Index_ i = 0; i < l; ++i) {
                                cursum[i] += col[i];
                            }
                        }

                        if (detected.size()) {
                            auto curdetected = detected[current] + s;
                            for (Index_ i = 0; i < l; ++i) {
                                curdetected[i] += (col[i] > 0);
                            }
                        }
                    }
                }
            }, p->nrow(), num_threads);
        }
    }

public:
    /**
     * Compute the per-gene sum and number of cells with detected expression for each level of a grouping factor.
     *
     * @tparam Data Type of data in the input matrix, should be numeric.
     * @tparam Index Integer type of index in the input matrix.
     * @tparam Factor Integer type of the factor.
     * @tparam Sum Type of the sum, usually the same as `Data`.
     * @tparam Detected Type for the number of detected cells, usually integer.
     *
     * @param input The input matrix where rows are features and columns are cells.
     * @param[in] factor Pointer to an array of length equal to the number of columns of `input`,
     * containing the factor level for each cell.
     * All levels should be integers in $[0, N)$ where $N$ is the number of unique levels.
     * @param[out] sums Vector of length $N$ (see `factor`),
     * containing pointers to arrays of length equal to the number of columns of `input`.
     * These will be filled with the summed expression across all cells in the corresponding level for each gene.
     * Alternatively, if the vector is of length 0, no sums will be computed.
     * @param[out] detected Vector of length $N$ (see `factor`),
     * containing pointers to arrays of length equal to the number of columns of `input`.
     * These will be filled with the number of cells with detected expression in the corresponding level for each gene.
     * Alternatively, if the vector is of length 0, no numbers will be computed.
     */
    template<typename Data, typename Index, typename Factor, typename Sum, typename Detected>
    void run(const tatami::Matrix<Data, Index>* input, const Factor* factor, std::vector<Sum*> sums, std::vector<Detected*> detected) {
        if (input->prefer_rows()) {
            if (input->sparse()) {
                compute<true, true>(input, factor, sums, detected);
            } else {
                compute<true, false>(input, factor, sums, detected);
            }
        } else {
            if (input->sparse()) {
                compute<false, true>(input, factor, sums, detected);
            } else {
                compute<false, false>(input, factor, sums, detected);
            }
        }
    } 

public:
    /**
     * @brief Default parameters for aggregation.
     */
    struct Defaults {
        /**
         * See `set_compute_sums()`.
         */
        static constexpr bool compute_sums = true;

        /**
         * See `set_compute_detected()`.
         */
        static constexpr bool compute_detected = true;

        /**
         * See `set_num_threads()`.
         */
        static constexpr int num_threads = 1;
    };

    /**
     * @param c Whether to compute the sum within each factor level.
     * @return A reference to this `AggregateAcrossCells` object.
     *
     * This function only affects `run()` when `sums` and `detected` are not supplied as inputs.
     */
    AggregateAcrossCells& set_compute_sums(bool c = Defaults::compute_sums) {
        compute_sums = c;
        return *this;
    }

    /**
     * @param c Whether to compute the number of detected cells within each factor level.
     * @return A reference to this `AggregateAcrossCells` object.
     * 
     * This function only affects `run()` when `sums` and `detected` are not supplied as inputs.
     */
    AggregateAcrossCells& set_compute_detected(bool c = Defaults::compute_detected) {
        compute_detected = c;
        return *this;
    }

    /**
     * @param n Number of threads to use. 
     * @return A reference to this `AggregateAcrossCells` object.
     */
    AggregateAcrossCells& set_num_threads(int n = Defaults::num_threads) {
        num_threads = n;
        return *this;
    }

private:
    bool compute_sums = Defaults::compute_sums;
    bool compute_detected = Defaults::compute_detected;
    int num_threads = Defaults::num_threads;

public:
    /**
     * @brief Container for the aggregation results.
     * @tparam Sum Type of the sum, should be numeric.
     * @tparam Detected Type for the number of detected cells, usually integer.
     */
    template <typename Sum, typename Detected>
    struct Results {
        /**
         * Vector of length equal to the number of factor levels.
         * Each inner vector is of length equal to the number of genes.
         * Each entry contains the summed expression across all cells in the corresponding level for the corresponding gene.
         *
         * If `compute_sums()` was set to `false`, this vector is empty.
         */
        std::vector<std::vector<Sum> > sums;

        /**
         * Vector of length equal to the number of factor levels.
         * Each inner vector is of length equal to the number of genes.
         * Each entry contains the number of cells in the corresponding level with detected expression for the corresponding gene.
         *
         * If `computed_detected()` was set to `false`, this vector is empty.
         */
        std::vector<std::vector<Detected> > detected;
    };

    /**
     * @tparam Sum Type of the sum, should be numeric.
     * @tparam Detected Type for the number of detected cells, usually integer.
     * @tparam Data Type of data in the input matrix, should be numeric.
     * @tparam Index Integer type of index in the input matrix.
     * @tparam Factor Integer type of the factor.
     *
     * @param input The input matrix where rows are features and columns are cells.
     * @param[in] factor Pointer to an array of length equal to the number of columns of `input`,
     * containing the factor level for each cell.
     * All levels should be integers in $[0, N)$ where $N$ is the number of unique levels.
     *
     * @return A `Results` object is returned containing the summed expression and the number of detected cells within each factor level across all genes.
     * 
     * This function will respect any user-supplied setting of `set_compute_sums()` and `set_compute_detected()`.
     * If either/both are false, the corresponding statistic(s) will not be computed and the corresponding vector in `Results` will be empty.
     */
    template<typename Sum = double, typename Detected = int, typename Data, typename Index, typename Factor>
    Results<Sum, Detected> run(const tatami::Matrix<Data, Index>* input, const Factor* factor) {
        size_t NC = input->ncol();
        size_t nlevels = count_ids(NC, factor);
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

        run(input, factor, std::move(sumptr), std::move(detptr));
        return output;
    } 
};

}

#endif
