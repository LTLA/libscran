#ifndef SCRAN_CHOOSE_HVGS_HPP
#define SCRAN_CHOOSE_HVGS_HPP

#include "../utils/macros.hpp"

#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdint>

/**
 * @file ChooseHvgs.hpp
 *
 * @brief Choose highly variable genes for downstream analyses.
 */

namespace scran {

/**
 * @brief Choose highly variable genes for downstream analyses.
 *
 * This is done by selecting the `top` number of genes with the largest values for a precomputed variance statistic.
 * We typically use the residual from the trend as computed by `ModelGeneVar`.
 */
class ChooseHvgs {
public:
    /**
     * @brief Default paramater settings.
     */
    struct Defaults {
        /**
         * See `set_top()` for more details.
         */
        static constexpr size_t top = 4000;
    };

private:
    size_t top = Defaults::top;

public:
    /**
     * @param t The number of top genes to consider as highly variable genes.
     *
     * @return A reference to this `ChooseHvgs` object.
     *
     * The choice is more or less arbitrary and is a trade-off between capturing more biological signal at the cost of increasing noise.
     * Values of 1000-5000 seem appropriate for most use cases.
     * Note that increasing the number of genes will also increase the size of the subdataset to be used in downstream computation.
     */
    ChooseHvgs& set_top(size_t t = Defaults::top) {
        top = t;
        return *this;
    }

public:
    /**
     * Choose HVGs to use in downstream analyses. 
     *
     * @tparam V Type of the variance statistic.
     * @tparam T Type to be used as a boolean.
     *
     * @param n Number of genes.
     * @param[in] statistic Pointer to an array of length `n` containing the per-gene variance statistics.
     * @param[out] output Pointer to an array of length `n`, used to store a boolean flag.
     * On completion, `output` is filled with `true` if the gene is to be retained and `false` otherwise.
     */
    template<typename V, typename T>
    void run(size_t n, const V* statistic, T* output) const {
        std::vector<size_t> collected(n);
        std::iota(collected.begin(), collected.end(), 0);
        std::sort(collected.begin(), collected.end(), [&](size_t l, size_t r) -> bool { return statistic[l] > statistic[r]; });
        
        auto limit = std::min(n, top);
        std::fill(output, output + n, false);
        for (size_t i = 0; i < limit; ++i) {
            output[collected[i]] = true; 
        }
    }

    /**
     * @tparam V Type of the variance statistic.
     * @tparam T Type to be used as a boolean.
     *
     * @param n Number of genes.
     * @param[in] statistic Pointer to an array of length `n` containing the per-gene variance statistics.
     *
     * @return A vector of booleans of length `n`, indicating whether each gene is to be retained.
     */
    template<typename T = uint8_t, typename V>
    std::vector<T> run(size_t n, const V* statistic) const {
        std::vector<T> output(n);
        run(n, statistic, output.data());
        return output;
    }
};

}
#endif
