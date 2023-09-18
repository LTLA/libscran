#ifndef SCRAN_QUICK_GROUPED_SIZE_FACTORS_HPP
#define SCRAN_QUICK_GROUPED_SIZE_FACTORS_HPP

#include "../utils/macros.hpp"

#include <algorithm>
#include <vector>
#include <cmath>
#include <functional>
#include <memory>

#include "tatami/tatami.hpp"
#include "kmeans/Kmeans.hpp"
#include "kmeans/InitializePCAPartition.hpp"

#include "../utils/blocking.hpp"
#include "../dimensionality_reduction/SimplePca.hpp"
#include "LogNormCounts.hpp"
#include "GroupedSizeFactors.hpp"

/**
 * @brief Quickly compute grouped size factors.
 * @file quick_grouped_size_factors.hpp
 */

namespace scran {

/**
 * @brief Quickly compute grouped size factors.
 * @namespace scran::quick_grouped_size_factors
 */
namespace quick_grouped_size_factors {

/**
 * @brief Options for `run()`.
 * @tparam Block_ Integer type for the blocking factor.
 * @tparam SizeFactor_ Floating-point type for initial size factors.
 */
template<
    typename Block_ = int,
    typename SizeFactor_ = double
>
struct Options {
    /**
     * Number of principal components of the log-expression matrix to retain for clustering.
     */
    int rank = 25;

    /**
     * Number of k-means clusters to obtain, as a function of the number of cells in each block.
     * This defaults to the square root with an upper bound of 50.
     */
    std::function<size_t(size_t)> clusters;

    /**
     * Pointer to an array of length equal to the number of cells.
     * Each entry should contain the block assignment for each cell,
     * as an integer in \f$[0, N)\f$ where \f$N\f$ is the total number of blocks.
     * If `NULL`, all cells are assumed to belong to a single block.
     */
    const Block_* block = NULL;

    /**
     * Pointer to an array of length equal to the number of cells.
     * Each entry should contain an initial size factor for each cell.
     * If `NULL`, the initial size factors are defined by `LogNormCounts`.
     */
    const SizeFactor_* initial_factors = NULL;

    /**
     * Number of threads to use in all internal calculations.
     */
    int num_threads = 1;
};

/**
 * @cond
 */
namespace internal {

template<
    typename Value_, 
    typename Index_
>
auto cluster(const tatami::Matrix<Value_, Index_>* mat, int rank, size_t clusters, int num_threads) {
    SimplePca pca_runner;
    pca_runner.set_rank(rank);
    pca_runner.set_num_threads(num_threads);
    auto pc_out = pca_runner.run(mat);
    const auto& pcs = pc_out.pcs;

    kmeans::Kmeans kmeans_runner;
    kmeans_runner.set_num_threads(num_threads);
    kmeans::InitializePCAPartition<Value_, Index_, Index_> init;
    return kmeans_runner.run(
        pcs.rows(), 
        pcs.cols(), 
        pcs.data(), 
        clusters,
        &init
    );
}

}
/**
 * @endcond
 */

/**
 * Quickly compute grouped size factors by deriving a sensible grouping from an expression matrix.
 * The idea is to break up the dataset into broad clusters so that `GroupedSizeFactors` can remove the composition biases between them.
 * It is primarily intended for ADT count data where large composition biases are introduced by the presence of a few highly abundant markers in each subpopulation.
 *
 * More specifically, this function will create an initial log-normalized matrix via `LogNormCounts`,
 * derive a low-dimensional representation via principal components analysis with `SimplePca`,
 * generate k-means clusters from the top principal components with the **kmeans** library,
 * and finally use those clusters in `GroupedSizeFactors`.
 *
 * If multiple blocks are present, dimensionality reduction and clustering is performed separately for each block.
 * This avoids wasting the cluster partitions on irrelevant differences between blocks, e.g., due to batch effects.
 * However, the final calculation of grouped size factors will be done using the entire dataset at once, so the factors will remove block-to-block scaling differences.
 *
 * @tparam Value_ Numeric type for the matrix values.
 * @tparam Index_ Integer type of the matrix row/column indices.
 * @tparam OutputFactor_ Floating-point ype of the output factors.
 * @tparam Block_ See `Options`.
 * @tparam SizeFactor_ See `Options`.
 *
 * @param mat Input count matrix, where rows are features and columns are cells.
 * @param[out] output Pointer to an array of length equal to the number of cells.
 * On output, this stores the output factors for each cell.
 * @param opt Further options.
 */
template<
    typename Value_, 
    typename Index_, 
    typename OutputFactor_, 
    typename Block_,
    typename SizeFactor_
>
void run(const tatami::Matrix<Value_, Index_>* mat, OutputFactor_* output, const Options<Block_, SizeFactor_>& opt) {
    std::vector<Index_> clusters;
    Index_ NC = mat->ncol();
    auto ptr = tatami::wrap_shared_ptr(mat);

    LogNormCounts logger;
    logger.set_num_threads(opt.num_threads);

    auto fun = opt.clusters;
    if (!fun) {
        fun = [](size_t n) -> size_t {
            size_t candidate = std::sqrt(static_cast<double>(n));
            return std::min(candidate, static_cast<size_t>(50));
        };
    }

    if (opt.block) {
        auto nblocks = count_ids(NC, opt.block);
        std::vector<std::vector<Index_> > assignments(nblocks);
        for (Index_ c = 0; c < NC; ++c) {
            assignments[opt.block[c]].push_back(c);
        }

        clusters.resize(NC);
        Index_ last_cluster = 0;

        for (size_t b = 0; b < nblocks; ++b) {
            const auto& inblock = assignments[b];
            auto subptr = tatami::make_DelayedSubset<1>(ptr, tatami::ArrayView<Index_>(inblock.data(), inblock.size()));

            std::shared_ptr<tatami::Matrix<Value_, Index_> > normalized;
            if (opt.initial_factors) {
                std::vector<SizeFactor_> fac;
                fac.reserve(inblock.size());
                for (auto i : inblock) {
                    fac.push_back(opt.initial_factors[i]);
                }
                normalized = logger.run(std::move(subptr), std::move(fac));
            } else {
                normalized = logger.run(std::move(subptr));
            }

            auto res = internal::cluster(normalized.get(), opt.rank, fun(inblock.size()), opt.num_threads);
            auto cIt = res.clusters.begin();
            for (auto i : inblock) {
                clusters[i] = *cIt + last_cluster;
                ++cIt;
            }
            last_cluster += *std::max_element(res.clusters.begin(), res.clusters.end()) + 1;
        }

    } else {
        std::shared_ptr<const tatami::Matrix<Value_, Index_> > normalized; // TODO: avoid propagating const'ness from LogNormCounts.
        if (opt.initial_factors) {
            std::vector<SizeFactor_> fac(opt.initial_factors, opt.initial_factors + NC);
            normalized = logger.run(std::move(ptr), std::move(fac));
        } else {
            normalized = logger.run(std::move(ptr));
        }

        auto res = internal::cluster(normalized.get(), opt.rank, fun(NC), opt.num_threads);
        clusters = std::move(res.clusters);
    }

    GroupedSizeFactors group_runner;
    group_runner.set_num_threads(opt.num_threads);
    group_runner.run(mat, clusters.data(), output);
    return;
}

/**
 * Overload of `run()` with default options.
 *
 * @tparam Value_ Numeric type for the matrix values.
 * @tparam Index_ Integer type of the matrix row/column indices.
 * @tparam OutputFactor_ Floating-point ype of the output factors.
 *
 * @param mat Input count matrix, where rows are features and columns are cells.
 * @param[out] output Pointer to an array of length equal to the number of cells.
 * On output, this stores the output factors for each cell.
 */
template<
    typename Value_, 
    typename Index_, 
    typename OutputFactor_
>
void run(const tatami::Matrix<Value_, Index_>* mat, OutputFactor_* output) {
    run(mat, output, Options<>());
}

/**
 * Overload of `run()` that handles the memory allocation of the output factors.
 *
 * @tparam OutputFactor_ Floating-point ype of the output factors.
 * @tparam Value_ Numeric type for the matrix values.
 * @tparam Index_ Integer type of the matrix row/column indices.
 * @tparam Block_ See `Options`.
 * @tparam SizeFactor_ See `Options`.
 *
 * @param mat Input count matrix, where rows are features and columns are cells.
 * @param opt Further options.
 *
 * @return Vector of length equal to the number of cells, containing the output factors for each cell.
 */
template<
    typename OutputFactor_ = double, 
    typename Value_, 
    typename Index_, 
    typename Block_,
    typename SizeFactor_
>
std::vector<OutputFactor_> run(const tatami::Matrix<Value_, Index_>* mat, const Options<Block_, SizeFactor_>& opt) {
    std::vector<OutputFactor_> output(mat->ncol());
    run(mat, output.data(), opt);
    return output;
}

/**
 * Overload of `run()` with default options.
 *
 * @tparam OutputFactor_ Floating-point ype of the output factors.
 * @tparam Value_ Numeric type for the matrix values.
 * @tparam Index_ Integer type of the matrix row/column indices.
 *
 * @param mat Input count matrix, where rows are features and columns are cells.
 *
 * @return Vector of length equal to the number of cells, containing the output factors for each cell.
 */
template<
    typename OutputFactor_ = double, 
    typename Value_, 
    typename Index_
>
std::vector<OutputFactor_> run(const tatami::Matrix<Value_, Index_>* mat) {
    std::vector<OutputFactor_> output(mat->ncol());
    run(mat, output.data(), Options<>());
    return output;
}

}

}

#endif
