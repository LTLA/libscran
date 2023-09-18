#ifndef SCRAN_QUICK_GROUPED_SIZE_FACTORS_HPP
#define SCRAN_QUICK_GROUPED_SIZE_FACTORS_HPP

#include "../utils/macros.hpp"

#include "tatami/tatami.hpp"
#include "../dimensionality_reduction/SimplePca.hpp"
#include "LogNormCounts.hpp"
#include "GroupedSizeFactors.hpp"
#include "kmeans/Kmeans.hpp"
#include "kmeans/InitializePCAPartition.hpp"

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
     * Number of k-means clusters to obtain.
     */
    int clusters = 25;

    /**
     * Pointer to an array of length equal to the number of cells.
     * Each entry should contain the block assignment for each cell,
     * as an integer in \f$[0, N)\f$ where \f$N\f$ is the total number of blocks.
     * If `NULL`, all cells are assumed to belong to a single block.
     * This parameter is only used in the initial normalization step.
     */
    const Block_* block = NULL;

    /**
     * Pointer to an array of length equal to the number of cells.
     * Each entry should contain an initial size factor for each cell.
     * If `NULL`, the initial size factors are defined by `LogNormCounts`.
     */
    const SizeFactor_* initial_factors = NULL;

    /**
     * @cond
     */
    typedef SizeFactor_ SizeFactor;
    /**
     * @endcond
     */
};

/**
 * Quickly compute grouped size factors by deriving a sensible grouping from an expression matrix.
 * The idea is to break up the dataset into broad clusters so that `GroupedSizeFactors` can remove the composition biases between them;
 * it is primarily intended for ADT count data where large composition biases are introduced by the presence of a few highly abundant markers in each subpopulation.
 * This function will create an initial log-normalized matrix,
 * derive a low-dimensional representation via principal components analysis,
 * generate k-means clusters from the top principal components,
 * and finally use those clusters in `GroupedSizeFactors`.
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
    // First, we normalize.
    auto ptr = tatami::wrap_shared_ptr(mat);
    LogNormCounts logger;

    if (opt.initial_factors) {
        std::vector<SizeFactor_> fac(opt.initial_factors, opt.initial_factors + ptr->ncol());
        if (opt.block) {
            ptr = logger.run_blocked(ptr, std::move(fac), opt.block);
        } else {
            ptr = logger.run(ptr, std::move(fac));
        }
    } else {
        if (opt.block) {
            ptr = logger.run_blocked(ptr, opt.block);
        } else {
            ptr = logger.run(ptr);
        }
    }

    // Now, we cut the dimensions.
    SimplePca pca_runner;
    pca_runner.set_rank(opt.rank);
    auto pc_out = pca_runner.run(ptr.get());
    const auto& pcs = pc_out.pcs;

    // And then we do some clustering.
    kmeans::Kmeans kmeans_runner;
    kmeans::InitializePCAPartition<double, int, int> init;
    auto clust_out = kmeans_runner.run(
        pcs.rows(), 
        pcs.cols(), 
        pcs.data(), 
        opt.clusters,
        &init
    );
        
    // Finally we use the clustering to run the groupings.
    GroupedSizeFactors group_runner;
    group_runner.run(mat, clust_out.clusters.data(), output);
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
