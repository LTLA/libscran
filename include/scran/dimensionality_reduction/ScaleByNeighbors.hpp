#ifndef SCRAN_NEIGHBOR_SCALING_HPP
#define SCRAN_NEIGHBOR_SCALING_HPP

#include <vector>
#include <cmath>
#include <memory>
#include "knncolle/knncolle.hpp"
#include "tatami/stats/medians.hpp"

/**
 * @file ScaleByNeighbors.hpp
 *
 * @brief Scale multi-modal embeddings based on their relative variance.
 */

namespace scran {

/**
 * @brief Scale multi-modal embeddings to adjust for differences in variance.
 *
 * The premise is that we have multiple embeddings for the same set of cells, usually generated from different data modalities (e.g., RNA, protein, and so on).
 * We would like to combine these embeddings into a single embedding for downstream analyses such as clustering and t-SNE/UMAP.
 * The easiest combining strategy is to just concatenate the matrices together into a single embedding that contains information from all modalities.
 * However, this is complicated by the differences in the variance between modalities, whereby higher noise in one modality might drown out biological signal in another embedding.
 *
 * This class implements a scaling approach to equalize noise across embeddings prior to concatenation.
 * We compute the median distance to the $k$-th nearest neighbor in each embedding;
 * this is used as a proxy for the noise within a subpopulation containing at least $k$ cells.
 * We then compute a scaling factor as the ratio of the medians for a "target" embedding compared to a "reference" embedding.
 * The idea is to scale the target embedding by this factor before concatenation of matrices.
 *
 * This approach aims to remove differences in the magnitude of noise while preserving genuine differences in biological signal.
 * Embedding-specific subpopulations are preserved in the concatenation, provided that the difference between subpopulations is greater than that within subpopulations.
 * By contrast, a naive scaling based on the total variance would penalize embeddings with strong biological heterogeneity.
 */
class ScaleByNeighbors {
public:
    /**
     * @brief Default parameter settings.
     */
    struct Defaults { 
        /**
         * See `set_neighbors()` for more details.
         */
        static constexpr int neighbors = 20;

        /**
         * See `set_approximate()` for more details.
         */
        static constexpr bool approximate = false;
    };

    /**
     * @param n Number of neighbors used in the nearest neighbor search.
     * This can be interpreted as the minimum size of each subpopulation.
     *
     * @return A reference to this `ScaleByNeighbors` instance.
     */
    ScaleByNeighbors& set_neighbors(int n = Defaults::neighbors) {
        num_neighbors = n;
        return *this;
    }

    /**
     * @param a Whether to perform an approximate neighbor search.
     *
     * @return A reference to this `ScaleByNeighbors` instance.
     *
     * This parameter only has an effect when an array is passed to `run()`,
     * otherwise the search uses the algorithm specified by the **knncolle** class.
     */
    ScaleByNeighbors& set_approximate(int a = Defaults::approximate) {
        approximate = a;
        return *this;
    }

private:
    int num_neighbors = Defaults::neighbors;
    bool approximate = Defaults::approximate;

public:
    /**
     * @param ndim Number of dimensions in the embedding.
     * @param ncells Number of cells in the embedding.
     * @param[in] data Pointer to an array containing the embedding matrix.
     * This should be stored in column-major layout where each row is a dimension and each column is a cell.
     *
     * @return Pair containing the median distance to the $k$-th nearest neighbor (first)
     * and the root-mean-squared distance across all cells (second).
     * These values can be used in `compute_scale()`.
     */
    std::pair<double, double> compute_distance(int ndim, size_t ncells, const double* data) const {
        std::unique_ptr<knncolle::Base<> > ptr;
        if (!approximate) {
            ptr.reset(new knncolle::VpTreeEuclidean<>(ndim, ncells, data));
        } else {
            ptr.reset(new knncolle::AnnoyEuclidean<>(ndim, ncells, data));
        }
        return compute_distance(ptr.get());
    }

    /**
     * @tparam Search Search index class, typically a `knncolle::Base` subclass.
     * @param search Search index for the embedding.
     *
     * @return Pair containing the median distance to the $k$-th nearest neighbor (first)
     * and the root-mean-squared distance across all cells (second).
     * These values can be used in `compute_scale()`.
     */
    template<class Search>
    std::pair<double, double> compute_distance(const Search* search) const {
        size_t nobs = search->nobs();
        std::vector<double> dist(nobs);

#ifndef SCRAN_CUSTOM_PARALLEL
        #pragma omp parallel for
        for (size_t i = 0; i < nobs; ++i) {
#else
        SCRAN_CUSTOM_PARALLEL(nobs, [&](size_t start, size_t end) -> void {
        for (size_t i = start; i < end; ++i) {
#endif
            auto neighbors = search->find_nearest_neighbors(i, num_neighbors);
            dist[i] = neighbors.back().second;
        }
#ifdef SCRAN_CUSTOM_PARALLEL
        });
#endif

        double med = tatami::stats::compute_median<double>(dist.data(), nobs);
        double rmsd = 0;
        for (auto d : dist) {
            rmsd += d * d;
        }
        rmsd = std::sqrt(rmsd);
        return std::make_pair(med, rmsd);
    }

public:
    /**
     * @param ref Output of `compute_scale()` for the reference embedding.
     * The first value contains the median distance while the second value contains the RMSD.
     * @param target Output of `compute_scale()` for the target embedding.
     *
     * @return A scaling factor to apply to the target embedding.
     * Scaling all values in the target matrix by the factor will equalize the magnitude of the noise to that of the reference embedding.
     *
     * If either of the median distances is zero, this function automatically switches to the root-mean-square-distance to the $k$-th neighbor.
     * The scaling factor is then defined as the ratio of the RMSDs between embeddings.
     * If the reference RMSDs is zero, this function will return zero;
     * if the target RMSD is zero, this function will return positive infinity.
     */
    double compute_scale(const std::pair<double, double>& ref, const std::pair<double, double>& target) const {
        if (target.first == 0 || ref.first == 0) {
            if (target.second == 0) {
                return std::numeric_limits<double>::infinity();
            } else if (ref.second == 0) {
                return 0;
            } else {
                return ref.second / target.second; 
            }
        } else {
            return ref.first / target.first;
        }
    }
};

}

#endif
