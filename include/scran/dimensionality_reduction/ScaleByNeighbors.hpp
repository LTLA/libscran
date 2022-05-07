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

    std::unique_ptr<knncolle::Base<> > build_index(int ndim, size_t nobs, const double* data) const {
        std::unique_ptr<knncolle::Base<> > ptr;
        if (!approximate) {
            ptr.reset(new knncolle::VpTreeEuclidean<>(ndim, nobs, data));
        } else {
            ptr.reset(new knncolle::AnnoyEuclidean<>(ndim, nobs, data));
        }
        return ptr;
    }

    template<class Search>
    std::pair<double, double> median_distance_to_neighbors(const Search* search) const {
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
     * @param ncells Number of cells in both embeddings.
     * @param nref Number of dimensions in the reference embedding.
     * @param[in] ref Pointer to an array containing the reference embedding matrix.
     * This should be stored in column-major layout where each row is a dimension and each column is a cell.
     * @param ntarget Number of dimensions in the target embedding.
     * @param[in] target Pointer to an array containing the target embedding matrix.
     * This should be stored in column-major layout where each row is a dimension and each column is a cell.
     *
     * @return The scaling factor to be applied to the target embedding, see the other `run()` method for details.
     */
    double run(size_t ncells, int nref, const double* ref, int ntarget, const double* target) const {
        auto search1 = build_index(nref, ncells, ref);
        auto search2 = build_index(ntarget, ncells, target);
        return run(search1.get(), search2.get());
    }

    /**
     * @tparam Search Search index class, typically a `knncolle::Base` subclass.
     * @param ref Search index for the reference embedding.
     * @param target Search index for the target embedding.
     *
     * @return The scaling factor to be applied to the target embedding.
     *
     * If either of the median distances is zero, this function automatically switches to the root-mean-square-distance to the $k$-th neighbor.
     * The scaling factor is then defined as the ratio of the RMSDs between embeddings.
     * If the reference RMSDs is zero, this function will return zero;
     * if the target RMSD is zero, this function will return positive infinity.
     */
    template<class Search>
    double run(const Search* ref, const Search* target) const {
        if (ref->nobs() != target->nobs()) {
            throw std::runtime_error("number of observations must be the same for both search indices");
        }

        auto ref_scale = median_distance_to_neighbors(ref);
        auto target_scale = median_distance_to_neighbors(target);
        if (target_scale.first == 0 || ref_scale.first == 0) {
            if (target_scale.second == 0) {
                return std::numeric_limits<double>::infinity();
            } else if (ref_scale.second == 0) {
                return 0;
            } else {
                return ref_scale.second / target_scale.second; 
            }
        } else {
            return ref_scale.first / target_scale.first;
        }
    }
};

}

#endif
