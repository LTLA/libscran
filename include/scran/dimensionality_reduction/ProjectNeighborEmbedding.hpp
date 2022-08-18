#ifndef SCRAN_PROJECT_EMBEDDING_HPP
#define SCRAN_PROJECT_EMBEDDING_HPP

#include "../utils/macros.hpp"

#include <vector>
#include <memory>
#include <cmath>

#include "knncolle/knncolle.hpp"
#include "tatami/stats/medians.hpp"

/** 
 * @file ProjectNeighborEmbedding.hpp
 *
 * @brief Project cells into a low-dimensional embedding using neighbors.
 */

namespace scran {

/**
 * @brief Project cells into a low-dimensional embedding using neighbors.
 *
 * This class projects cells into a "destination" embedding based on their neighbors in another "source" embedding.
 * It is primarily intended for mapping cells in a test dataset onto a 2-dimensional visualization generated from a reference dataset, 
 * after detecting the nearest neighbors in the reference for each test cell in PC space.
 * For example, we could downsample a dataset with `DownsampleByNeighbors`, generate a 2D visualization for the subset from the PCs, 
 * and then use `ProjectNeighborEmbedding` to project all cells onto the visualization.
 *
 * The projected location in the destination embedding for each test cell is defined as a weighted average of the coordinates of its neighbors.
 * The weight for each neighbor is a function of its distance to the test cell in the source embedding.
 * We use a tricube weighting scheme so that distant neighbors in low-density regions are given less weight in the average.
 * The bandwidth for each test cell is defined as the median distance among `k` neighbors plus a multiple of the MAD;
 * this follows the same MAD-based outlier detection approach as `IsOutlier`.
 */
class ProjectNeighborEmbedding {
public:
    /**
     * @brief Default parameter settings.
     */
    struct Defaults {
        /**
         * See `set_num_neighbors()` for details.
         */
        static constexpr int num_neighbors = 10;

        /**
         * See `set_approximate()` for details.
         */
        static constexpr int approximate = false;


        /**
         * See `set_num_threads()` for details.
         */
        static constexpr int num_threads = 1;

        /**
         * See `set_nmads()` for details.
         */
        static constexpr double nmads = 3;

        /**
         * See `set_scale()` for details.
         */
        static constexpr double scale = 5;
    };

    /**
     * @param k Number of neighbors to use for projection.
     * Larger values improve stability but increase the risk of including irrelevant neighbors.
     *
     * @return A reference to this `ProjectNeighborEmbedding` object.
     * 
     * Note that this parameter only has an effect in `run()` methods that do not accept a list of precomputed neighbors,
     * in which case the number of neighbors is taken from the list. 
     */
    ProjectNeighborEmbedding& set_num_neighbors(int k = Defaults::num_neighbors) {
        num_neighbors = k;
        return *this;
    }

    /**
     * @param a Whether an approximate neighbor search should be performed.
     *
     * @return A reference to this `ProjectNeighborEmbedding` object.
     */
    ProjectNeighborEmbedding& set_approximate(int a = Defaults::approximate) {
        approximate = a;
        return *this;
    }

    /**
     * @param n Number of threads to use.
     *
     * @return A reference to this `ProjectNeighborEmbedding` object.
     */
    ProjectNeighborEmbedding& set_num_threads(int n = Defaults::num_threads) {
        nthreads = n;
        return *this;
    }

    /**
     * @param n Number of MADs to use for identifying outlier neighbors.
     * Smaller values improve robustness to outliers at the cost of reducing the stability of the averages.
     *
     * @return A reference to this `ProjectNeighborEmbedding` object.
     */
    ProjectNeighborEmbedding& set_nmads(double n = Defaults::nmads) {
        nmads = n;
        return *this;
    }

    /**
     * @param s Scaling of the bandwidth for the gaussian kernel.
     * Larger values reduce the influence of the closest neighbor in the reference on the projected location of each test cell - 
     * this usually results in less clumping in the visualization.
     *
     * @return A reference to this `ProjectNeighborEmbedding` object.
     */
    ProjectNeighborEmbedding& set_scale(double s = Defaults::scale) {
        scale = s;
        return *this;
    }

private:
    int nthreads = Defaults::num_threads;
    int num_neighbors = Defaults::num_neighbors;
    double nmads = Defaults::nmads;
    double scale = Defaults::scale;
    bool approximate = Defaults::approximate;

    template<typename Index, typename Float>
    std::vector<std::vector<Index> > find_embedded_neighbors(int ndim, size_t nobs, const Float* embedding) const {
        std::shared_ptr<knncolle::Base<Index, Float> > ptr;
        if (approximate) {
            ptr.reset(new knncolle::AnnoyEuclidean<Index, Float>(ndim, nobs, embedding));
        } else {
            ptr.reset(new knncolle::VpTreeEuclidean<Index, Float>(ndim, nobs, embedding));
        }

        std::vector<std::vector<Index> > output(nobs);

#ifndef SCRAN_CUSTOM_PARALLEL
        #pragma omp parallel for num_threads(nthreads)
        for (size_t o = 0; o < nobs; ++o) {
#else
        SCRAN_CUSTOM_PARALLEL(nobs, [&](size_t start, size_t end) -> void {
        for (size_t o = start; o < end; ++o) {
#endif

            auto current = ptr->find_nearest_neighbors(embedding + o * ndim, num_neighbors);
            auto& store = output[o];
            store.reserve(current.size() + 1);
            store.push_back(o); // adding self for easier iteration later.
            for (const auto& x : current) {
                store.push_back(x.first);
            }

#ifndef SCRAN_CUSTOM_PARALLEL
        }
#else
        }
        }, nthreads);
#endif

        return output;
    }

    template<typename Index, typename Float>
    void project_location(
            const std::vector<std::pair<Index, Float> >& neighbors, 
            const std::vector<std::vector<Index> >& embedded_neighbors, 
            int ndim, 
            const Float* ref,
            const Float* coord,
            int nembed,
            const Float* embedding, 
            std::unordered_map<Index, Float>& cache, 
            Float* output) 
    const {
        if (neighbors.empty()) {
            std::fill(output, output + nembed, std::numeric_limits<Float>::quiet_NaN());
            return;
        }

        Float bandwidth = neighbors.front().second;
        if (bandwidth == 0) {
            auto src = embedding + nembed * neighbors.front().first;
            std::copy(src, src + nembed, output);
            return;
        }

        // Compute the weight using a gaussian kernel.
        Float denom = bandwidth * bandwidth * 2 * scale;
        auto compute_weight = [&](Float dist2) -> Float {
            // Protect against float overflow. 
            if (denom > 1 || std::numeric_limits<Float>::max() * denom > dist2) {
                return std::exp(-dist2 / denom);
            } else {
                return 0;
            }
        };

        // Prefilling the immediate neighbors to avoid unnecessary compute.
        cache.clear();
        for (const auto& neighbor : neighbors) {
            cache[neighbor.first] = compute_weight(neighbor.second * neighbor.second);
        }

        // Finding the neighbor with the max total weight amongst its own neighbor set.
        Float max_weight = 0;
        Index best_neighbor = neighbors.front().first;

        for (const auto& neighbor : neighbors) {
            const auto& candidates = embedded_neighbors[neighbor.first];
            Float current_weight = 0;

            for (auto x : candidates) {
                Float w;
                auto it = cache.find(x);
                if (it == cache.end()) {
                    Float dist2 = 0;
                    auto pos = ref + x * ndim;
                    for (int d = 0; d < ndim; ++d) {
                        dist2 += (pos[d] - coord[d]) * (pos[d] - coord[d]);
                    }
                    w = compute_weight(dist2);
                    cache[x] = w;
                } else {
                    w = it->second;
                }
                current_weight += w;
            }

            if (current_weight > max_weight) {
                max_weight = current_weight;
                best_neighbor = neighbor.first;
            }
        }

        // Computing the average of the neighbor set with the largest weights.
        // The maximum weight better be positive!
        std::fill(output, output + nembed, 0);
        const auto& candidates = embedded_neighbors[best_neighbor];
        for (auto x : candidates) {
            auto w = cache.at(x) / max_weight;
            auto src = embedding + x * nembed;
            for (int e = 0; e < nembed; ++e) {
                output[e] += w * src[e];
            }
        }
    }

public:
    /**
     * @tparam Index Integer type for the indices.
     * @tparam Float Floating point type for distances and embeddings.
     *
     * @param neighhbors Vector of vectors of neighbors for each cell in the test dataset.
     * Each entry of the outer vector corresponds to a test cell while each entry of the inner vector contains the index of and distance to a neighboring cell in the reference dataset.
     * Inner vectors should be sorted by increasing distance.
     * @param nembed Number of dimensions of the destination embedding.
     * @param ref_embedding Pointer to a column-major array of dimensions (rows) and cells (columns),
     * containing coordinates of the destination embedding for the reference dataset.
     * @param[out] output Pointer to a column-major array of dimensions (rows) and cells (columns),
     * to be filled with the projected coordinates in the destination embedding for the test dataset.
     * This should have number of rows and columns equal to `nembed` and `neighbors.size()`, respectively.
     */
    template<typename Index, typename Float>
    void run(int ndim, size_t nref, const Float* ref, size_t ntest, const Float* test, int nembed, const Float* ref_embedding, const std::vector<std::vector<std::pair<Index, Float> > >& neighbors, Float* output) const {
        if (ntest != neighbors.size()) {
            throw std::runtime_error("'neighbors' should have length equal to 'ntest'");
        }
        auto embedded_neighbors = find_embedded_neighbors<Index, Float>(nembed, nref, ref_embedding);

#ifndef SCRAN_CUSTOM_PARALLEL
        #pragma omp parallel num_threads(nthreads)
        {
#else
        SCRAN_CUSTOM_PARALLEL(ntest, [&](size_t start, size_t end) -> void {
#endif

            std::unordered_map<Index, Float> cache;

#ifndef SCRAN_CUSTOM_PARALLEL
            #pragma omp for
            for (size_t o = 0; o < ntest; ++o) {
#else
            for (size_t o = start; o < end; ++o) {
#endif

                project_location(neighbors[o], embedded_neighbors, ndim, ref, test + o * ndim, nembed, ref_embedding, cache, output + o * nembed);

#ifndef SCRAN_CUSTOM_PARALLEL
            }
        }
#else
            }
        }, nthreads);
#endif
    }

    /**
     * @tparam Index Integer type for the indices.
     * @tparam Float Floating point type for distances and embeddings.
     *
     * @param index Pointer to a `knncolle::Base` object containing a neighbor search index for the reference dataset,
     * constructed from the source embedding.
     * @param nembed Number of dimensions of the destination embedding.
     * @param ref_embedding Pointer to a column-major array of dimensions (rows) and cells (columns),
     * containing coordinates of the destination embedding for the reference dataset.
     * @param[out] output Pointer to a column-major array of dimensions (rows) and cells (columns),
     * to be filled with the projected coordinates in the destination embedding for the test dataset.
     * This should have number of rows and columns equal to `nembed` and `neighbors.size()`, respectively.
     */
    template<typename Index, typename Float>
    void run(int ndim, size_t nref, const Float* ref, size_t ntest, const Float* test, int nembed, const Float* ref_embedding, const knncolle::Base<Index, Float>* index, Float* output) const {
        if (nref != index->nobs()) {
            throw std::runtime_error("'index' should have number of observations equal to 'nref'");
        }
        if (ndim != index->ndim()) {
            throw std::runtime_error("'index' should have number of dimensions equal to 'ndim'");
        }
        auto embedded_neighbors = find_embedded_neighbors<Index, Float>(nembed, nref, ref_embedding);

#ifndef SCRAN_CUSTOM_PARALLEL
        #pragma omp parallel num_threads(nthreads)
        {
#else
        SCRAN_CUSTOM_PARALLEL(ntest, [&](size_t start, size_t end) -> void {
#endif

            std::unordered_map<Index, Float> cache;

#ifndef SCRAN_CUSTOM_PARALLEL
            #pragma omp for
            for (size_t o = 0; o < ntest; ++o) {
#else
            for (size_t o = start; o < end; ++o) {
#endif

                auto current = index->find_nearest_neighbors(test + o * ndim, num_neighbors);
                project_location(current, embedded_neighbors, ndim, ref, test + o * ndim, nembed, ref_embedding, cache, output + o * nembed);

#ifndef SCRAN_CUSTOM_PARALLEL
            }
        }
#else
            }
        }, nthreads);
#endif
    }

    /**
     * @tparam Index Integer type for the indices.
     * @tparam Float Floating point type for distances and embeddings.
     *
     * @param ndim Number of dimensions in the source embedding.
     * @param nref Number of cells in the reference dataset.
     * @param[in] ref Pointer to a column-major array of coordinates of dimensions (rows) and cells (columns) for the source embedding of the reference dataset.
     * @param ntest Number of cells in the test dataset.
     * @param[in] ref Pointer to a column-major array of coordinates of dimensions (rows) and cells (columns) for the source embedding of the test dataset.
     * @param nembed Number of dimensions of the destination embedding.
     * @param ref_embedding Pointer to a column-major array of dimensions (rows) and cells (columns),
     * containing coordinates of the destination embedding for the reference dataset.
     * @param[out] output Pointer to a column-major array of dimensions (rows) and cells (columns),
     * to be filled with the projected coordinates in the destination embedding for the test dataset.
     * This should have number of rows and columns equal to `nembed` and `neighbors.size()`, respectively.
     */
    template<typename Index = int, typename Float>
    void run(int ndim, size_t nref, const Float* ref, size_t ntest, const Float* test, int nembed, const Float* ref_embedding, Float* output) const {
        std::shared_ptr<knncolle::Base<Index, Float> > ptr;
        if (approximate) {
            ptr.reset(new knncolle::AnnoyEuclidean<Index, Float>(ndim, nref, ref));
        } else {
            ptr.reset(new knncolle::VpTreeEuclidean<Index, Float>(ndim, nref, ref));
        }
        run(ndim, nref, ref, ntest, test, nembed, ref_embedding, ptr.get(), output);
    }

    /**
     * @tparam Index Integer type for the indices.
     * @tparam Float Floating point type for distances and embeddings.
     *
     * @param ndim Number of dimensions in the source embedding.
     * @param nref Number of cells in the reference dataset.
     * @param[in] ref Pointer to a column-major array of coordinates of dimensions (rows) and cells (columns) for the source embedding of the reference dataset.
     * @param ntest Number of cells in the test dataset.
     * @param[in] ref Pointer to a column-major array of coordinates of dimensions (rows) and cells (columns) for the source embedding of the test dataset.
     * @param nembed Number of dimensions of the destination embedding.
     * @param ref_embedding Pointer to a column-major array of dimensions (rows) and cells (columns),
     * containing coordinates of the destination embedding for the reference dataset.
     *
     * @return Vector containing the projected coordinates in the destination embedding for the test dataset.
     * This should be interpreted as a column-major array with number of rows and columns equal to `nembed` and `neighbors.size()`, respectively.
     */
    template<typename Index = int, typename Float>
    std::vector<Float> run(int ndim, size_t nref, const Float* ref, size_t ntest, const Float* test, int nembed, const Float* ref_embedding) const {
        std::vector<Float> output(ntest * nembed);
        run<Index, Float>(ndim, nref, ref, ntest, test, nembed, ref_embedding, output.data());
        return output;
    }
};

}

#endif
