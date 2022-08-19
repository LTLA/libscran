#ifndef SCRAN_PROJECT_EMBEDDING_HPP
#define SCRAN_PROJECT_EMBEDDING_HPP

#include "../utils/macros.hpp"

#include <vector>
#include <memory>
#include <cmath>

#include "knncolle/knncolle.hpp"
#include "aarand/aarand.hpp"

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
 * The projected location in the destination embedding for each test cell is determined by voting for its best neighbor.
 * For each test cell, we find all of its "primary" neighbors in the source embedding.
 * For each primary neighbor, we examine its own nearest neighbors in the destination embedding.
 * For that set of neighbor's neighbors, we compute the distance to the test cell, and we convert that into a weight using a Gaussian kernel.
 * Summation of the weights yields a score for the primary neighbor; the primary neighbor with the highest score is elected as the projection location.
 *
 * This approach avoids introducing any artifacts in the destination embedding.
 * In particular, we avoid the formation of spurious intermediate populations, which is common when naively averaging the destination coordinates of the primary neighbors.
 * This is because many embedding algorithms will separate primary neighbors in the destination embedding, such that an average will not be close to any of those neighbors.
 * Additionally, we do not jitter the test cells around the elected primary neighbor, as this forms visually distracting clumps when many test cells elect the same primary neighbor.
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

        static constexpr int steps = 3;
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

private:
    int nthreads = Defaults::num_threads;
    int num_neighbors = Defaults::num_neighbors;
    bool approximate = Defaults::approximate;
    int steps = Defaults::steps;

private:
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

    template<typename Index, typename Float, typename Thing,  class Rng>
    static Index choose_step(const std::vector<Thing>& neighbors, std::vector<Float>& buffer, Rng& rng) {
        for (size_t n = 1; n < buffer.size(); ++n) {
            buffer[n] += buffer[n-1];
        }

        Float total_weight = buffer.back();
        Float sample = total_weight * aarand::standard_uniform<Float>(rng);

        for (size_t n = 0; n < neighbors.size(); ++n) {
            if (buffer[n] >= sample) {
                if constexpr(std::is_same<Thing, Index>::value) {
                    return neighbors[n];
                } else {
                    return neighbors[n].first;
                }
            }
        }

        if constexpr(std::is_same<Thing, Index>::value) {
            return neighbors[0];
        } else {
            return neighbors[0].first;
        }
    }

    template<typename Index, typename Float>
    void project_location(
            size_t index,
            const std::vector<std::pair<Index, Float> >& neighbors, 
            const std::vector<std::vector<Index> >& embedded_neighbors, 
            int ndim, 
            const Float* ref,
            const Float* test,
            int nembed,
            const Float* embedding, 
            std::unordered_map<Index, Float>& cache, 
            std::vector<Float>& buffer,
            Float* output) 
    const {
        auto curout = output + index * nembed;
        if (neighbors.empty()) {
            std::fill(curout, curout + nembed, std::numeric_limits<Float>::quiet_NaN());
            return;
        }

        Float bandwidth = neighbors.front().second;
        if (bandwidth == 0) {
            auto src = embedding + nembed * neighbors.front().first;
            std::copy(src, src + nembed, curout);
            return;
        }

        // Compute the weight using a gaussian kernel.
        Float denom = bandwidth * bandwidth * 2;
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
        buffer.clear();
        for (const auto& neighbor : neighbors) {
            auto w = compute_weight(neighbor.second * neighbor.second);
            cache[neighbor.first] = w;
            buffer.push_back(w);
        }

        // Choosing the starting point.
        std::mt19937_64 rng(index + 5867);
        Index current_position = this->choose_step<Index, Float>(neighbors, buffer, rng);

        // Performing a random walk around the starting point.
        auto coord = test + index * ndim;
        for (int s = 0; s <= steps; ++s) {
            const auto& candidates = embedded_neighbors[current_position];
            buffer.clear();

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
                buffer.push_back(w);
            }

            if (s < steps) {
                current_position = this->choose_step<Index, Float>(candidates, buffer, rng);
            } else {
                // Final location uses a weighted average where the existing 
                // weights are jiggled a little to provide some jitter.
                Float total_weight = 0;
                for (size_t n = 0; n < buffer.size(); ++n) {
                    auto w = buffer[n] * std::max(aarand::standard_uniform<Float>(rng), static_cast<Float>(0.000001));
                    total_weight += w;
                    auto src = embedding + candidates[n] * nembed;
                    for (int e = 0; e < nembed; ++e) {
                        curout[e] += w * src[e];
                    }
                }

                for (int e = 0; e < nembed; ++e) {
                    curout[e] /= total_weight;
                }
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
            std::vector<Float> buffer;

#ifndef SCRAN_CUSTOM_PARALLEL
            #pragma omp for
            for (size_t o = 0; o < ntest; ++o) {
#else
            for (size_t o = start; o < end; ++o) {
#endif

                const auto& current = neighbors[o];
                project_location(o, current, embedded_neighbors, ndim, ref, test, nembed, ref_embedding, cache, buffer, output);

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
            std::vector<Float> buffer;

#ifndef SCRAN_CUSTOM_PARALLEL
            #pragma omp for
            for (size_t o = 0; o < ntest; ++o) {
#else
            for (size_t o = start; o < end; ++o) {
#endif

                auto current = index->find_nearest_neighbors(test + o * ndim, num_neighbors);
                project_location(o, current, embedded_neighbors, ndim, ref, test, nembed, ref_embedding, cache, buffer, output);

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
