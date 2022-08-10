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
 * For example, we could downsample a dataset with `DownsampleByNeighbors`, generate a 2D visualization for the subset from the PCs, and then use this class to project all cells onto the visualization.
 *
 * The projected location in the destination embedding for each test cell is defined as a weighted average of the coordinates of its neighbors.
 * The weight for each neighbor is a function of its distances to the test cell in the source embedding.
 * We use a tricube weighting scheme so that distant neighbors in low-density regions are given less weight in the average.
 * The bandwidth for each test cell is defined as the largest distance in its set of neighbors. 
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
        static constexpr int num_neighbors = 20;

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
     * 
     * Note that this parameter only has an effect in the `run()` method that accepts a data matrix for the source embedding.
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

private:
    int nthreads = Defaults::num_threads;
    int num_neighbors = Defaults::num_neighbors;
    double nmads = Defaults::nmads;
    bool approximate = Defaults::approximate;

    template<typename Index, typename Float>
    void compute_weighted_average(const std::vector<std::pair<Index, Float> >& neighbors, Float* buffer, int ndim, const Float* embedding, Float* output) const {
        size_t nneighbors = neighbors.size();

        Float bandwidth = 0;
        if (nneighbors) {
            for (size_t n = 0; n < nneighbors; ++n) {
                buffer[n] = neighbors[n].second;
            }
            Float median = tatami::stats::compute_median<Float>(buffer, nneighbors);

            for (size_t n = 0; n < nneighbors; ++n) {
                buffer[n] = std::abs(neighbors[n].second - median);
            }

            // Scaling equivalence with a normal distribution's SD. Not sure
            // how much difference it makes, but we do it for IsOutlier so we
            // might as well do it here.
            Float mad = tatami::stats::compute_median<Float>(buffer, nneighbors) * 1.4826; 

            bandwidth = median + mad * nmads;
        }

        Float total = 0;
        if (bandwidth) {
            for (size_t i = 0; i < neighbors.size(); ++i) {
                Float diff = std::min(1.0, neighbors[i].second / bandwidth);
                Float sub = 1 - diff * diff * diff;
                buffer[i] = sub * sub * sub;
                total += buffer[i];
            }
        }

        if (total == 0) {
            if (nneighbors) { // everyone's equal distance.
                std::fill(output, output + ndim, 0);
                for (size_t i = 0; i < neighbors.size(); ++i) {
                    auto nptr = embedding + neighbors[i].first * ndim;
                    for (int d = 0; d < ndim; ++d) {
                        output[d] += nptr[d];
                    }
                }
                for (int d = 0; d < ndim; ++d) {
                    output[d] /= neighbors.size();
                }
            } else { // no neighbors at all.
                std::fill(output, output + ndim, std::numeric_limits<Float>::quiet_NaN());
            }
        } else {
            std::fill(output, output + ndim, 0);
            for (size_t i = 0; i < neighbors.size(); ++i) {
                auto nptr = embedding + neighbors[i].first * ndim;
                Float weight = buffer[i] / total;
                for (int d = 0; d < ndim; ++d) {
                    output[d] += nptr[d] * weight;
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
    void run(const std::vector<std::vector<std::pair<Index, Float> > >& neighbors, int nembed, const Float* ref_embedding, Float* output) const {
        size_t nobs = neighbors.size();

#ifndef SCRAN_CUSTOM_PARALLEL
        #pragma omp parallel num_threads(nthreads)
        {
#else
        SCRAN_CUSTOM_PARALLEL(nobs, [&](size_t start, size_t end) -> void {
#endif

            std::vector<Float> buffer;

#ifndef SCRAN_CUSTOM_PARALLEL
            #pragma omp for
            for (size_t o = 0; o < nobs; ++o) {
#else
            for (size_t o = start; o < end; ++o) {
#endif

                const auto& current = neighbors[o];
                buffer.resize(current.size());
                compute_weighted_average(current, buffer.data(), nembed, ref_embedding, output + o * nembed);

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
    void run(const knncolle::Base<Index, Float>* index, size_t ntest, const Float* test, int nembed, const Float* ref_embedding, Float* output) const {
        int ndim = index->ndim();

#ifndef SCRAN_CUSTOM_PARALLEL
        #pragma omp parallel num_threads(nthreads)
        {
#else
        SCRAN_CUSTOM_PARALLEL(ntest, [&](size_t start, size_t end) -> void {
#endif

            std::vector<Float> buffer(num_neighbors);

#ifndef SCRAN_CUSTOM_PARALLEL
            #pragma omp for
            for (size_t o = 0; o < ntest; ++o) {
#else
            for (size_t o = start; o < end; ++o) {
#endif

                auto current = index->find_nearest_neighbors(test + o * ndim, num_neighbors);
                buffer.resize(current.size());
                compute_weighted_average(current, buffer.data(), nembed, ref_embedding, output + o * nembed);

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
    template<typename Index, typename Float>
    void run(int ndim, size_t nref, const Float* ref, size_t ntest, const Float* test, int nembed, const Float* ref_embedding, Float* output) const {
        std::shared_ptr<knncolle::Base<Index, Float> > ptr;
        if (approximate) {
            ptr.reset(new knncolle::AnnoyEuclidean<Index, Float>(ndim, nref, ref));
        } else {
            ptr.reset(new knncolle::VpTreeEuclidean<Index, Float>(ndim, nref, ref));
        }
        run(ptr.get(), ntest, test, nembed, ref_embedding, output);
    }
};

}

#endif
