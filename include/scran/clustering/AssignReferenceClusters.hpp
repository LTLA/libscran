#ifndef SCRAN_ASSIGN_REFERENCE_CLUSTERS_HPP
#define SCRAN_ASSIGN_REFERENCE_CLUSTERS_HPP

#include "../utils/macros.hpp"

#include <vector>
#include <memory>

#include "knncolle/knncolle.hpp"

/**
 * @file AssignReferenceClusters.hpp
 *
 * @brief Assign cells to their closest reference cluster.
 */

namespace scran {

/**
 * @brief Assign cells to their closest reference cluster.
 *
 * This uses a [**SingleR**](https://bioconductor.org/packages/SingleR)-like approach to assign a cell to a cluster in a reference dataset.
 * The idea is that we have a test dataset containing cells with no clusters, and a reference dataset containing cluster assignments.
 * For each test cell, we compute a score for each reference cluster, defined as a certain (low) quantile of the distances to all reference cells in that cluster.
 * The test cell is then assigned to the cluster with the lowest quantile.
 *
 * We use this approach as it adjusts for differences in the number of reference cells in each cluster.
 * For comparison, a simpler approach would be to just define the cluster of a test cell as the majority of the cluster assignments of its neighbors in the reference dataset.
 * This approach would favor assignment of test cells to clusters with more reference cells, simply because it is easier to obtain a majority.
 * Our approach is also better than using the closest centroid of each reference cluster as the quantile accounts for heterogeneity within each cluster;
 * more dispersed clusters are not overly penalized for having many observations far from the centroid.
 * 
 * The reference dataset is usually generated as a subset of the test dataset, e.g., using `DownsampleByNeighbors`.
 * In this manner, we can quickly cluster on the subset and then propagate assignments back to the full dataset.
 * Note that this function expects low-dimensional coordinates as input - see [**singlepp**](https://ltla.github.io/singlepp) for classification based on the original count matrix.
 */
class AssignReferenceClusters {
public:
    /**
     * @brief Default parameter settings.
     */
    struct Defaults {
        /**
         * See `set_quantile()` for details.
         */
        static constexpr double quantile = 0.2;

        /**
         * See `set_approximate()` for details.
         */
        static constexpr bool approximate = false;

        /**
         * See `set_num_threads()` for details.
         */
        static constexpr int num_threads = 1;
    };

    /**
     * @param q Quantile to use for creating a per-cluster score.
     * Smaller values focus more on the local neighborhood around each test cell, while larger values focus on the behavior of the bulk of the cluster. 
     *
     * @return A reference to this `DownsampleByNeighbors` object.
     */
    AssignReferenceClusters& set_quantile(double q = Defaults::quantile) {
        quantile = q;
        return *this;
    }

    /**
     * @param n Number of threads to use for neighbor detection.
     *
     * @return A reference to this `DownsampleByNeighbors` object.
     */
    AssignReferenceClusters& set_num_threads(int n = Defaults::num_threads) {
        nthreads = n;
        return *this;
    }

    /**
     * @param a Whether approximate neighbor detection should be used.
     *
     * @return A reference to this `DownsampleByNeighbors` object.
     */
    AssignReferenceClusters& set_approximate(int a = Defaults::approximate) {
        approximate = a;
        return *this;
    }

private:
    double quantile = Defaults::quantile;
    int nthreads = Defaults::num_threads;
    bool approximate = Defaults::approximate;

public:
    /**
     * Vector of neighbor search indices for the reference clusters.
     * This has length equal to the number of clusters, where each entry contains a search index for the corresponding cluster in `clusters` of `build()`.
     *
     * @tparam Index Integer type for the cell indices.
     * @tparam Float Floating point type for the distances.
     */
    template<typename Index, typename Float>
    using Prebuilt = std::vector<std::shared_ptr<knncolle::Base<Index, Float> > >;

    /**
     * Build the search indices from the reference dataset, to be used for cluster assignment in `run()`.
     *
     * @tparam Index Integer type for the cell indices.
     * @tparam Float Floating point type for the distances.
     * @tparam Cluster Integer type for the cluster assignments.
     *
     * @param ndim Number of dimensions.
     * @param nref Number of cells in the reference dataset.
     * @param[in] ref Pointer to a column-major array of coordinates of dimensions (rows) and cells (columns) for the reference dataset.
     * This should be a low-dimensional embedding, e.g., from `RunPCA`.
     * @param[in] clusters Pointer to an array of length equal to `nref`, containing the cluster assignment for each cell in the reference.
     * All integers in $[0, N)$ should be present at least once, where $N$ is the total number of unique clusters.
     * 
     * @return The prebuilt indices, for use in `run()`.
     */
    template<typename Index = int, typename Float, typename Cluster>
    Prebuilt<Index, Float> build(int ndim, size_t nref, const Float* ref, const Cluster* clusters) const {
        Cluster nclusters = (nref ? *(std::max_element(clusters, clusters + nref)) + 1 : 0);

        std::vector<std::vector<Float> > buffers(nclusters);
        for (size_t r = 0; r < nref; ++r) {
            auto& current = buffers[clusters[r]];
            auto ptr = ref + r * ndim;
            current.insert(current.end(), ptr, ptr + ndim);
        }

        Prebuilt<Index, Float> indices(nclusters);

#ifndef SCRAN_CUSTOM_PARALLEL
        #pragma omp parallel for num_threads(nthreads)
        for (Cluster c = 0; c < nclusters; ++c) {
#else
        SCRAN_CUSTOM_PARALLEL(nclusters, [&](size_t start, size_t end) -> void {
        for (size_t c = start; c < end; ++c) {
#endif

            const auto& buffer = buffers[c];
            size_t nobs = buffer.size() / ndim;
            if (approximate) {
                indices[c].reset(new knncolle::AnnoyEuclidean<Index, Float>(ndim, nobs, buffer.data()));
            } else {
                indices[c].reset(new knncolle::VpTreeEuclidean<Index, Float>(ndim, nobs, buffer.data()));
            }

#ifndef SCRAN_CUSTOM_PARALLEL
        }
#else
        }
        }, nthreads);
#endif

        return indices;
    }

public:
    /**
     * Assign cells in the test dataset to reference clusters.
     *
     * @tparam Index Integer type for the cell indices.
     * @tparam Float Floating point type for the distances.
     * @tparam Cluster Integer type for the cluster assignments.
     *
     * @param ndim Number of dimensions.
     * @param nref Number of cells in the reference dataset.
     * @param[in] ref Pointer to a column-major array of coordinates of dimensions (rows) and cells (columns) for the reference dataset.
     * This should be a low-dimensional embedding, e.g., from `RunPCA`.
     * @param[in] clusters Pointer to an array of length equal to `nref`, containing the cluster assignment for each cell in the reference.
     * All integers in $[0, N)$ should be present at least once, where $N$ is the total number of unique clusters.
     * @param ntest Number of cells in the test dataset.
     * @param[in] test Pointer to a column-major array of coordinates of dimensions (rows) and cells (columns) for the test dataset.
     * This should be a low-dimensional embedding in the same space as `ref`.
     * 
     * @return Vector of cluster assignments for each cell in the test dataset.
     */
    template<typename Index = int, typename Float, typename Cluster>
    std::vector<Cluster> run(int ndim, size_t nref, const Float* ref, const Cluster* clusters, size_t ntest, const Float* test) const {
        std::vector<Cluster> output(ntest);
        auto built = build(ndim, nref, ref, clusters);
        run(built, ntest, test, output.data());
        return output;
    }

    /**
     * Assign cells in the test dataset to reference clusters.
     *
     * @tparam Index Integer type for the cell indices.
     * @tparam Float Floating point type for the distances.
     * @tparam Cluster Integer type for the cluster assignments.
     *
     * @param ndim Number of dimensions.
     * @param nref Number of cells in the reference dataset.
     * @param[in] ref Pointer to a column-major array of coordinates of dimensions (rows) and cells (columns) for the reference dataset.
     * This should be a low-dimensional embedding, e.g., from `RunPCA`.
     * @param[in] clusters Pointer to an array of length equal to `nref`, containing the cluster assignment for each cell in the reference.
     * All integers in $[0, N)$ should be present at least once, where $N$ is the total number of unique clusters.
     * @param ntest Number of cells in the test dataset.
     * @param[in] test Pointer to a column-major array of coordinates of dimensions (rows) and cells (columns) for the test dataset.
     * This should be a low-dimensional embedding in the same space as `ref`.
     * @param[out] output Pointer to an array of length equal to `ntest`.
     * On output, this is filled with the cluster assignments for each cell in the test dataset.
     */
    template<typename Index = int, typename Float, typename Cluster>
    void run(int ndim, size_t nref, const Float* ref, const Cluster* clusters, size_t ntest, const Float* test, Cluster* output) const {
        auto built = build(ndim, nref, ref, clusters);
        run(built, ntest, test, output);
    }

    /**
     * Assign cells in the test dataset to reference clusters based on pre-built search indices.
     *
     * @tparam Index Integer type for the cell indices.
     * @tparam Float Floating point type for the distances.
     * @tparam Cluster Integer type for the cluster assignments.
     *
     * @param built Prebuilt search indices created by `build()`.
     * @param ntest Number of cells in the test dataset.
     * @param[in] test Pointer to a column-major array of coordinates of dimensions (rows) and cells (columns) for the test dataset.
     * This should be a low-dimensional embedding in the same space as that used to construct `built`.
     * @param[out] output Pointer to an array of length equal to `ntest`.
     * On output, this is filled with the cluster assignments for each cell in the test dataset.
     */
    template<typename Index, typename Float, typename Cluster>
    void run(const Prebuilt<Index, Float>& built, size_t ntest, const Float* test, Cluster* output) const {
        size_t nclusters = built.size();
        int ndim = (built.size() ? built.front()->ndim() : 0); // setting to zero doesn't matter if there are no clusters.

        // This part is pretty much stolen from singlepp.
        std::vector<Index> search(nclusters);
        std::vector<Float> left(nclusters), right(nclusters);

        for (size_t c = 0; c < nclusters; ++c) {
            size_t denom = built[c]->nobs() - 1;
            double prod = denom * quantile;
            auto k = std::ceil(prod) + 1;
            search[c] = k;
            left[c] = static_cast<double>(k - 1) - prod;
            right[c] = prod - static_cast<double>(k - 2);
        }

#ifndef SCRAN_CUSTOM_PARALLEL
        #pragma omp parallel num_threads(nthreads)
        for (size_t o = 0; o < ntest; ++o) {
#else
        SCRAN_CUSTOM_PARALLEL(ntest, [&](size_t start, size_t end) -> void {
        for (size_t o = start; o < end; ++o) {
#endif

            auto ptr = test + o * ndim;
            double best = -1;
            Cluster choice;

            for (size_t c = 0; c < nclusters; ++c) {
                auto k = search[c];
                auto found = built[c]->find_nearest_neighbors(ptr, k);
                auto curscore = found[k - 1].second;
                if (k != 1) {
                    curscore = left[c] * found[k - 2].second + right[c] * curscore;
                }

                if (best < 0 || curscore < best) {
                    best = curscore;
                    choice = c;
                }
            }

            output[o] = choice;

#ifndef SCRAN_CUSTOM_PARALLEL
        }
#else
        }
        }, nthreads);
#endif
    }
};

}

#endif
