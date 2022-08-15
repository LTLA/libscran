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
 * This class assigns the test cell to the most frequent cluster among its neighbors in the reference dataset.
 * The reference dataset is usually generated as a subset of the test dataset, e.g., using `DownsampleByNeighbors`.
 * In this manner, we can quickly cluster on the subset and then propagate assignments back to the full dataset.
 * 
 * Admittedly, this is not the most "correct" approach for reference assignment as it favors abundant clusters that are more likely to achieve a majority.
 * Nonetheless, we use it because (i) it's fast and (ii) any inaccuracies near the cluster boundaries are acceptable given that the boundaries are arbitrary anyway.
 * A more accurate approach would be to use **singlepp**-like classification but this is very slow for single-cell reference datasets.
 *
 * Note that this function expects low-dimensional coordinates for neighbor searching.
 * See [**singlepp**](https://ltla.github.io/singlepp) for classification based on the original count matrix instead.
 */
class AssignReferenceClusters {
public:
    /**
     * @brief Default parameter settings.
     */
    struct Defaults {
        /**
         * See `set_num_neighbors()` for details.
         */
        static constexpr double num_neighbors = 20;

        /**
         * See `set_approximate()` for details.
         */
        static constexpr bool approximate = false;

        /**
         * See `set_num_threads()` for details.
         */
        static constexpr int num_threads = 1;

        /**
         * See `set_report_best()` for details.
         */
        static constexpr bool report_best = true;

        /**
         * See `set_report_best()` for details.
         */
        static constexpr bool report_second = true;
    };

    /**
     * @param k Number of neighbors to use for assigning a cluster.
     * Smaller values focus more on the local neighborhood around each test cell, while larger values focus on the behavior of the bulk of the cluster. 
     *
     * @return A reference to this `AssignReferenceClusters` object.
     */
    AssignReferenceClusters& set_num_neighbors(int n = Defaults::num_neighbors) {
        num_neighbors = n;
        return *this;
    }

    /**
     * @param n Number of threads to use for neighbor detection.
     *
     * @return A reference to this `AssignReferenceClusters` object.
     */
    AssignReferenceClusters& set_num_threads(int n = Defaults::num_threads) {
        nthreads = n;
        return *this;
    }

    /**
     * @param a Whether approximate neighbor detection should be used.
     *
     * @return A reference to this `AssignReferenceClusters` object.
     */
    AssignReferenceClusters& set_approximate(int a = Defaults::approximate) {
        approximate = a;
        return *this;
    }

    /**
     * @param r Whether to report the proportion of the best cluster for each cell.
     * This can be a useful diagnostic to remove poor assignments when the best proportion is low.
     *
     * @return A reference to this `AssignReferenceClusters` object.
     */
    AssignReferenceClusters& set_report_best(bool r = Defaults::report_best) {
        report_best = r;
        return *this;
    }

    /**
     * @param r Whether to report the proportion of the second-best cluster for each cell.
     * This can be a useful diagnostic to remove ambiguous assignments when the second-best proportion is close to the best proportion.
     *
     * @return A reference to this `AssignReferenceClusters` object.
     */
    AssignReferenceClusters& set_report_second(bool r = Defaults::report_second) {
        report_second = r;
        return *this;
    }

private:
    int num_neighbors = Defaults::num_neighbors;
    int nthreads = Defaults::num_threads;
    bool approximate = Defaults::approximate;
    bool report_best = Defaults::report_best;
    bool report_second = Defaults::report_second;

    template<typename Index, typename Float, typename Cluster>
    static std::tuple<Cluster, Float, Float> assign(const std::vector<std::pair<Index, Float> >& neighbors, const Cluster* clusters, Cluster nclusters, int* buffer) {
        std::fill(buffer, buffer + nclusters, 0);
        for (const auto& n : neighbors) {
            ++(buffer[clusters[n.first]]);
        }

        Cluster best = 0, second = 0;
        bool no_second = true;
        for (Cluster c = 1; c < nclusters; ++c) {
            if (buffer[best] < buffer[c]) {
                second = best;
                best = c;
                no_second = false;
            } else if (no_second || buffer[second] < buffer[c]) {
                second = c;
                no_second = false;
            }
        }

        Float best_prop = (nclusters ? static_cast<Float>(buffer[best]) / neighbors.size() : std::numeric_limits<Float>::quiet_NaN());
        Float second_prop = (nclusters > 1 ? static_cast<Float>(buffer[second]) / neighbors.size() : std::numeric_limits<Float>::quiet_NaN());
        return std::make_tuple(best, best_prop, second_prop);
    }

    template<typename T>
    static T* harvest_pointer(std::vector<T>& source, bool doit) {
        return (doit ? source.data() : static_cast<T*>(NULL));
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
     * @param[out] assigned Pointer to an array of length equal to `ntest`.
     * On output, this is filled with the cluster assignments for each cell in the test dataset.
     * @param[out] best_prop Pointer to an array of length equal to `ntest`.
     * On output, this is filled with the proportion of neighbors supporting the cluster assignment in `assigned`.
     * If `NULL`, this value is not saved.
     * @param[out] second_prop Pointer to an array of length equal to `ntest`.
     * On output, this is filled with the second-highest proportion of neighbors for a cluster other than the one reported in `assigned`.
     * If `NULL`, this value is not saved.
     */
    template<typename Index = int, typename Float, typename Cluster>
    void run(int ndim, size_t nref, const Float* ref, const Cluster* clusters, size_t ntest, const Float* test, Cluster* assigned, Float* best_prop, Float* second_prop) const {
        std::shared_ptr<knncolle::Base<Index, Float> > ptr;
        if (approximate) {
            ptr.reset(new knncolle::AnnoyEuclidean<Index, Float>(ndim, nref, ref));
        } else {
            ptr.reset(new knncolle::VpTreeEuclidean<Index, Float>(ndim, nref, ref));
        }
        run(ptr.get(), clusters, ntest, test, assigned, best_prop, second_prop);
    }

    /**
     * Assign cells in the test dataset to reference clusters.
     *
     * @tparam Index Integer type for the cell indices.
     * @tparam Float Floating point type for the distances.
     * @tparam Cluster Integer type for the cluster assignments.
     *
     * @param index Pointer to a `knncolle::Base` neighbor search index, constructed from the reference dataset.
     * @param[in] clusters Pointer to an array of length equal to `nref`, containing the cluster assignment for each cell in the reference.
     * All integers in $[0, N)$ should be present at least once, where $N$ is the total number of unique clusters.
     * @param ntest Number of cells in the test dataset.
     * @param[in] test Pointer to a column-major array of coordinates of dimensions (rows) and cells (columns) for the test dataset.
     * This should be a low-dimensional embedding in the same space as the reference dataset used to construct `index`.
     * @param[out] assigned Pointer to an array of length equal to `ntest`.
     * On output, this is filled with the cluster assignments for each cell in the test dataset.
     * @param[out] best_prop Pointer to an array of length equal to `ntest`.
     * On output, this is filled with the proportion of neighbors supporting the cluster assignment in `assigned`.
     * If `NULL`, this value is not saved.
     * @param[out] second_prop Pointer to an array of length equal to `ntest`.
     * On output, this is filled with the second-highest proportion of neighbors for a cluster other than the one reported in `assigned`.
     * If `NULL`, this value is not saved.
     */
    template<typename Index = int, typename Float, typename Cluster>
    void run(const knncolle::Base<Index, Float>* index, const Cluster* clusters, size_t ntest, const Float* test, Cluster* assigned, Float* best_prop, Float* second_prop) const {
        Cluster nclusters = (index->nobs() ? *std::max_element(clusters, clusters + index->nobs()) + 1 : 0);
        size_t ndim = index->ndim();

#ifndef SCRAN_CUSTOM_PARALLEL
        #pragma omp parallel num_threads(nthreads)
        {
#else
        SCRAN_CUSTOM_PARALLEL(ntest, [&](size_t start, size_t end) -> void {
#endif

            std::vector<int> buffer(nclusters);

#ifndef SCRAN_CUSTOM_PARALLEL
            #pragma omp for
            for (size_t o = 0; o < ntest; ++o) {
#else
            for (size_t o = start; o < end; ++o) {
#endif

                auto neighbors = index->find_nearest_neighbors(test + o * ndim, num_neighbors);
                auto details = assign(neighbors, clusters, nclusters, buffer.data());
                assigned[o] = std::get<0>(details);
                if (best_prop != NULL) {
                    best_prop[o] = std::get<1>(details);
                }
                if (second_prop != NULL) {
                    second_prop[o] = std::get<2>(details);
                }

#ifndef SCRAN_CUSTOM_PARALLEL
            }
        }
#else
            }
        }, nthreads);
#endif
    }

    /**
     * Assign cells in the test dataset to reference clusters based on pre-built search indices.
     *
     * @tparam Index Integer type for the cell indices.
     * @tparam Float Floating point type for the distances.
     * @tparam Cluster Integer type for the cluster assignments.
     *
     * @param neighbors Precomputed neighbors for each test cell.
     * The outer vector should have length equal to the number of cells in the test dataset.
     * Each inner vector contains the neighbors for the corresponding test cell.
     * @param nref Number of cells in the reference dataset.
     * @param[in] clusters Pointer to an array of length equal to `nref`, containing the cluster assignment for each cell in the reference.
     * All integers in $[0, N)$ should be present at least once, where $N$ is the total number of unique clusters.
     * @param[out] assigned Pointer to an array of length equal to `neighbors.size()`.
     * On output, this is filled with the cluster assignments for each cell in the test dataset.
     * @param[out] best_prop Pointer to an array of length equal to `neighbors.size()`.
     * On output, this is filled with the proportion of neighbors supporting the cluster assignment in `assigned`.
     * If `NULL`, this value is not saved.
     * @param[out] second_prop Pointer to an array of length equal to `neighbors.size()`.
     * On output, this is filled with the second-highest proportion of neighbors for a cluster other than the one reported in `assigned`.
     * If `NULL`, this value is not saved.
     */
    template<typename Index, typename Float, typename Cluster>
    void run(const std::vector<std::vector<std::pair<Index, Float> > >& neighbors, size_t nref, const Cluster* clusters, Cluster* assigned, Float* best_prop, Float* second_prop) const {
        size_t ntest = neighbors.size();
        Cluster nclusters = (nref ? *std::max_element(clusters, clusters + nref) + 1 : 0);

#ifndef SCRAN_CUSTOM_PARALLEL
        #pragma omp parallel num_threads(nthreads)
        {
#else
        SCRAN_CUSTOM_PARALLEL(ntest, [&](size_t start, size_t end) -> void {
#endif

            std::vector<int> buffer(nclusters);

#ifndef SCRAN_CUSTOM_PARALLEL
            #pragma omp for
            for (size_t o = 0; o < ntest; ++o) {
#else
            for (size_t o = start; o < end; ++o) {
#endif

                auto details = assign(neighbors[o], clusters, nclusters, buffer.data());
                assigned[o] = std::get<0>(details);
                if (best_prop != NULL) {
                    best_prop[o] = std::get<1>(details);
                }
                if (second_prop != NULL) {
                    second_prop[o] = std::get<2>(details);
                }

#ifndef SCRAN_CUSTOM_PARALLEL
            }
        }
#else
            }
        }, nthreads);
#endif
    }

public:
    /**
     * @brief Results of the cluster assignment.
     *
     * @tparam Float Floating point type for the distances.
     * @tparam Cluster Integer type for the cluster assignments.
     */
    template<typename Float, typename Cluster>
    struct Results {
        /**
         * @cond
         */
        Results(size_t n, bool best, bool second) : assigned(n), best_prop(best ? n : 0), second_prop(second ? n : 0) {}
        /**
         * @endcond
         */
        
        /**
         * Vector of length equal to the number of cells in the test dataset, containing the assigned cluster for each cell.
         */
        std::vector<Cluster> assigned;

        /**
         * Vector of length equal to the number of cells in the test dataset, containing the proportion of neighbors supporting a cell's assignment in `assigned`.
         * If `set_report_best()` is `false`, this vector is empty.
         */
        std::vector<Float> best_prop;

        /**
         * Vector of length equal to the number of cells in the test dataset, containing the proportion of neighbors supporting the second-most frequent cluster for a cell.
         * If `set_report_second()` is `false`, this vector is empty.
         */
        std::vector<Float> second_prop;
    };

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
     * @return A `Results` object containing the cluster assignment results for each cell in the test dataset.
     */
    template<typename Index = int, typename Float, typename Cluster>
    Results<Float, Cluster> run(int ndim, size_t nref, const Float* ref, const Cluster* clusters, size_t ntest, const Float* test) const {
        Results<Float, Cluster> output(ntest, report_best, report_second);
        run(ndim, nref, ref, clusters, ntest, test, output.assigned.data(), harvest_pointer(output.best_prop, report_best), harvest_pointer(output.second_prop, report_second));
        return output;
    }

    /**
     * Assign cells in the test dataset to reference clusters.
     *
     * @tparam Index Integer type for the cell indices.
     * @tparam Float Floating point type for the distances.
     * @tparam Cluster Integer type for the cluster assignments.
     *
     * @param index Pointer to a `knncolle::Base` neighbor search index, constructed from the reference dataset.
     * @param[in] clusters Pointer to an array of length equal to `nref`, containing the cluster assignment for each cell in the reference.
     * All integers in $[0, N)$ should be present at least once, where $N$ is the total number of unique clusters.
     * @param ntest Number of cells in the test dataset.
     * @param[in] test Pointer to a column-major array of coordinates of dimensions (rows) and cells (columns) for the test dataset.
     * This should be a low-dimensional embedding in the same space as the reference dataset used to construct `index`.
     *
     * @return A `Results` object containing the cluster assignment results for each cell in the test dataset.
     */
    template<typename Index = int, typename Float, typename Cluster>
    Results<Float, Cluster> run(const knncolle::Base<Index, Float>* index, const Cluster* clusters, size_t ntest, const Float* test) const {
        Results<Float, Cluster> output(ntest, report_best, report_second);
        run(index, clusters, ntest, test, output.assigned.data(), harvest_pointer(output.best_prop, report_best), harvest_pointer(output.second_prop, report_second));
        return output;
    }

    /**
     * Assign cells in the test dataset to reference clusters based on pre-built search indices.
     *
     * @tparam Index Integer type for the cell indices.
     * @tparam Float Floating point type for the distances.
     * @tparam Cluster Integer type for the cluster assignments.
     *
     * @param neighbors Precomputed neighbors for each test cell.
     * The outer vector should have length equal to the number of cells in the test dataset.
     * Each inner vector contains the neighbors for the corresponding test cell.
     * @param nref Number of cells in the reference dataset.
     * @param[in] clusters Pointer to an array of length equal to `nref`, containing the cluster assignment for each cell in the reference.
     * All integers in $[0, N)$ should be present at least once, where $N$ is the total number of unique clusters.
     * 
     * @return A `Results` object containing the cluster assignment results for each cell in the test dataset.
     */
    template<typename Index, typename Float, typename Cluster>
    Results<Float, Cluster> run(const std::vector<std::vector<std::pair<Index, Float> > >& neighbors, size_t nref, const Cluster* clusters) {
        Results<Float, Cluster> output(neighbors.size(), report_best, report_second);
        run(neighbors, nref, clusters, output.assigned.data(), harvest_pointer(output.best_prop, report_best), harvest_pointer(output.second_prop, report_second));
        return output;
    }
};

}

#endif
