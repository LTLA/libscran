#ifndef SCRAN_DOWNSAMPLE_BY_NEIGHBORS
#define SCRAN_DOWNSAMPLE_BY_NEIGHBORS

#include "../utils/macros.hpp"

#include <vector>
#include <algorithm>
#include <limits>

#include "knncolle/knncolle.hpp"

/**
 * @file DownsampleByNeighbors.hpp
 *
 * @brief Downsample a dataset based on its neighbors.
 */

namespace scran {

/**
 * @brief Downsample a dataset based on its neighbors.
 *
 * This function generates a deterministic downsampling of a dataset based on nearest neighbors. 
 * The algorithm is fairly simple - we identify the `k`-nearest neighbors of each cell,
 * we sort all cells by the distance to their `k`-th neighbor,
 * and then we only retain cells in the subset if they are not neighbors of a cell with a lower `k`-th distance.
 * Thus, each retained cell serves as a representative for up to `k` of its neighboring cells.
 *
 * This approach ensures that the subsetted points are well-distributed across the dataset.
 * Low-frequency subpopulations will always have at least a few representatives if they are sufficiently distant from other subpopulations.
 * We also preserve the relative density across the dataset as more representatives will be generated from high-density regions. 
 * This simplifies the interpretation of analysis results generated from the subsetted dataset.
 */
class DownsampleByNeighbors {
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
         * See `set_num_threads()` for details.
         */
        static constexpr int num_threads = 1;

        /**
         * See `set_approximate()` for details.
         */
        static constexpr bool approximate = false;
    };

    /**
     * @param k Number of neighbors to use for downsampling.
     * Larger values result in more downsampling, at the cost of some speed.
     *
     * @return A reference to this `DownsampleByNeighbors` object.
     *
     * Note that this is only used in `run()` when a list of neighbors is not supplied.
     */
    DownsampleByNeighbors& set_num_neighbors(int k = Defaults::num_neighbors) {
        num_neighbors = k;
        return *this;
    }

    /**
     * @param n Number of threads to use for neighbor detection.
     *
     * @return A reference to this `DownsampleByNeighbors` object.
     *
     * Note that this is only used in `run()` when a list of neighbors is not supplied.
     */
    DownsampleByNeighbors& set_num_threads(int n = Defaults::num_threads) {
        nthreads = n;
        return *this;
    }

    /**
     * @param a Whether approximate neighbor detection should be used.
     *
     * @return A reference to this `DownsampleByNeighbors` object.
     *
     * Note that this is only used in `run()` when a data matrix is supplied.
     */
    DownsampleByNeighbors& set_approximate(int a = Defaults::approximate) {
        approximate = a;
        return *this;
    }

private:
    int num_neighbors = Defaults::num_neighbors;
    int nthreads = Defaults::num_threads;
    bool approximate = Defaults::approximate;

public:
    /**
     * @tparam Index Integer type for the indices.
     * @tparam Float Floating point type for the distances.
     *
     * @param neighbors Vector of vector of neighbors for each cell.
     * Each entry of the outer vector corresponds to a cell,
     * and each inner vector contains the index and distance of its nearest neighbors.
     * It is assumed that each inner vector is sorted by increasing distance.
     *
     * @return Vector of indices of the chosen representative cells.
     */
    template<typename Index, typename Float>
    std::vector<Index> run(const std::vector<std::vector<std::pair<Index, Float> > >& neighbors) const {
        size_t nobs = neighbors.size();

        std::vector<std::tuple<int, Float, Index> > ordered;
        ordered.reserve(nobs);
        std::vector<Index> chosen;
        std::vector<char> covered(nobs);

        while (1) {
            // Identifying all non-covered points and counting the number of covered neighbors.
            ordered.clear();
            bool fresh = chosen.empty();

            for (size_t n = 0; n < nobs; ++n) {
                if (covered[n]) {
                    continue;
                }
                const auto& current = neighbors[n];
                Float dist_to_k = (current.empty() ? std::numeric_limits<Float>::infinity() : current.back().second);

                int num_covered = 0;
                if (!fresh) { // must be zero at the start, so no need to loop.
                    for (auto x : current) {
                        num_covered += covered[x.first];
                    }
                }
                ordered.emplace_back(num_covered, dist_to_k, n);
            }

            if (ordered.empty()) {
                break;
            }

            // Sorting by the number of covered neighbors (first) and the distance to the k-th neighbor (second).
            std::sort(ordered.begin(), ordered.end(), [](const auto& left, const auto& right) -> bool {
                if (std::get<0>(left) < std::get<0>(right)) {
                    return true;
                } else if (std::get<0>(left) == std::get<0>(right)) {
                    return std::get<1>(left) < std::get<1>(right);
                }
                return false;
            });

            // Sweeping through the ordered list and choosing new representatives. This loop needs to
            // consider the possibility that a representative chosen in an earlier iteration will update
            // the coverage count in later iterations - so we stop iterating as soon as there is a 
            // potential change in the order that would require a resort via the outer 'while' loop.
            bool needs_resort = false;
            int resort_limit;

            for (const auto& o : ordered) {
                auto candidate = std::get<2>(o);
                auto original_num = std::get<0>(o);
                if (covered[candidate]) {
                    continue;
                } else if (needs_resort && original_num >= resort_limit) {
                    break;
                }

                const auto& current = neighbors[candidate];
                int updated_num = 0;
                for (auto x : current) {
                    updated_num += covered[x.first];
                }

                if (updated_num == original_num && (!needs_resort || updated_num < resort_limit)) {
                    chosen.push_back(candidate);
                    covered[candidate] = 1;
                    for (const auto& x : current) {
                        covered[x.first] = 1;
                    }
                } else {
                    if (!needs_resort) {
                        needs_resort = true;
                        resort_limit = updated_num;
                    } else if (resort_limit > updated_num) {
                        // Narrowing the resort limit. Note that this won't compromise previous uses of 'resort_limit'
                        // that resulted in a choice of a representative. 'updated_num' must be greater than the original 
                        // coverage number for the current iteration, and all previous choices of representatives must 
                        // have had equal or lower coverage numbers, so the narrowing wouldn't have affected them.
                        resort_limit = updated_num;
                    }
                }
            }
        }

        std::sort(chosen.begin(), chosen.end());
        return chosen;
    }

public:
    /**
     * @tparam Index Integer type for the indices.
     * @tparam Float Floating point type for the distances.
     *
     * @param ndim Number of dimensions.
     * @param nobs Number of observations, i.e., cells.
     * @param data Pointer to a column-major array of dimensions (rows) by cells (columns) containing coordinates for each cell, typically in some kind of embedding.
     *
     * @return Vector of indices of the chosen representative cells.
     */
    template<typename Index = int, typename Float>
    std::vector<Index> run(int ndim, size_t nobs, const Float* data) const {
        std::shared_ptr<knncolle::Base<int, Float> > ptr;
        if (approximate) {
            ptr.reset(new knncolle::AnnoyEuclidean<int, Float>(ndim, nobs, data));
        } else {
            ptr.reset(new knncolle::VpTreeEuclidean<int, Float>(ndim, nobs, data));
        }
        return run(ptr.get()); 
    }

    /**
     * @tparam Index Integer type for the indices.
     * @tparam Float Floating point type for the distances.
     *
     * @param index Pointer to a `knncolle::Base` index object,
     * containing a pre-built neighbor index for a dataset.
     *
     * @return Vector of indices of the chosen representative cells.
     */
    template<typename Index, typename Float>
    std::vector<Index> run(const knncolle::Base<Index, Float>* index) const {
        size_t nobs = index->nobs();
        std::vector<std::vector<std::pair<Index, Float> > > neighbors(nobs);

#ifndef SCRAN_CUSTOM_PARALLEL
        #pragma omp parallel for num_threads(nthreads)
        for (size_t i = 0; i < nobs; ++i) {
#else
        SCRAN_CUSTOM_PARALLEL(nobs, [&](size_t first, size_t last) -> void {
        for (size_t i = first; i < last; ++i) {
#endif

            neighbors[i] = index->find_nearest_neighbors(i, num_neighbors);

#ifndef SCRAN_CUSTOM_PARALLEL
        }
#else
        }
        }, nthreads);
#endif

        return run(neighbors);
    }
};

}

#endif
