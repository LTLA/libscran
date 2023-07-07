#ifndef SCRAN_DOWNSAMPLE_BY_NEIGHBORS
#define SCRAN_DOWNSAMPLE_BY_NEIGHBORS

#include "../utils/macros.hpp"

#include <vector>
#include <algorithm>
#include <limits>
#include <memory>

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
 * To do so, we identify the `k`-nearest neighbors of each cell and use that to define its local neighborhood.
 * We find the cell that does not belong in the local neighborhood of any previously retained cell,
 * and has the fewest neighbors in any of the local neighborhoods of previously retained cells;
 * ties are broken using the smallest distance to the cell's `k`-th neighbor (i.e., the densest region of space).
 * This cell is retained in the downsampled subset and we repeat this process until all cells have been processed.
 *
 * Each retained cell serves as a representative for up to `k` of its nearest neighboring cells.
 * This approach ensures that the downsampled points are well-distributed across the dataset.
 * Low-frequency subpopulations will always have at least a few representatives if they are sufficiently distant from other subpopulations.
 * In contrast, random sampling does not provide strong guarantees for capture of a rare subpopulation.
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
     * @param[out] assigned Vector of length equal to the number of cells in `neighbors`.
     * On completion, this contains the index of the representative for each cell in the original dataset.
     * `assigned` may also be a null pointer, in which case nothing is returned.
     *
     * @return Vector of indices of the chosen representative cells.
     * The length of this vector depends on the dataset and the specified number of neighbors in `set_num_neighbors()`. 
     * Indices are sorted in increasing order.
     */
    template<typename Index, typename Float>
    std::vector<Index> run(const std::vector<std::vector<std::pair<Index, Float> > >& neighbors, Index* assigned) const {
        size_t nobs = neighbors.size();

        struct Observation {
            Observation(Float d, Index i, int c) : distance(d), index(i), covered(c) {}
            Float distance;
            Index index;
            int covered;
        };

        std::vector<Observation> ordered;
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
                ordered.emplace_back(dist_to_k, n, num_covered);
            }

            if (ordered.empty()) {
                break;
            }

            // Sorting by the number of covered neighbors (first) and the distance to the k-th neighbor (second).
            std::sort(ordered.begin(), ordered.end(), [](const Observation& left, const Observation& right) -> bool {
                if (left.covered < right.covered) {
                    return true;
                } else if (left.covered == right.covered) {
                    if (left.distance < right.distance) {
                        return true;
                    } else if (left.distance == right.distance) {
                        return left.index < right.index; // for tied distances, if two cells are each other's k-th neighbor.
                    }
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
                auto candidate = o.index;
                auto original_num = o.covered;
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

                    // Do this before 'covered' is modified.
                    if (assigned) {
                        assigned[candidate] = candidate;
                        for (const auto& x : current) {
                            if (!covered[x.first]) {
                                assigned[x.first] = candidate;
                            }
                        }
                    }

                    covered[candidate] = 1;
                    for (const auto& x : current) {
                        if (!covered[x.first]) {
                            covered[x.first] = 1;
                        }
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
     * @param[out] assigned Vector of length equal to the number of cells in `neighbors`.
     * On completion, this contains the index of the representative for each cell in the original dataset.
     * `assigned` may also be a null pointer, in which case nothing is returned.
     *
     * @return Vector of indices of the chosen representative cells.
     * The length of this vector depends on the dataset and the specified number of neighbors in `set_num_neighbors()`. 
     * Indices are sorted in increasing order.
     */
    template<typename Index = int, typename Float>
    std::vector<Index> run(int ndim, size_t nobs, const Float* data, Index* assigned) const {
        std::shared_ptr<knncolle::Base<Index, Float> > ptr;
        if (approximate) {
            ptr.reset(new knncolle::AnnoyEuclidean<Index, Float>(ndim, nobs, data));
        } else {
            ptr.reset(new knncolle::VpTreeEuclidean<Index, Float>(ndim, nobs, data));
        }
        return run(ptr.get(), assigned); 
    }

    /**
     * @tparam Index Integer type for the indices.
     * @tparam Float Floating point type for the distances.
     *
     * @param index Pointer to a `knncolle::Base` index object,
     * containing a pre-built neighbor index for a dataset.
     * @param[out] assigned Vector of length equal to the number of cells in `neighbors`.
     * On completion, this contains the index of the representative for each cell in the original dataset.
     * `assigned` may also be a null pointer, in which case nothing is returned.
     *
     * @return Vector of indices of the chosen representative cells.
     * The length of this vector depends on the dataset and the specified number of neighbors in `set_num_neighbors()`. 
     * Indices are sorted in increasing order.
     */
    template<typename Index, typename Float>
    std::vector<Index> run(const knncolle::Base<Index, Float>* index, Index* assigned) const {
        size_t nobs = index->nobs();
        std::vector<std::vector<std::pair<Index, Float> > > neighbors(nobs);

#ifndef SCRAN_CUSTOM_PARALLEL
        #pragma omp parallel for num_threads(nthreads)
        for (size_t i = 0; i < nobs; ++i) {
#else
        SCRAN_CUSTOM_PARALLEL([&](size_t, size_t start, size_t length) -> void {
        for (size_t i = start, end = start + length; i < end; ++i) {
#endif

            neighbors[i] = index->find_nearest_neighbors(i, num_neighbors);

#ifndef SCRAN_CUSTOM_PARALLEL
        }
#else
        }
        }, nobs, nthreads);
#endif

        return run(neighbors, assigned);
    }
};

}

#endif
