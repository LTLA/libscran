#ifndef SCRAN_BUILDSNNGRAPH_HPP
#define SCRAN_BUILDSNNGRAPH_HPP

#include <vector>
#include <algorithm>
#include <memory>

#include "knncolle/knncolle.hpp"

/**
 * @file BuildSNNGraph.hpp
 *
 * @brief Build a shared nearest-neighbor graph on the cells.
 */

namespace scran {

/**
 * @brief Build a shared nearest-neighbor graph with cells as nodes.
 *
 * In a shared nearest neighbor graph, pairs of cells are connected to each other by an edge with weight determined from their shared nearest neighbors.
 * If two cells are close together but have distinct sets of neighbors, the corresponding edge is downweighted as the two cells are unlikely to be part of the same neighborhood.
 * In this manner, highly weighted edges will form within highly interconnected neighborhoods where many cells share the same neighbors.
 * This provides a more sophisticated definition of similarity between cells compared to a simpler (unweighted) nearest neighbor graph that just focuses on immediate proximity.
 *
 * A key parameter in the construction of the graph is the number of nearest neighbors $k$ to consider.
 * Larger values increase the connectivity of the graph and reduce the granularity of any subsequent community detection steps (see `scran::ClusterSNNGraph`) at the cost of speed.
 * The nearest neighbor search can be performed using either vantage point trees (exact) or with the Annoy algorithm (approximate) -
 * see the [**knncolle**](https://github.com/LTLA/knncolle) library for details.
 * 
 * For the edges, a variety of weighting schemes are possible:
 * 
 * - `RANKED` defines the weight between two nodes as $k - r/2$ where $r$ is the smallest sum of ranks for any shared neighboring node (Xu and Su, 2015).
 * For the purposes of this ranking, each node has a rank of zero in its own nearest-neighbor set. 
 * More shared neighbors, or shared neighbors that are close to both observations, will generally yield larger weights.
 * - `NUMBER` defines the weight between two nodes as the number of shared nearest neighbors between them. 
 * The weight can range from zero to $k + 1$, as the node itself is included in its own nearest-neighbor set. 
 * This is a simpler scheme that is also slightly faster but does not account for the ranking of neighbors within each set.
 * - `JACCARD` defines the weight between two nodes as the Jaccard index of their neighbor sets.
 * This weight can range from zero to 1, and is a monotonic transformation of the weight used by `NUMBER`.
 *
 * See the `ClusterSNNGraph` class to perform community detection on the graph returned by `run()`.
 *
 * @see
 * Xu C and Su Z (2015).
 * Identification of cell types from single-cell transcriptomes using a novel clustering method.
 * _Bioinformatics_ 31, 1974-80
 */
class BuildSNNGraph {
public:
    /** 
     * Choices for the edge weighting scheme during graph construction.
     */
    enum Scheme { RANKED, NUMBER, JACCARD };

    /**
     * @brief Default parameter settings.
     */
    struct Defaults {
        /**
         * See `set_neighbors()` for details.
         */
        static constexpr int neighbors = 10;

        /**
         * See `set_weighting_scheme()` for details.
         */
        static constexpr Scheme weighting_scheme = RANKED;

        /**
         * See `set_approximate()` for details.
         */
        static constexpr bool approximate = false;
    };
private:
    int num_neighbors = Defaults::neighbors;
    Scheme weight_scheme = Defaults::weighting_scheme;
    bool approximate = Defaults::approximate;

public:
    /** 
     * @param k Number of neighbors to use in the nearest neighbor search.
     *
     * @return A reference to this `BuildSNNGraph` object.
     */
    BuildSNNGraph& set_neighbors(int k = Defaults::neighbors) {
        num_neighbors = k;
        return *this;
    }

    /** 
     * @param a Whether to perform an approximate nearest neighbor search.
     *
     * @return A reference to this `BuildSNNGraph` object.
     */
    BuildSNNGraph& set_approximate(bool a = Defaults::approximate) {
        approximate = a;
        return *this;
    }

    /** 
     * @param w The edge weighting scheme to use.
     *
     * @return A reference to this `BuildSNNGraph` object.
     */
    BuildSNNGraph& set_weighting_scheme(Scheme w = Defaults::weighting_scheme) {
        weight_scheme = w;
        return *this;
    }

public:
    /**
     * A tuple containing the index of the first node, the index of the second node, and the weight of the edge between them.
     */
    typedef std::tuple<size_t, size_t, double> WeightedEdge;

    /**
     * @param ndims Number of dimensions.
     * @param ncells Number of cells.
     * @param mat Pointer to an array of expression values or a low-dimensional representation thereof.
     * Rows should be dimensions while columns should be cells.
     * Data should be stored in column-major format.
     *
     * @return A `deque` containing all the edges of the graph as `WeightedEdge`s.
     * In each `WeightedEdge`, the first index is guaranteed to be larger than the second index, so as to avoid unnecessary duplicates.
     */
    std::deque<WeightedEdge> run(size_t ndims, size_t ncells, const double* mat) const {
        std::unique_ptr<knncolle::Base<> > ptr;
        if (!approximate) {
            ptr.reset(new knncolle::VpTreeEuclidean<>(ndims, ncells, mat));
        } else {
            ptr.reset(new knncolle::AnnoyEuclidean<>(ndims, ncells, mat));
        }
        return run(ptr.get());
    }

    /**
     * @tparam Algorithm Any instance of a `knncolle::Base` subclass.
     *
     * @param search Pointer to a `knncolle::Base` instance to use for the nearest-neighbor search.
     *
     * @return A `deque` identical to that returned by the other `run()` methods.
     */
    template<class Algorithm>
    std::deque<WeightedEdge> run(const Algorithm* search) const {
        // Collecting neighbors.
#ifdef SCRAN_LOGGER
        SCRAN_LOGGER("scran::BuildSNNGraph", "Identifying nearest neighbors");
#endif
        size_t ncells = search->nobs();
        std::vector<std::vector<int> > indices(ncells);

        #pragma omp parallel for
        for (size_t i = 0; i < ncells; ++i) {
            auto neighbors = search->find_nearest_neighbors(i, num_neighbors);
            auto& current = indices[i];
            for (auto x : neighbors) {
                current.push_back(x.first);
            }
        }

        return run(indices);
    }

    /**
     * @param indices Vector of indices of the neighbors for each cell, sorted by increasing distance.
     *
     * @return A `deque` identical to that returned by the other `run()` methods.
     */
    std::deque<WeightedEdge> run(const std::vector<std::vector<int> >& indices) const {
        size_t ncells = indices.size();

        // Not parallel-frendly, so we don't construct this with the neighbor search
        std::vector<std::vector<std::pair<int, size_t> > > hosts(ncells);
        for (size_t i = 0; i < ncells; ++i) {
            hosts[i].push_back(std::make_pair(0, i)); // each point is its own 0-th nearest neighbor
            const auto& current = indices[i];
            int counter = 1;
            for (auto x : current) {
                hosts[x].push_back(std::make_pair(counter, i));
                ++counter;
            }
        }

        // Constructing the shared neighbor graph.
#ifdef SCRAN_LOGGER
        SCRAN_LOGGER("scran::BuildSNNGraph", "Computing shared neighbor weights");
#endif
        std::vector<size_t> current_added;
        current_added.reserve(ncells);
        std::vector<int> current_score(ncells);
        std::deque<WeightedEdge> store;

        for (size_t j = 0; j < ncells; ++j) {
            const auto& current_neighbors = indices[j];

            for (int i = 0; i <= current_neighbors.size(); ++i) {
                // First iteration treats 'j' as the zero-th neighbor.
                // Remaining iterations go through the neighbors of 'j'.
                const int cur_neighbor = (i==0 ? j : current_neighbors[i-1]);

                // Going through all observations 'h' for which 'cur_neighbor'
                // is a nearest neighbor, a.k.a., 'cur_neighbor' is a shared
                // neighbor of both 'h' and 'j'.
                for (auto& h : hosts[cur_neighbor]) {
                    const auto& othernode = h.second;

                    if (othernode < j) { // avoid duplicates from symmetry in the SNN calculations.
                        int& existing_other = current_score[othernode];
                        if (weight_scheme == RANKED) {
                            // Recording the lowest combined rank per neighbor.
                            int currank = h.first + i;
                            if (existing_other == 0) { 
                                existing_other = currank;
                                current_added.push_back(othernode);
                            } else if (existing_other > currank) {
                                existing_other = currank;
                            }
                        } else {
                            // Recording the number of shared neighbors.
                            if (existing_other==0) { 
                                current_added.push_back(othernode);
                            } 
                            ++existing_other;
                        }
                    }
                }
            }
           
            // Converting to edges.
            for (auto othernode : current_added) {
                int& otherscore=current_score[othernode];
                double finalscore;
                if (weight_scheme == RANKED) {
                    finalscore = static_cast<double>(current_neighbors.size()) - 0.5 * static_cast<double>(otherscore);
                } else {
                    finalscore = otherscore;
                    if (weight_scheme == JACCARD) {
                        finalscore = finalscore / (2 * (current_neighbors.size() + 1) - finalscore);
                    }
                }

                // Ensuring that an edge with a positive weight is always reported.
                store.push_back(WeightedEdge(j, othernode, std::max(finalscore, 1e-6)));

                // Resetting all those added to zero.
                otherscore=0;
            }
            current_added.clear();
        }

        return store;
    }
};

}

#endif
