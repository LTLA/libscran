#ifndef SCRAN_BUILDSNNGRAPH_HPP
#define SCRAN_BUILDSNNGRAPH_HPP

#include "../utils/macros.hpp"

#if __has_include("igraph.h")
#include "igraph.h"
#include "igraph_utils.hpp"
#endif

#include <vector>
#include <algorithm>
#include <memory>

#include "knncolle/knncolle.hpp"

/**
 * @file BuildSnnGraph.hpp
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
 * Larger values increase the connectivity of the graph and reduce the granularity of any subsequent community detection steps (see `scran::ClusterSnnGraph`) at the cost of speed.
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
class BuildSnnGraph {
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

        /**
         * See `set_num_threads()`.
         */
        static constexpr int num_threads = 1;
    };
private:
    int num_neighbors = Defaults::neighbors;
    Scheme weight_scheme = Defaults::weighting_scheme;
    bool approximate = Defaults::approximate;
    int nthreads = Defaults::num_threads;

public:
    /** 
     * @param k Number of neighbors to use in the nearest neighbor search.
     *
     * @return A reference to this `BuildSnnGraph` object.
     */
    BuildSnnGraph& set_neighbors(int k = Defaults::neighbors) {
        num_neighbors = k;
        return *this;
    }

    /** 
     * @param a Whether to perform an approximate nearest neighbor search.
     *
     * @return A reference to this `BuildSnnGraph` object.
     */
    BuildSnnGraph& set_approximate(bool a = Defaults::approximate) {
        approximate = a;
        return *this;
    }

    /** 
     * @param w The edge weighting scheme to use.
     *
     * @return A reference to this `BuildSnnGraph` object.
     */
    BuildSnnGraph& set_weighting_scheme(Scheme w = Defaults::weighting_scheme) {
        weight_scheme = w;
        return *this;
    }

    /**
     * @param n Number of threads to use. 
     * @return A reference to this `BuildSnnGraph` object.
     */
    BuildSnnGraph& set_num_threads(int n = Defaults::num_threads) {
        nthreads = n;
        return *this;
    }

public:
    /**
     * @brief Results of SNN graph construction.
     *
     * Edges and weights are stored using **igraph** data types in preparation for community detection.
     */
    struct Results {
        /**
         * Number of cells in the dataset.
         */
        size_t ncells;

        /**
         * Vector of paired indices defining the edges between cells.
         * The number of edges is half the length of `edges`, where `edges[2*i]` and `edges[2*i+1]` define the vertices for edge `i`.
         */
#if __has_include("igraph.h")
        std::vector<igraph_integer_t> edges;
#else
        std::vector<int> edges;
#endif

        /**
         * Vector of weights for the edges.
         * This is of length equal to the number of edges, where each `weights[i]` corresponds to an edge `i` in `edges`. 
         */
#if __has_include("igraph.h")
        std::vector<igraph_real_t> weights;
#else
        std::vector<double> weights;
#endif

#if __has_include("igraph.h")
        /**
         * Create an **igraph** graph object from the edges.
         * Note that weights are not included in the output object and should be supplied separately to relevant functions.
         *
         * @return An undirected graph created from `edges`.
         */
        igraph::Graph to_igraph() const {
            igraph_vector_int_t edge_view;
            igraph_vector_int_view(&edge_view, edges.data(), edges.size());
            return igraph::Graph(&edge_view, ncells, /* directed = */ false);
        }
#endif
    };

    /**
     * @param ndims Number of dimensions.
     * @param ncells Number of cells.
     * @param mat Pointer to an array of expression values or a low-dimensional representation thereof.
     * Rows should be dimensions while columns should be cells.
     * Data should be stored in column-major format.
     *
     * @return The edges and weights of the constructed SNN graph.
     */
    Results run(size_t ndims, size_t ncells, const double* mat) const {
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
     * @return The edges and weights of the constructed SNN graph.
     */
    template<class Algorithm>
    Results run(const Algorithm* search) const {
        // Collecting neighbors.
        size_t ncells = search->nobs();
        std::vector<std::vector<int> > indices(ncells);

#ifndef SCRAN_CUSTOM_PARALLEL
        #pragma omp parallel for num_threads(nthreads)
        for (size_t i = 0; i < ncells; ++i) {
#else
        SCRAN_CUSTOM_PARALLEL([&](size_t, size_t start, size_t length) -> void {
        for (size_t i = start, end = start + length; i < end; ++i) {
#endif
            auto neighbors = search->find_nearest_neighbors(i, num_neighbors);
            auto& current = indices[i];
            for (auto x : neighbors) {
                current.push_back(x.first);
            }
        }
#ifdef SCRAN_CUSTOM_PARALLEL
        }, ncells, nthreads);
#endif

        return run(indices);
    }

    /**
     * @tparam Indices Vector of integer indices.
     *
     * @param indices Vector of indices of the neighbors for each cell, sorted by increasing distance.
     *
     * @return The edges and weights of the constructed SNN graph.
     */
    template<class Indices>
    Results run(const std::vector<Indices>& indices) const {
        size_t ncells = indices.size();

        // Not parallel-frendly, so we don't construct this with the neighbor search
        std::vector<std::vector<std::pair<int, int> > > hosts(ncells);
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
        std::vector<std::vector<int> > edge_stores(ncells);
        std::vector<std::vector<double> > weight_stores(ncells);

#ifndef SCRAN_CUSTOM_PARALLEL
        #pragma omp parallel num_threads(nthreads)
        {
#else
        SCRAN_CUSTOM_PARALLEL([&](size_t, size_t start, size_t length) -> void {
#endif

            std::vector<int> current_score(ncells);
            std::vector<int> current_added;
            current_added.reserve(ncells);

#ifndef SCRAN_CUSTOM_PARALLEL
            #pragma omp for
            for (size_t j = 0; j < ncells; ++j) {
#else
            for (size_t j = start, end = start + length; j < end; ++j) {
#endif

                const auto& current_neighbors = indices[j];
                for (int i = 0; i <= current_neighbors.size(); ++i) {
                    // First iteration treats 'j' as the zero-th neighbor.
                    // Remaining iterations go through the neighbors of 'j'.
                    const int cur_neighbor = (i==0 ? j : current_neighbors[i-1]);

                    // Going through all observations 'h' for which 'cur_neighbor'
                    // is a nearest neighbor, a.k.a., 'cur_neighbor' is a shared
                    // neighbor of both 'h' and 'j'.
                    for (const auto& h : hosts[cur_neighbor]) {
                        auto othernode = h.second;

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
                auto& current_edges = edge_stores[j];
                current_edges.reserve(current_added.size() * 2);
                auto& current_weights = weight_stores[j];
                current_weights.reserve(current_added.size() * 2);

                for (auto othernode : current_added) {
                    int& otherscore = current_score[othernode];
                    double finalscore;
                    if (weight_scheme == RANKED) {
                        finalscore = static_cast<double>(current_neighbors.size()) - 0.5 * static_cast<double>(otherscore);
                    } else {
                        finalscore = otherscore;
                        if (weight_scheme == JACCARD) {
                            finalscore = finalscore / (2 * (current_neighbors.size() + 1) - finalscore);
                        }
                    }

                    current_edges.push_back(j);
                    current_edges.push_back(othernode);
                    current_weights.push_back(std::max(finalscore, 1e-6)); // Ensuring that an edge with a positive weight is always reported.

                    // Resetting all those added to zero.
                    otherscore = 0;
                }
                current_added.clear();

#ifndef SCRAN_CUSTOM_PARALLEL
            }
        } 
#else
            }
        }, ncells, nthreads);
#endif

        // Collating the total number of edges.
        size_t nedges = 0;
        for (const auto& w : weight_stores) {
            nedges += w.size();
        }

        Results output;
        output.ncells = ncells;

        output.weights.reserve(nedges);
        for (const auto& w : weight_stores) {
            output.weights.insert(output.weights.end(), w.begin(), w.end());
        }
        weight_stores.clear();
        weight_stores.shrink_to_fit(); // forcibly release memory so that we have some more space for edges.

        output.edges.reserve(nedges * 2);
        for (const auto& e : edge_stores) {
            output.edges.insert(output.edges.end(), e.begin(), e.end());
        }

        return output;
    }
};

}

#endif
