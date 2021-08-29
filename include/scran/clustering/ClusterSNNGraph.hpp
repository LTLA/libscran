#ifndef SCRAN_SNNGRAPHCLUSTERING_HPP
#define SCRAN_SNNGRAPHCLUSTERING_HPP

#ifndef SCRAN_EXCLUDE_IGRAPH

#include "BuildSNNGraph.hpp"

#include <vector>
#include <algorithm>

#include "igraph.h"

/**
 * @file ClusterSNNGraph.hpp
 *
 * @brief Identify clusters of cells from a shared nearest-neighbor graph.
 */

namespace scran {

/**
 * @brief Cluster cells by community detection on a shared nearest-neighbor graph.
 *
 * We use the [**igraph**](https://igraph.org/) library to perform community detection on the graph constructed by `scran::BuildSNNGraph()`.
 * This yields a clustering on the cells that can be used for further characterization of subpopulations.
 */
class ClusterSNNGraph {
private:
    BuildSNNGraph builder;

public:
    /**
     * Set the number of neighbors to the default, see `BuildSNNGraph::set_neighbors()`.
     *
     * @return A reference to this `ClusterSNNGraph` object.
     */
    ClusterSNNGraph& set_neighbors() {
        builder.set_neighbors();
        return *this;
    }

    /**
     * @param k Number of neighbors, see `BuildSNNGraph::set_neighbors()`.
     *
     * @return A reference to this `ClusterSNNGraph` object.
     */
    ClusterSNNGraph& set_neighbors(int k) {
        builder.set_neighbors(k);
        return *this;
    }

    /**
     * Set the approximate algorithm flag to the default, see `BuildSNNGraph::set_approximate()`.
     *
     * @return A reference to this `ClusterSNNGraph` object.
     */
    ClusterSNNGraph& set_approximate() {
        builder.set_approximate();
        return *this;
    }

    /**
     * @param a Whether to use approximate nearest neighbor search, see `BuildSNNGraph::set_approximate()`.
     *
     * @return A reference to this `ClusterSNNGraph` object.
     */
    ClusterSNNGraph& set_approximate(bool a) {
        builder.set_approximate(a);
        return *this;
    }

    /**
     * Set the weighting scheme to the default, see `BuildSNNGraph::set_weighting_scheme()`.
     *
     * @return A reference to this `ClusterSNNGraph` object.
     */
    ClusterSNNGraph& set_weighting_scheme() {
        builder.set_weighting_scheme();
        return *this;
    }

    /**
     * @param s Weighting scheme to use, see `BuildSNNGraph::set_weighting_scheme()`.
     *
     * @return A reference to this `ClusterSNNGraph` object.
     */
    ClusterSNNGraph& set_weighting_scheme(BuildSNNGraph::Scheme s) {
        builder.set_weighting_scheme(s);
        return *this;
    }

public:
    /**
     * @brief Wrapper around the `igraph_t` class from **igraph**.
     * The objects therein will not be deep-copied when instances of this class are copied, so use with caution.
     */
    struct Graph {
        /**
         * Representation of an **igraph** graph.
         */
        igraph_t graph;

        /**
         * An **igraph** vector to be used to hold the weights.
         */
        igraph_vector_t weights;

        /**
         * A convenient method to release the memory allocated in `graph` and `weights`.
         */
        void destroy() {
            igraph_destroy(&graph);
            igraph_vector_destroy(&weights);
        }
    };

    /**
     * Build a shared nearest neighbor graph and convert it into an **igraph**-compatible form.
     *
     * @param ndims Number of dimensions.
     * @param ncells Number of cells.
     * @param mat Pointer to an array of expression values or a low-dimensional representation thereof.
     * Rows should be dimensions while columns should be cells.
     * Data should be stored in column-major format.
     *
     * @return A `Graph` object containing an **igraph** graph with weights.
     */
    Graph build(size_t ndims, size_t ncells, const double* mat) const {
        auto store = builder.run(ndims, ncells, mat);
        return build(ncells, store);
    }

    /**
     * Convert pre-built edges of a shared nearest neighbor graph into an **igraph**-compatible form.
     *
     * @param ncells Number of cells.
     * @param store A `deque` of edges, usually generated by a previous call to `BuildSNNGraph::run()`.
     *
     * @return A `Graph` object containing an **igraph** graph with weights.
     */
    Graph build(size_t ncells, const std::deque<BuildSNNGraph::WeightedEdge>& store) const {
        Graph output;
        igraph_vector_t edges;
        igraph_vector_init(&edges, store.size() * 2);

        igraph_vector_t& weights = output.weights;
        igraph_vector_init(&weights, store.size());

        size_t counter = 0;
        for (size_t i = 0; i < store.size(); ++i, counter += 2) { // not entirely sure it's safe to use std::copy here.
            const auto& edge = store[i];
            VECTOR(edges)[counter] = std::get<0>(edge);
            VECTOR(edges)[counter + 1] = std::get<1>(edge);
            VECTOR(weights)[i] = std::get<2>(edge);
        }

        igraph_create(&(output.graph), &edges, ncells, 0); 
        igraph_vector_destroy(&edges);

        return output;
    }

public:
    /**
     * @brief Result of the **igraph** multi-level community detection algorithm.
     *
     * A separate set of clustering results are reported for each level.
     * The level providing the highest modularity is also reported; the clustering at this level is usually a good default choice.
     */
    struct MultiLevelResult {
        /** 
         * Output status.
         * A value of zero indicates that the algorithm completed successfully.
         */
        int status;
        
        /**
         * The level that maximizes the modularity.
         * This can be used to index a particular result in `membership` and `modularity`.
         */
        size_t max;

        /**
         * Each vector contains the clustering result for a particular level.
         * Each vector is of length equal to the number of cells and contains 0-indexed cluster identities.
         */
        std::vector<std::vector<int> > membership;

        /**
         * Modularity scores at each level.
         * This is of the same length as `membership`.
         */
        std::vector<double> modularity;
    };

    /**
     * Run the multi-level community detection algorithm on a shared nearest-neighbor graph constructed from an expression matrix.
     * See [here](https://igraph.org/c/doc/igraph-Community.html#igraph_community_multilevel) for more details. 
     *
     * @param ndims Number of dimensions.
     * @param ncells Number of cells.
     * @param mat Pointer to an array of expression values or a low-dimensional representation thereof.
     * Rows should be dimensions while columns should be cells.
     * Data should be stored in column-major format.
     * @param resolution Resolution of the clustering, must be non-negative.
     * Lower values favor fewer, larger communities; higher values favor more, smaller communities.
     *
     * @return A `MultiLevelResult` object containing the clustering results for all cells.
     */
    MultiLevelResult run_multilevel(size_t ndims, size_t ncells, const double* mat, double resolution = 1) {
        auto graph_info = build(ndims, ncells, mat);
        auto output = run_multilevel(graph_info, resolution);
        graph_info.destroy();
        return output;
    }

    /**
     * Run the multi-level community detection algorithm on a pre-constructed shared nearest-neighbor graph as a `Graph` object.
     *
     * @param graph_info An existing `Graph` object, e.g., constructed by `build()`.
     * @param resolution Resolution of the clustering, see `run_multilevel()` for details.
     *
     * @return A `MultiLevelResult` object containing the clustering results for all cells.
     */
    MultiLevelResult run_multilevel(const Graph& graph_info, double resolution = 1) {
        igraph_vector_t membership, modularity;
        igraph_matrix_t memberships;

        igraph_vector_init(&modularity, 0);
        igraph_vector_init(&membership, 0);
        igraph_matrix_init(&memberships, 0, 0);

        // I just can't be bothered to do anything fancier here, so this is what we've got.
        igraph_rng_seed(igraph_rng_default(), 42);

        MultiLevelResult output;
        output.status = igraph_community_multilevel(&graph_info.graph, &graph_info.weights, resolution, &membership, &memberships, &modularity);

        if (!output.status) {
            output.max = igraph_vector_which_max(&modularity);

            size_t nmods = igraph_vector_size(&modularity);
            output.modularity.resize(nmods);
            for (size_t i = 0; i < nmods; ++i) {
                output.modularity[i] = VECTOR(modularity)[i];
            }

            size_t ncells = igraph_vcount(&graph_info.graph);
            size_t nlevels = igraph_matrix_nrow(&memberships);
            output.membership.resize(nlevels);
            
            for (size_t i = 0; i < nlevels; ++i) {
                auto& current = output.membership[i];
                current.resize(ncells);
                for (size_t j = 0; j < ncells; ++j) {
                    current[j] = MATRIX(memberships, i, j);
                }
            }
        }

        igraph_vector_destroy(&modularity);
        igraph_vector_destroy(&membership);
        igraph_matrix_destroy(&memberships);

        return output;
    }
};

}

#endif

#endif

