#ifndef SCRAN_SNNGRAPHCLUSTERING_HPP
#define SCRAN_SNNGRAPHCLUSTERING_HPP

#include "../utils/macros.hpp"

#include "BuildSnnGraph.hpp"

#include <vector>
#include <algorithm>

#include "igraph.h"
#include "igraph_utils.hpp"

/**
 * @file ClusterSnnGraph.hpp
 *
 * @brief Identify clusters of cells from a shared nearest-neighbor graph.
 */

namespace scran {

/**
 * @brief Multi-level clustering on a shared nearest-neighbor graph.
 *
 * This applies multi-level (i.e., "Louvain") clustering on a shared nearest neighbor graph.
 * See [here](https://igraph.org/c/doc/igraph-Community.html#igraph_community_multilevel) for more details on the multi-level algorithm. 
 */
class ClusterSnnGraphMultiLevel {
public:
    /**
     * @brief Default parameter settings.
     */
    struct Defaults {
        /**
         * See `set_resolution()` for more details.
         */
        static constexpr double resolution = 1;

        /**
         * See `set_seed()` for more details.
         */
        static constexpr int seed = 42;
    };

private:
    double resolution = Defaults::resolution;
    int seed = Defaults::seed;

public:
    /**
     * @param s Seed for the default **igraph** random number generator.
     * 
     * @return A reference to this `ClusterSnnGraphMultiLevel` object.
     */
    ClusterSnnGraphMultiLevel& set_seed(int s = Defaults::seed) {
        seed = s;
        return *this;
    }

    /**
     * @param r Resolution of the clustering, must be non-negative.
     * Lower values favor fewer, larger communities; higher values favor more, smaller communities.
     *
     * @return A reference to this `ClusterSnnGraphMultiLevel` object.
     */
    ClusterSnnGraphMultiLevel& set_resolution(double r = Defaults::resolution) {
        resolution = r;
        return *this;
    }

public:
    /**
     * @brief Result of the **igraph** multi-level community detection algorithm.
     *
     * Instances should be constructed using the `ClusterSnnGraphMultiLevel::run()` methods.
     * A separate set of clustering results are reported for each level.
     * The level providing the highest modularity is also reported; the clustering at this level is usually a good default choice.
     */
    struct Results {
        /** 
         * Output status.
         * A value of zero indicates that the algorithm completed successfully.
         */
        int status = 0;
        
        /**
         * The level that maximizes the modularity.
         * This can be used to index a particular result in `membership` and `modularity`.
         */
        size_t max = 0;

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
     * Run the multi-level community detection algorithm on a shared nearest-neighbor graph constructed from `knncolle::Base` object.
     *
     * @param store SNN graph built by `BuildSnnGraph::run()`.
     *
     * @return A `Results` object containing the clustering results for all cells.
     */
    Results run(const BuildSnnGraph::Results& store) const {
        return run(store.to_igraph(), store.weights.data());
    }

    /**
     * Run the multi-level community detection algorithm on a pre-constructed shared nearest-neighbor graph as a `Graph` object.
     *
     * @param graph An existing `igraph::Graph` object, typically built by `BuildSnnGraph::Results::to_igraph()`.
     * @param weights Pointer to an array of weights of length equal to the number of edges in `graph`. 
     *
     * @return A `Results` object containing the clustering results for all cells.
     */
    Results run(const igraph::Graph& graph, const igraph_real_t* weights) const {
        igraph::IntegerVector membership_holder;
        igraph::RealVector modularity_holder;
        igraph::IntegerMatrix memberships_holder;
        igraph::RNGScope rngs(seed);

        auto& modularity = modularity_holder.vector;
        auto& membership = membership_holder.vector;
        auto& memberships = memberships_holder.matrix;

        // No need to free this, as it's just a view.
        igraph_vector_t weight_view;
        igraph_vector_view(&weight_view, weights, igraph_ecount(graph.get_graph()));

        Results output;
        output.status = igraph_community_multilevel(graph.get_graph(), &weight_view, resolution, &membership, &memberships, &modularity);

        if (!output.status) {
            output.max = igraph_vector_which_max(&modularity);

            size_t nmods = igraph_vector_size(&modularity);
            output.modularity.resize(nmods);
            for (size_t i = 0; i < nmods; ++i) {
                output.modularity[i] = VECTOR(modularity)[i];
            }

            size_t ncells = igraph_vcount(graph.get_graph());
            size_t nlevels = igraph_matrix_int_nrow(&memberships);
            output.membership.resize(nlevels);
            
            for (size_t i = 0; i < nlevels; ++i) {
                auto& current = output.membership[i];
                current.resize(ncells);
                for (size_t j = 0; j < ncells; ++j) {
                    current[j] = MATRIX(memberships, i, j);
                }
            }
        }

        return output;
    }
};

/**
 * @brief Walktrap clustering on a shared nearest-neighbor graph.
 *
 * This applies Walktrap clustering on a shared nearest neighbor graph.
 * See [here](https://igraph.org/c/doc/igraph-Community.html#igraph_community_walktrap) for more details on the Walktrap algorithm. 
 */
class ClusterSnnGraphWalktrap {
public:
    /**
     * @brief Default parameter settings.
     */
    struct Defaults {
        /**
         * See `set_steps()` for more details.
         * The default is based on the example in the **igraph** documentation.
         */
        static constexpr int steps = 4;
    };

private:
    int steps = Defaults::steps;

public:
    /**
     * @param s Number of steps of the random walk.
     *
     * @return A reference to this `ClusterSnnGraphWalktrap` object.
     */
    ClusterSnnGraphWalktrap& set_steps(int s = Defaults::steps) {
        steps = s;
        return *this;
    }

public:
    /**
     * @brief Result of the **igraph** Walktrap community detection algorithm.
     * 
     * Instances should be constructed using the `ClusterSnnGraphWalktrap::run()` methods.
     */
    struct Results {
        /** 
         * Output status.
         * A value of zero indicates that the algorithm completed successfully.
         */
        int status = 0;
        
        /**
         * Vector of length equal to the number of cells, containing 0-indexed cluster identities.
         */
        std::vector<int> membership;

        /**
         * Vector of length equal to the number of merge steps, containing the identities of the two clusters being merged.
         * Note that cluster IDs here are not the same as those in `membership`.
         */
        std::vector<std::pair<int, int> > merges;

        /**
         * Vector of length equal to `merges` plus 1, containing the modularity score before and after each merge step.
         * The maximum value is the modularity corresponding to the clustering in `membership`.
         */
        std::vector<double> modularity;
    };

    /**
     * Run the Walktrap community detection algorithm on a shared nearest-neighbor graph constructed from `knncolle::Base` object.
     *
     * @param store SNN graph built by `BuildSnnGraph::run()`.
     *
     * @return A `Results` object containing the clustering results for all cells.
     */
    Results run(const BuildSnnGraph::Results& store) const {
        return run(store.to_igraph(), store.weights.data());
    }

    /**
     * Run the Walktrap community detection algorithm on a pre-constructed shared nearest-neighbor graph as a `Graph` object.
     *
     * @param graph An existing `igraph::Graph` object, typically built by `BuildSnnGraph::Results::to_igraph()`.
     * @param weights Pointer to an array of weights of length equal to the number of edges in `graph`. 
     *
     * @return A `Results` object containing the clustering results for all cells.
     */
    Results run(const igraph::Graph& graph, const igraph_real_t* weights) const {
        igraph::IntegerMatrix merges_holder;
        igraph::RealVector modularity_holder;
        igraph::IntegerVector membership_holder;

        auto& modularity = modularity_holder.vector;
        auto& membership = membership_holder.vector;
        auto& merges = merges_holder.matrix;

        // No need to free this, as it's just a view.
        igraph_vector_t weight_view;
        igraph_vector_view(&weight_view, weights, igraph_ecount(graph.get_graph()));

        Results output;
        output.status = igraph_community_walktrap(graph.get_graph(), &weight_view, steps, &merges, &modularity, &membership);

        if (!output.status) {
            size_t nmods = igraph_vector_size(&modularity);
            output.modularity.resize(nmods);
            for (size_t i = 0; i < nmods; ++i) {
                output.modularity[i] = VECTOR(modularity)[i];
            }

            size_t nmerges = igraph_matrix_int_nrow(&merges);
            output.merges.resize(nmerges);
            for (size_t i = 0; i < nmerges; ++i) {
                output.merges[i].first = MATRIX(merges, i, 0);
                output.merges[i].second = MATRIX(merges, i, 1);
            }

            size_t ncells = igraph_vcount(graph.get_graph());
            output.membership.resize(ncells);
            for (size_t i = 0; i < ncells; ++i) {
                output.membership[i] = VECTOR(membership)[i];
            }
        }

        return output;
    }
};

/**
 * @brief Leiden clustering on a shared nearest-neighbor graph.
 *
 * This applies Leiden clustering on a shared nearest neighbor graph.
 * See [here](https://igraph.org/c/doc/igraph-Community.html#igraph_community_leiden) for more details on the Leiden algorithm. 
 */
class ClusterSnnGraphLeiden {
public:
    /**
     * @brief Default parameter settings.
     */
    struct Defaults {
        /**
         * See `set_resolution()` for more details.
         * The default is based on `?cluster_leiden` in the **igraph** R package.
         */
        static constexpr double resolution = 1;

        /**
         * See `set_beta()` for more details.
         * The default is based on `?cluster_leiden` in the **igraph** R package.
         */
        static constexpr double beta = 0.01;

        /**
         * See `set_iterations()` for more details.
         * The default is based on `?cluster_leiden` in the **igraph** R package.
         */
        static constexpr int iterations = 2;

        /**
         * See `set_modularity()` for more details.
         * The default is based on `?cluster_leiden` in the **igraph** R package.
         */
        static constexpr bool modularity = false;

        /**
         * See `set_seed()` for more details.
         */
        static constexpr int seed = 42;
    };

private:
    double resolution = Defaults::resolution;
    double beta = Defaults::beta;
    int iterations = Defaults::iterations;
    bool modularity = Defaults::modularity;
    int seed = Defaults::seed;

public:
    /**
     * @param s Seed for the default **igraph** random number generator.
     * 
     * @return A reference to this `ClusterSnnGraphLeiden` object.
     */
    ClusterSnnGraphLeiden& set_seed(int s = Defaults::seed) {
        seed = s;
        return *this;
    }

     /**
     * @param r Resolution of the clustering.
     * Larger values result in more fine-grained communities.
     *
     * @return A reference to this `ClusterSnnGraphLeiden` object.
     */
    ClusterSnnGraphLeiden& set_resolution(double r = Defaults::resolution) {
        resolution = r;
        return *this;
    }

    /**
     * @param b Level of randomness used during refinement.
     *
     * @return A reference to this `ClusterSnnGraphLeiden` object.
     */
    ClusterSnnGraphLeiden& set_beta(double b = Defaults::beta) {
        beta = b;
        return *this;
    }

    /**
     * @param i Number of iterations of the Leiden algorithm.
     * More iterations can improve separation at the cost of computational time.
     *
     * @return A reference to this `ClusterSnnGraphLeiden` object.
     */
    ClusterSnnGraphLeiden& set_iterations(int i = Defaults::iterations) {
        iterations = i;
        return *this;
    }

    /**
     * @param m Whether to optimize the modularity instead of the Constant Potts Model.
     *
     * The modularity is closely related to the Constant Potts Model, but the magnitude of the resolution is different.
     *
     * @return A reference to this `ClusterSnnGraphLeiden` object.
     */
    ClusterSnnGraphLeiden& set_modularity(bool m = Defaults::modularity) {
        modularity = m;
        return *this;
    }

public:
    /**
     * @brief Result of the **igraph** leiden community detection algorithm.
     *
     * Instances should be constructed using the `ClusterSnnGraphLeiden::run()` methods.
     */
    struct Results {
        /** 
         * Output status.
         * A value of zero indicates that the algorithm completed successfully.
         */
        int status = 0;
        
        /**
         * Vector of length equal to the number of cells, containing 0-indexed cluster identities.
         */
        std::vector<int> membership;

        /**
         * Quality of the clustering, closely related to the modularity.
         */
        double quality = 0;
    };

    /**
     * Run the Leiden community detection algorithm on a shared nearest-neighbor graph constructed from `knncolle::Base` object.
     *
     * @param store SNN graph built by `BuildSnnGraph::run()`.
     *
     * @return A `Results` object containing the clustering results for all cells.
     */
    Results run(const BuildSnnGraph::Results& store) const {
        return run(store.to_igraph(), store.weights.data());
    }

    /**
     * Run the Leiden community detection algorithm on a pre-constructed shared nearest-neighbor graph as a `Graph` object.
     *
     * @param graph An existing `igraph::Graph` object, typically built by `BuildSnnGraph::Results::to_igraph()`.
     * @param weights Pointer to an array of weights of length equal to the number of edges in `graph`. 
     *
     * @return A `Results` object containing the clustering results for all cells.
     */
    Results run(const igraph::Graph& graph, const igraph_real_t* weights) const {
        igraph::IntegerVector membership_holder;
        auto& membership = membership_holder.vector;
        igraph_integer_t nb_clusters;
        igraph_real_t quality;

        igraph::RNGScope rngs(seed);
        Results output;

        // No need to free this, as it's just a view.
        igraph_vector_t weight_view;
        size_t nedges = igraph_ecount(graph.get_graph());
        igraph_vector_view(&weight_view, weights, nedges);

        if (!modularity) {
            output.status = igraph_community_leiden(graph.get_graph(), &weight_view, NULL, resolution, beta, false, iterations, &membership, &nb_clusters, &quality);
        } else {
            // More-or-less translated from igraph::cluster_leiden in the R package,
            // but with the iterations moved into igraph_community_leiden itself.
            igraph::RealVector strength_holder(igraph_vcount(graph.get_graph()));
            auto& strength = strength_holder.vector;
            igraph_strength(graph.get_graph(), &strength, igraph_vss_all(), IGRAPH_ALL, 1, &weight_view);

            double total_weights = std::accumulate(weights, weights + nedges, 0.0);
            double mod_resolution = resolution / total_weights;

            output.status = igraph_community_leiden(graph.get_graph(), &weight_view, &strength, mod_resolution, beta, false, iterations, &membership, &nb_clusters, &quality);
        }

        if (!output.status) {
            size_t ncells = igraph_vcount(graph.get_graph());
            output.membership.resize(ncells);
            for (size_t i = 0; i < ncells; ++i) {
                output.membership[i] = VECTOR(membership)[i];
            }
            output.quality = quality;
        }

        return output;
    }
};

}

#endif

