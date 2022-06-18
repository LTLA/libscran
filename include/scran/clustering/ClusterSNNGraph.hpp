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
 * @brief Virtual class for clustering on a shared nearest-neighbor graph.
 *
 * This is a virtual class that takes the graph constructed by `scran::BuildSNNGraph()` and prepares it for use in the [**igraph**](https://igraph.org/) library.
 * Concrete subclasses should apply specific community detection algorithms to obtain a clustering on the cells. 
 */
class ClusterSNNGraph {
protected:
    /**
     * @cond
     */
    BuildSNNGraph builder;

    struct IgraphRNGScope {
        IgraphRNGScope(int seed) : previous(igraph_rng_default()) {
            if (igraph_rng_init(&rng, &igraph_rngtype_mt19937)) {
                throw std::runtime_error("failed to initialize an instance of igraph's RNG");
            }

            if (igraph_rng_seed(&rng, seed)) {
                igraph_rng_destroy(&rng);
                throw std::runtime_error("failed to set the seed on igraph's RNG");
            }

            igraph_rng_set_default(&rng);
        }

        // Just deleting the methods here, because the RNGScope
        // is strictly internal and we don't do any of these.
        IgraphRNGScope(const IgraphRNGScope& other) = delete;
        IgraphRNGScope& operator=(const IgraphRNGScope& other) = delete;
        IgraphRNGScope(IgraphRNGScope&& other) = delete;
        IgraphRNGScope& operator=(IgraphRNGScope&& other) = delete;

        ~IgraphRNGScope() {
            if (active) {
                igraph_rng_set_default(previous);
                igraph_rng_destroy(&rng);
            }
        }

        igraph_rng_t* previous;
        igraph_rng_t rng;
        bool active = true;
    };

    struct IgraphVector {
    private:
        static void try_init(igraph_vector_t& vector, size_t size) {
            if (igraph_vector_init(&vector, size)) {
                throw std::runtime_error("failed to initialize igraph vector of size " + std::to_string(size));
            }
        }

        static void try_copy(igraph_vector_t& dest, const igraph_vector_t& source, bool source_active) {
            if (source_active) {
                auto size = igraph_vector_size(&source);
                try_init(dest, size);
                if (igraph_vector_copy(&dest, &source)) {
                    throw std::runtime_error("failed to copy igraph vector of size " + std::to_string(size));
                }
            }
        }

        static void try_destroy(igraph_vector_t& vector, bool active) {
            if (active) {
                igraph_vector_destroy(&vector);
            }
        }

    public:
        IgraphVector(size_t size = 0) {
            try_init(vector, size);
        }

        IgraphVector(const IgraphVector& other) : active(other.active) {
            try_copy(vector, other.vector, other.active);
        }
        
        IgraphVector& operator=(const IgraphVector& other) {
            if (this != &other) {
                try_destroy(vector, active);
                try_copy(vector, other.vector, other.active);
                active = other.active;
            }
            return *this;
        }

        // See https://docs.microsoft.com/en-us/cpp/cpp/move-constructors-and-move-assignment-operators-cpp?view=msvc-170
        IgraphVector(IgraphVector&& other) : vector(std::move(other.vector)), active(other.active) {
            other.active = false;
        }

        IgraphVector& operator=(IgraphVector&& other) {
            if (this != &other) {
                try_destroy(vector, active);
                vector = std::move(other.vector);
                active = other.active;
                other.active = false;
            }
            return *this;
        }

        ~IgraphVector() {
            try_destroy(vector, active);
        }

        igraph_vector_t vector;
        bool active = true;
    };

    struct IgraphMatrix {
        IgraphMatrix(size_t nrows = 0, size_t ncols = 0) {
            if (igraph_matrix_init(&matrix, nrows, ncols)) {
                throw std::runtime_error("failed to initialize igraph " + std::to_string(nrows) + "x" + std::to_string(ncols) + " matrix");
            }
        }

        // Just deleting the methods here, because the Matrix
        // is strictly internal and we don't do any of these.
        IgraphMatrix(const IgraphMatrix& other) = delete;
        IgraphMatrix& operator=(const IgraphMatrix& other) = delete;
        IgraphMatrix(IgraphMatrix&& other) = delete;
        IgraphMatrix& operator=(IgraphMatrix&& other) = delete;

        ~IgraphMatrix() {
            if (active) {
                igraph_matrix_destroy(&matrix);
            }
        }

        igraph_matrix_t matrix;
        bool active = true;
    };

    struct IgraphGraph {
    private:
        static void try_copy(igraph_t& dest, const igraph_t& source, bool source_active) {
            if (source_active) {
                if (igraph_copy(&dest, &source)) {
                    throw std::runtime_error("failed to copy igraph's graph");
                }
            }
        }

        static void try_destroy(igraph_t& graph, bool active) {
            if (active) {
                igraph_destroy(&graph);
            }
        }

    public:
        IgraphGraph(const IgraphVector& edges, size_t nvertices, bool directed) { 
            if (igraph_create(&graph, &edges.vector, nvertices, directed)) {
                throw std::runtime_error("failed to initialize igraph's graph object"); 
            }
        }

        IgraphGraph(const IgraphGraph& other) : active(other.active) {
            try_copy(graph, other.graph, other.active);
        }

        IgraphGraph& operator=(const IgraphGraph& other) {
            if (this != &other) {
                try_destroy(graph, active);
                try_copy(graph, other.graph, other.active);
                active = other.active;
            }
            return *this;
        }

        IgraphGraph(IgraphGraph&& other) : graph(std::move(other.graph)), active(other.active) {
            other.active = false;
        }

        IgraphGraph& operator=(IgraphGraph&& other) {
            if (this != &other) {
                try_destroy(graph, active);
                graph = std::move(other.graph);
                active = other.active;
                other.active = false;
            }
            return *this;
        }

        ~IgraphGraph() {
            try_destroy(graph, active);
        }

        igraph_t graph;
        bool active = true;
    };
    /**
     * @endcond
     */

public:
    /**
     * @brief Wrapper around the `igraph_t` class from **igraph**.
     *
     * The objects therein will not be deep-copied when instances of this class are copied, so use with caution.
     */
    struct Graph {
        /**
         * @cond
         */
    private:
        IgraphGraph graph;

        IgraphVector weights;
        /**
         * @endcond
         */
    public:
        /**
         * @cond
         */
        Graph(IgraphGraph g, IgraphVector w) : graph(std::move(g)), weights(std::move(w)) {}
        /**
         * @endcond
         */

        /**
         * @name Get the graph.
         *
         * @return Pointer to an **igraph** graph.
         * Nodes are cells with edges being formed between its nearest neighbors.
         *
         * Users should not pass this pointer to `igraph_destroy`; the `Graph` destructor will handle the freeing.
         */
        //@{
        /** 
         * Non-`const` overload.
         */
        igraph_t* get_graph() {
            return &graph.graph;
        }

        /** 
         * `const` overload.
         */
        const igraph_t* get_graph() const {
            return &graph.graph;
        }
        //@}

        /**
         * @name Get the weights.
         *  
         * @return Pointer to an **igraph** vector containing the weights for each edge in the graph.
         *
         * Users should not pass this pointer to `igraph_vector_destroy`; the `Graph` destructor will handle the freeing.
         */
        //@{
        /** 
         * Non-`const` overload.
         */
         igraph_vector_t* get_weights() {
            return &weights.vector;
        }

        /** 
         * `const` overload.
         */
         const igraph_vector_t* get_weights() const {
            return &weights.vector;
        }
        //@}
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
     * Build a shared nearest neighbor graph from an existing `knncolle::Base` object.
     *
     * @tparam Algorithm Any instance of a `knncolle::Base` subclass.
     *
     * @param search Pointer to a `knncolle::Base` instance to use for the nearest-neighbor search.
     *
     * @return A `Graph` object containing an **igraph** graph with weights.
     */
    template<class Algorithm>
    Graph build(const Algorithm* search) const {
        auto store = builder.run(search);
        return build(search->nobs(), store);
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
        IgraphVector edge_holder(store.size() * 2);
        auto& edges = edge_holder.vector;

        IgraphVector weight_holder(store.size());
        auto& weights = weight_holder.vector;

        size_t counter = 0;
        for (size_t i = 0, end = store.size(); i < end; ++i, counter += 2) { 
            const auto& edge = store[i];
            VECTOR(edges)[counter] = std::get<0>(edge);
            VECTOR(edges)[counter + 1] = std::get<1>(edge);
            VECTOR(weights)[i] = std::get<2>(edge);
        }

        IgraphGraph graph(edge_holder, ncells, /* directed = */ false);
        return Graph(std::move(graph), std::move(weight_holder));
    }
};

/**
 * @brief Multi-level clustering on a shared nearest-neighbor graph.
 *
 * This applies multi-level (i.e., "Louvain") clustering on a shared nearest neighbor graph.
 * See [here](https://igraph.org/c/doc/igraph-Community.html#igraph_community_multilevel) for more details on the multi-level algorithm. 
 */
class ClusterSNNGraphMultiLevel : public ClusterSNNGraph {
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
     * @param k Number of neighbors, see `BuildSNNGraph::set_neighbors()`.
     *
     * @return A reference to this `ClusterSNNGraphMultiLevel` object.
     */
    ClusterSNNGraphMultiLevel& set_neighbors(int k = BuildSNNGraph::Defaults::neighbors) {
        builder.set_neighbors(k);
        return *this;
    }

    /**
     * @param a Whether to use a approximate nearest neighbor search, see `BuildSNNGraph::set_approximate()`.
     *
     * @return A reference to this `ClusterSNNGraph` object.
     */
    ClusterSNNGraphMultiLevel& set_approximate(bool a = BuildSNNGraph::Defaults::approximate) {
        builder.set_approximate(a);
        return *this;
    }

    /**
     * @param s Weighting scheme to use, see `BuildSNNGraph::set_weighting_scheme()`.
     *
     * @return A reference to this `ClusterSNNGraphMultiLevel` object.
     */
    ClusterSNNGraphMultiLevel& set_weighting_scheme(BuildSNNGraph::Scheme s = BuildSNNGraph::Defaults::weighting_scheme) {
        builder.set_weighting_scheme(s);
        return *this;
    }

    /**
     * @param s Seed for the default **igraph** random number generator.
     * 
     * @return A reference to this `ClusterSNNGraphMultiLevel` object.
     */
    ClusterSNNGraphMultiLevel& set_seed(int s = Defaults::seed) {
        seed = s;
        return *this;
    }

    /**
     * @param r Resolution of the clustering, must be non-negative.
     * Lower values favor fewer, larger communities; higher values favor more, smaller communities.
     *
     * @return A reference to this `ClusterSNNGraphMultiLevel` object.
     */
    ClusterSNNGraphMultiLevel& set_resolution(double r = Defaults::resolution) {
        resolution = r;
        return *this;
    }

public:
    /**
     * @brief Result of the **igraph** multi-level community detection algorithm.
     *
     * Instances should be constructed using the `ClusterSNNGraphMultiLevel::run()` methods.
     * A separate set of clustering results are reported for each level.
     * The level providing the highest modularity is also reported; the clustering at this level is usually a good default choice.
     */
    struct Results {
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
     *
     * @param ndims Number of dimensions.
     * @param ncells Number of cells.
     * @param mat Pointer to an array of expression values or a low-dimensional representation thereof.
     * Rows should be dimensions while columns should be cells.
     * Data should be stored in column-major format.
     *
     * @return A `Results` object containing the clustering results for all cells.
     */
    Results run(size_t ndims, size_t ncells, const double* mat) {
        auto graph_info = build(ndims, ncells, mat);
        auto output = run(graph_info);
        return output;
    }

    /**
     * Run the multi-level community detection algorithm on a shared nearest-neighbor graph constructed from `knncolle::Base` object.
     *
     * @tparam Algorithm Any instance of a `knncolle::Base` subclass.
     *
     * @param search Pointer to a `knncolle::Base` instance to use for the nearest-neighbor search.
     *
     * @return A `Results` object containing the clustering results for all cells.
     */
    template<class Algorithm>
    Results run(const Algorithm* search) {
        auto graph_info = build(search);
        auto output = run(graph_info);
        return output;
    }

    /**
     * Run the multi-level community detection algorithm on a shared nearest-neighbor graph constructed from `knncolle::Base` object.
     *
     * @param ncells Number of cells.
     * @param store A `deque` of edges, usually generated by a previous call to `BuildSNNGraph::run()`.
     *
     * @return A `Results` object containing the clustering results for all cells.
     */
    Results run(size_t ncells, const std::deque<BuildSNNGraph::WeightedEdge>& store) {
        auto graph_info = build(ncells, store);
        auto output = run(graph_info);
        return output;
    }

    /**
     * Run the multi-level community detection algorithm on a pre-constructed shared nearest-neighbor graph as a `Graph` object.
     *
     * @param graph_info An existing `Graph` object, e.g., constructed by `build()`.
     *
     * @return A `Results` object containing the clustering results for all cells.
     */
    Results run(const Graph& graph_info) {
        IgraphVector membership_holder, modularity_holder;
        IgraphMatrix memberships_holder;
        IgraphRNGScope rngs(seed);

        auto& modularity = modularity_holder.vector;
        auto& membership = membership_holder.vector;
        auto& memberships = memberships_holder.matrix;

        Results output;
        output.status = igraph_community_multilevel(graph_info.get_graph(), graph_info.get_weights(), resolution, &membership, &memberships, &modularity);

        if (!output.status) {
            output.max = igraph_vector_which_max(&modularity);

            size_t nmods = igraph_vector_size(&modularity);
            output.modularity.resize(nmods);
            for (size_t i = 0; i < nmods; ++i) {
                output.modularity[i] = VECTOR(modularity)[i];
            }

            size_t ncells = igraph_vcount(graph_info.get_graph());
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

        return output;
    }
};

/**
 * @brief Walktrap clustering on a shared nearest-neighbor graph.
 *
 * This applies Walktrap clustering on a shared nearest neighbor graph.
 * See [here](https://igraph.org/c/doc/igraph-Community.html#igraph_community_walktrap) for more details on the Walktrap algorithm. 
 */
class ClusterSNNGraphWalktrap : public ClusterSNNGraph {
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
     * @param k Number of neighbors, see `BuildSNNGraph::set_neighbors()`.
     *
     * @return A reference to this `ClusterSNNGraphWalktrap` object.
     */
    ClusterSNNGraphWalktrap& set_neighbors(int k = BuildSNNGraph::Defaults::neighbors) {
        builder.set_neighbors(k);
        return *this;
    }

    /**
     * @param a Whether to use a approximate nearest neighbor search, see `BuildSNNGraph::set_approximate()`.
     *
     * @return A reference to this `ClusterSNNGraph` object.
     */
    ClusterSNNGraphWalktrap& set_approximate(bool a = BuildSNNGraph::Defaults::approximate) {
        builder.set_approximate(a);
        return *this;
    }

    /**
     * @param s Weighting scheme to use, see `BuildSNNGraph::set_weighting_scheme()`.
     *
     * @return A reference to this `ClusterSNNGraphWalktrap` object.
     */
    ClusterSNNGraphWalktrap& set_weighting_scheme(BuildSNNGraph::Scheme s = BuildSNNGraph::Defaults::weighting_scheme) {
        builder.set_weighting_scheme(s);
        return *this;
    }

    /**
     * @param s Number of steps of the random walk.
     *
     * @return A reference to this `ClusterSNNGraphWalktrap` object.
     */
    ClusterSNNGraphWalktrap& set_steps(int s = Defaults::steps) {
        steps = s;
        return *this;
    }

public:
    /**
     * @brief Result of the **igraph** Walktrap community detection algorithm.
     * 
     * Instances should be constructed using the `ClusterSNNGraphWalktrap::run()` methods.
     */
    struct Results {
        /** 
         * Output status.
         * A value of zero indicates that the algorithm completed successfully.
         */
        int status;
        
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
     * Run the Walktrap community detection algorithm on a shared nearest-neighbor graph constructed from an expression matrix.
     *
     * @param ndims Number of dimensions.
     * @param ncells Number of cells.
     * @param mat Pointer to an array of expression values or a low-dimensional representation thereof.
     * Rows should be dimensions while columns should be cells.
     * Data should be stored in column-major format.
     *
     * @return A `Results` object containing the clustering results for all cells.
     */
    Results run(size_t ndims, size_t ncells, const double* mat) {
        auto graph_info = build(ndims, ncells, mat);
        auto output = run(graph_info);
        return output;
    }

    /**
     * Run the Walktrap community detection algorithm on a shared nearest-neighbor graph constructed from `knncolle::Base` object.
     *
     * @tparam Algorithm Any instance of a `knncolle::Base` subclass.
     *
     * @param search Pointer to a `knncolle::Base` instance to use for the nearest-neighbor search.
     *
     * @return A `Results` object containing the clustering results for all cells.
     */
    template<class Algorithm>
    Results run(const Algorithm* search) {
        auto graph_info = build(search);
        auto output = run(graph_info);
        return output;
    }

    /**
     * Run the Walktrap community detection algorithm on a shared nearest-neighbor graph constructed from `knncolle::Base` object.
     *
     * @param ncells Number of cells.
     * @param store A `deque` of edges, usually generated by a previous call to `BuildSNNGraph::run()`.
     *
     * @return A `Results` object containing the clustering results for all cells.
     */
    Results run(size_t ncells, const std::deque<BuildSNNGraph::WeightedEdge>& store) {
        auto graph_info = build(ncells, store);
        auto output = run(graph_info);
        return output;
    }

    /**
     * Run the Walktrap community detection algorithm on a pre-constructed shared nearest-neighbor graph as a `Graph` object.
     *
     * @param graph_info An existing `Graph` object, e.g., constructed by `build()`.
     *
     * @return A `Results` object containing the clustering results for all cells.
     */
    Results run(const Graph& graph_info) {
        IgraphMatrix merges_holder;
        IgraphVector modularity_holder, membership_holder;

        auto& modularity = modularity_holder.vector;
        auto& membership = membership_holder.vector;
        auto& merges = merges_holder.matrix;

        Results output;
        output.status = igraph_community_walktrap(graph_info.get_graph(), graph_info.get_weights(), steps, &merges, &modularity, &membership);

        if (!output.status) {
            size_t nmods = igraph_vector_size(&modularity);
            output.modularity.resize(nmods);
            for (size_t i = 0; i < nmods; ++i) {
                output.modularity[i] = VECTOR(modularity)[i];
            }

            size_t nmerges = igraph_matrix_nrow(&merges);
            output.merges.resize(nmerges);
            for (size_t i = 0; i < nmerges; ++i) {
                output.merges[i].first = MATRIX(merges, i, 0);
                output.merges[i].second = MATRIX(merges, i, 1);
            }

            size_t ncells = igraph_vcount(graph_info.get_graph());
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
class ClusterSNNGraphLeiden : public ClusterSNNGraph {
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
     * @param k Number of neighbors, see `BuildSNNGraph::set_neighbors()`.
     *
     * @return A reference to this `ClusterSNNGraphLeiden` object.
     */
    ClusterSNNGraphLeiden& set_neighbors(int k = BuildSNNGraph::Defaults::neighbors) {
        builder.set_neighbors(k);
        return *this;
    }

    /**
     * @param a Whether to use a approximate nearest neighbor search, see `BuildSNNGraph::set_approximate()`.
     *
     * @return A reference to this `ClusterSNNGraph` object.
     */
    ClusterSNNGraphLeiden& set_approximate(bool a = BuildSNNGraph::Defaults::approximate) {
        builder.set_approximate(a);
        return *this;
    }

    /**
     * @param s Weighting scheme to use, see `BuildSNNGraph::set_weighting_scheme()`.
     *
     * @return A reference to this `ClusterSNNGraphLeiden` object.
     */
    ClusterSNNGraphLeiden& set_weighting_scheme(BuildSNNGraph::Scheme s = BuildSNNGraph::Defaults::weighting_scheme) {
        builder.set_weighting_scheme(s);
        return *this;
    }

    /**
     * @param s Seed for the default **igraph** random number generator.
     * 
     * @return A reference to this `ClusterSNNGraphLeiden` object.
     */
    ClusterSNNGraphLeiden& set_seed(int s = Defaults::seed) {
        seed = s;
        return *this;
    }

     /**
     * @param r Resolution of the clustering.
     * Larger values result in more fine-grained communities.
     *
     * @return A reference to this `ClusterSNNGraphLeiden` object.
     */
    ClusterSNNGraphLeiden& set_resolution(double r = Defaults::resolution) {
        resolution = r;
        return *this;
    }

    /**
     * @param b Level of randomness used during refinement.
     *
     * @return A reference to this `ClusterSNNGraphLeiden` object.
     */
    ClusterSNNGraphLeiden& set_beta(double b = Defaults::beta) {
        beta = b;
        return *this;
    }

    /**
     * @param i Number of iterations of the Leiden algorithm.
     * More iterations can improve separation at the cost of computational time.
     *
     * @return A reference to this `ClusterSNNGraphLeiden` object.
     */
    ClusterSNNGraphLeiden& set_iterations(double i = Defaults::iterations) {
        iterations = i;
        return *this;
    }

    /**
     * @param m Whether to optimize the modularity instead of the Constant Potts Model.
     *
     * The modularity is closely related to the Constant Potts Model, but the magnitude of the resolution is different.
     *
     * @return A reference to this `ClusterSNNGraphLeiden` object.
     */
    ClusterSNNGraphLeiden& set_modularity(double m = Defaults::modularity) {
        modularity = m;
        return *this;
    }

public:
    /**
     * @brief Result of the **igraph** leiden community detection algorithm.
     *
     * Instances should be constructed using the `ClusterSNNGraphLeiden::run()` methods.
     */
    struct Results {
        /** 
         * Output status.
         * A value of zero indicates that the algorithm completed successfully.
         */
        int status;
        
        /**
         * Vector of length equal to the number of cells, containing 0-indexed cluster identities.
         */
        std::vector<int> membership;

        /**
         * Quality of the clustering, closely related to the modularity.
         */
        double quality;
    };

    /**
     * Run the Leiden community detection algorithm on a shared nearest-neighbor graph constructed from an expression matrix.
     *
     * @param ndims Number of dimensions.
     * @param ncells Number of cells.
     * @param mat Pointer to an array of expression values or a low-dimensional representation thereof.
     * Rows should be dimensions while columns should be cells.
     * Data should be stored in column-major format.
     *
     * @return A `Results` object containing the clustering results for all cells.
     */
    Results run(size_t ndims, size_t ncells, const double* mat) {
        auto graph_info = build(ndims, ncells, mat);
        auto output = run(graph_info);
        return output;
    }

    /**
     * Run the Leiden community detection algorithm on a shared nearest-neighbor graph constructed from `knncolle::Base` object.
     *
     * @tparam Algorithm Any instance of a `knncolle::Base` subclass.
     *
     * @param search Pointer to a `knncolle::Base` instance to use for the nearest-neighbor search.
     *
     * @return A `Results` object containing the clustering results for all cells.
     */
    template<class Algorithm>
    Results run(const Algorithm* search) {
        auto graph_info = build(search);
        auto output = run(graph_info);
        return output;
    }

    /**
     * Run the Leiden community detection algorithm on a shared nearest-neighbor graph constructed from `knncolle::Base` object.
     *
     * @param ncells Number of cells.
     * @param store A `deque` of edges, usually generated by a previous call to `BuildSNNGraph::run()`.
     *
     * @return A `Results` object containing the clustering results for all cells.
     */
    Results run(size_t ncells, const std::deque<BuildSNNGraph::WeightedEdge>& store) {
        auto graph_info = build(ncells, store);
        auto output = run(graph_info);
        return output;
    }

    /**
     * Run the Leiden community detection algorithm on a pre-constructed shared nearest-neighbor graph as a `Graph` object.
     *
     * @param graph_info An existing `Graph` object, e.g., constructed by `build()`.
     *
     * @return A `Results` object containing the clustering results for all cells.
     */
    Results run(const Graph& graph_info) {
        IgraphVector membership_holder;
        auto& membership = membership_holder.vector;
        igraph_integer_t nb_clusters;
        igraph_real_t quality;

        IgraphRNGScope rngs(seed);
        Results output;

        if (!modularity) {
            for (int i = 0; i < iterations; ++i) {
                output.status = igraph_community_leiden(graph_info.get_graph(), graph_info.get_weights(), NULL, resolution, beta, (i > 0), &membership, &nb_clusters, &quality);
                if (output.status) {
                    break;
                }
            }
        } else {
            // Based on https://igraph.org/c/doc/igraph-Community.html#igraph_community_leiden.
            IgraphVector degree_holder(igraph_vcount(graph_info.get_graph()));
            auto& degree = degree_holder.vector;
            igraph_degree(graph_info.get_graph(), &degree, igraph_vss_all(), IGRAPH_ALL, 1);

            // This assumes that resolution = 1 in the example in the C documentation. 
            // igraph::cluster_leiden in the R package does the same thing.
            double mod_resolution = resolution / (2 * igraph_ecount(graph_info.get_graph()));
            
            for (int i = 0; i < iterations; ++i) {
                output.status = igraph_community_leiden(graph_info.get_graph(), graph_info.get_weights(), &degree, mod_resolution, beta, (i > 0), &membership, &nb_clusters, &quality);
                if (output.status) {
                    break;
                }
            }
        }

        if (!output.status) {
            size_t ncells = igraph_vcount(graph_info.get_graph());
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

#endif

