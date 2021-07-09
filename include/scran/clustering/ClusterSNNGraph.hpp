#ifndef SCRAN_SNNGRAPHCLUSTERING_HPP
#define SCRAN_SNNGRAPHCLUSTERING_HPP

#ifndef SCRAN_EXCLUDE_IGRAPH

#include "BuildSNNGraph.hpp"

#include <vector>
#include <algorithm>

#include "igraph.h"

namespace scran {

class ClusterSNNGraph {
public:
    BuildSNNGraph builder;

    struct Graph {
        igraph_t graph;
        igraph_vector_t weights;
        void destroy() {
            igraph_destroy(&graph);
            igraph_vector_destroy(&weights);
        }
    };

    Graph build(size_t ndims, size_t ncells, const double* mat) const {
        auto store = builder.run(ndims, ncells, mat);
        return build(ncells, store);
    }

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
    struct MultiLevelResult {
        int status;
        size_t max;
        std::vector<std::vector<int> > membership;
        std::vector<double> modularity;
    };

    MultiLevelResult run_multilevel(size_t ndims, size_t ncells, const double* mat, double resolution = 1) {
        auto graph_info = build(ndims, ncells, mat);
        auto output = run_multilevel(graph_info, resolution);
        graph_info.destroy();
        return output;
    }

    MultiLevelResult run_multilevel(const Graph& graph_info, double resolution = 1) {
        igraph_vector_t membership, modularity;
        igraph_matrix_t memberships;

        igraph_vector_init(&modularity, 0);
        igraph_vector_init(&membership, 0);
        igraph_matrix_init(&memberships, 0, 0);

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

