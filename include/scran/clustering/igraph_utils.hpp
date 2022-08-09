#ifndef SCRAN_IGRAPH_UTILS_HPP
#define SCRAN_IGRAPH_UTILS_HPP

#ifndef SCRAN_EXCLUDE_IGRAPH
#include "../utils/macros.hpp"

#include "igraph.h"

/**
 * @file igraph_utils.hpp
 *
 * @brief Utilities for manipulating **igraph** data structures.
 */

namespace scran {

namespace igraph {

/**
 * @cond
 */
struct RNGScope {
    RNGScope(int seed) : previous(igraph_rng_default()) {
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
    RNGScope(const RNGScope& other) = delete;
    RNGScope& operator=(const RNGScope& other) = delete;
    RNGScope(RNGScope&& other) = delete;
    RNGScope& operator=(RNGScope&& other) = delete;

    ~RNGScope() {
        if (active) {
            igraph_rng_set_default(previous);
            igraph_rng_destroy(&rng);
        }
    }

    igraph_rng_t* previous;
    igraph_rng_t rng;
    bool active = true;
};

struct Vector {
private:
    static void try_copy(igraph_vector_t& dest, const igraph_vector_t& source, bool source_active) {
        if (source_active) {
            if (igraph_vector_copy(&dest, &source)) {
                throw std::runtime_error("failed to copy igraph vector of size " + std::to_string(igraph_vector_size(&source)));
            }
        }
    }

    static void try_destroy(igraph_vector_t& vector, bool active) {
        if (active) {
            igraph_vector_destroy(&vector);
        }
    }

public:
    Vector(size_t size = 0) {
        if (igraph_vector_init(&vector, size)) {
            throw std::runtime_error("failed to initialize igraph vector of size " + std::to_string(size));
        }
    }
    
    // Just deleting the methods here, because the Vector 
    // is strictly internal and we don't do any of these.
    Vector(const Vector& other) = delete;
    Vector& operator=(const Vector& other) = delete;
    Vector(Vector&& other) = delete;
    Vector& operator=(Vector&& other) = delete;

    ~Vector() {
        try_destroy(vector, active);
    }

    igraph_vector_t vector;
    bool active = true;
};

struct Matrix {
    Matrix(size_t nrows = 0, size_t ncols = 0) {
        if (igraph_matrix_init(&matrix, nrows, ncols)) {
            throw std::runtime_error("failed to initialize igraph " + std::to_string(nrows) + "x" + std::to_string(ncols) + " matrix");
        }
    }

    // Just deleting the methods here, because the Matrix
    // is strictly internal and we don't do any of these.
    Matrix(const Matrix& other) = delete;
    Matrix& operator=(const Matrix& other) = delete;
    Matrix(Matrix&& other) = delete;
    Matrix& operator=(Matrix&& other) = delete;

    ~Matrix() {
        if (active) {
            igraph_matrix_destroy(&matrix);
        }
    }

    igraph_matrix_t matrix;
    bool active = true;
};
/**
 * @endcond
 */

/**
 * @brief Wrapper around the `igraph_t` class from **igraph**.
 */
struct Graph {
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
    /**
     * @cond
     */
    Graph() : active(false) {}

    Graph(const Vector& edges, size_t nvertices, bool directed) : Graph(&(edges.vector), nvertices, directed) {} 

    Graph(const igraph_vector_t* edges, size_t nvertices, bool directed) { 
        if (igraph_create(&graph, edges, nvertices, directed)) {
            throw std::runtime_error("failed to initialize igraph's graph object"); 
        }
    }

    Graph(const Graph& other) : active(other.active) {
        try_copy(graph, other.graph, other.active);
    }

    Graph& operator=(const Graph& other) {
        if (this != &other) {
            try_destroy(graph, active);
            try_copy(graph, other.graph, other.active);
            active = other.active;
        }
        return *this;
    }

    // See https://docs.microsoft.com/en-us/cpp/cpp/move-constructors-and-move-assignment-operators-cpp?view=msvc-170
    Graph(Graph&& other) : graph(std::move(other.graph)), active(other.active) {
        other.active = false;
    }

    Graph& operator=(Graph&& other) {
        if (this != &other) {
            try_destroy(graph, active);
            graph = std::move(other.graph);
            active = other.active;
            other.active = false;
        }
        return *this;
    }

    ~Graph() {
        try_destroy(graph, active);
    }

    bool active = true;

    igraph_t graph;
    /**
     * @cond
     */

    /**
     * @name Get the graph.
     *
     * @return Pointer to an **igraph** graph.
     * Nodes are cells with edges being formed between its nearest neighbors.
     *
     * Users should not pass this pointer to `igraph_destroy`; the `Graph` destructor will handle the freeing.
     */
    const igraph_t* get_graph() const {
        return &graph;
    }
};

}

}

#endif 

#endif
